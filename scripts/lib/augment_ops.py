"""
오디오 증강 단위 함수 (재사용 라이브러리).

§ 6 양방향 증강 정책의 "합성음 only" 카테고리를 구현.
모두 numpy float32 in/out, sr 유지, 결정론용 rng 인자 받음.
본 합성·학습 파이프라인에서도 재사용 예정.
"""
from __future__ import annotations

import audioop
from pathlib import Path
from typing import Sequence

import librosa
import numpy as np
import soundfile as sf
from scipy import signal


def _rms(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(y**2) + 1e-12))


def _pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Voss-McCartney 근사 1/f 핑크 노이즈."""
    n_rows = 16
    n_cols = max(2, int(np.ceil(np.log2(max(n, 2)))))
    array = rng.standard_normal((n_rows, n_cols))
    cumsum = np.cumsum(array, axis=1)
    out = np.zeros(n)
    for k in range(n_cols):
        repeats = 2**k
        col = np.repeat(cumsum[:, k], repeats)[:n]
        if len(col) < n:
            col = np.pad(col, (0, n - len(col)))
        out += col
    out -= out.mean()
    out /= np.std(out) + 1e-12
    return out.astype(np.float32)


def add_synthetic_noise(
    y: np.ndarray, sr: int, snr_db: float, rng: np.random.Generator
) -> np.ndarray:
    """가우시안+1/f 핑크 mix 노이즈를 SNR 기준으로 더함.

    Args:
        snr_db: 신호 대 잡음비 (dB). 낮을수록 노이즈 강함.
    """
    n = len(y)
    gauss = rng.standard_normal(n).astype(np.float32)
    pink = _pink_noise(n, rng)
    mix_ratio = float(rng.uniform(0.3, 0.7))
    noise = mix_ratio * gauss + (1 - mix_ratio) * pink

    sig_rms = _rms(y)
    noise_rms = _rms(noise)
    target_noise_rms = sig_rms / (10 ** (snr_db / 20))
    if noise_rms > 0:
        noise = noise * (target_noise_rms / noise_rms)

    return (y + noise).astype(np.float32)


def apply_telephony_codec(
    y: np.ndarray, sr: int, rng: np.random.Generator | None = None
) -> np.ndarray:
    """16k → μ-law 8k round-trip → 16k upsample. 전화/보이스피싱 시나리오."""
    if sr != 16000:
        raise ValueError(f"sr=16000 only, got {sr}")

    y_8k = librosa.resample(y, orig_sr=16000, target_sr=8000)
    y_8k = np.clip(y_8k, -1.0, 1.0)
    pcm16 = (y_8k * 32767).astype(np.int16).tobytes()

    ulaw = audioop.lin2ulaw(pcm16, 2)
    pcm16_back = audioop.ulaw2lin(ulaw, 2)

    y_8k_back = np.frombuffer(pcm16_back, dtype=np.int16).astype(np.float32) / 32767.0
    y_16k = librosa.resample(y_8k_back, orig_sr=8000, target_sr=16000)

    # 길이 맞춤
    if len(y_16k) > len(y):
        y_16k = y_16k[: len(y)]
    elif len(y_16k) < len(y):
        y_16k = np.pad(y_16k, (0, len(y) - len(y_16k)))

    return y_16k.astype(np.float32)


def apply_reverb(
    y: np.ndarray,
    sr: int,
    rir_pool: Sequence[np.ndarray],
    wet: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """약한 reverb. wet=0.0(dry) ~ 1.0(full reverb)."""
    if not rir_pool:
        return y
    rir = rir_pool[int(rng.integers(0, len(rir_pool)))]
    wet_signal = signal.fftconvolve(y, rir, mode="full")[: len(y)].astype(np.float32)
    if _rms(wet_signal) > 0:
        wet_signal = wet_signal * (_rms(y) / _rms(wet_signal))
    out = (1 - wet) * y + wet * wet_signal
    return out.astype(np.float32)


def apply_volume_jitter(
    y: np.ndarray,
    n_segments: int,
    gain_db_range: tuple[float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """구간별 gain 변동 (마이크 거리 변화 시뮬레이션)."""
    out = y.copy()
    seg_len = max(1, len(y) // n_segments)
    for i in range(n_segments):
        start = i * seg_len
        end = (i + 1) * seg_len if i < n_segments - 1 else len(y)
        gain_db = float(rng.uniform(*gain_db_range))
        out[start:end] *= 10 ** (gain_db / 20)
    return out.astype(np.float32)


def apply_chain(
    y: np.ndarray,
    sr: int,
    cfg: dict,
    rir_pool: Sequence[np.ndarray] | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[str]]:
    """4 op를 cfg 확률대로 순차 적용.

    cfg 예: {"noise": 0.7, "codec": 0.5, "reverb": 0.4, "volume": 0.6}
    """
    out = y.copy()
    applied: list[str] = []

    if rng.random() < cfg.get("noise", 0):
        snr = float(rng.uniform(10, 20))
        out = add_synthetic_noise(out, sr, snr, rng)
        applied.append(f"noise(snr={snr:.1f})")

    if rng.random() < cfg.get("codec", 0):
        out = apply_telephony_codec(out, sr, rng)
        applied.append("codec")

    if rir_pool and rng.random() < cfg.get("reverb", 0):
        wet = float(rng.uniform(0.10, 0.20))
        out = apply_reverb(out, sr, rir_pool, wet, rng)
        applied.append(f"reverb(wet={wet:.2f})")

    if rng.random() < cfg.get("volume", 0):
        out = apply_volume_jitter(
            out, n_segments=3, gain_db_range=(-4, 4), rng=rng
        )
        applied.append("volume")

    return out, applied


def load_rir_pool(rir_dir: Path, sr: int = 16000) -> list[np.ndarray]:
    """assets/rir/*.wav 풀 로드."""
    rirs: list[np.ndarray] = []
    for p in sorted(Path(rir_dir).glob("*.wav")):
        rir, _ = librosa.load(p, sr=sr)
        rirs.append(rir.astype(np.float32))
    return rirs


# ─── v3: RawBoost-style 카테고리 ──────────────────────────────────
# 학술 근거: Tak et al. 2022 "RawBoost" (arxiv 2111.04433)
#   - linear convolutive (channel/mic EQ)
#   - non-linear convolutive (amplifier saturation)
#   - impulsive additive (clicks)
#   - stationary additive (ambient real noise)


def apply_iir_eq(y: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    """linear convolutive: 마이크/채널 EQ 시뮬레이션 (random shelf 또는 telephony bandpass)."""
    mode = rng.choice(["high_shelf", "low_shelf", "bandpass_phone", "bandpass_wide"])
    if mode == "high_shelf":
        # 고음 살짝 깎기 (저가 마이크)
        cutoff = float(rng.uniform(4000, 7000))
        sos = signal.butter(2, cutoff, btype="lowpass", fs=sr, output="sos")
        out = signal.sosfilt(sos, y)
        # 약하게 mix (완전 컷 X)
        out = 0.7 * y + 0.3 * out
    elif mode == "low_shelf":
        # 저음 살짝 깎기
        cutoff = float(rng.uniform(80, 200))
        sos = signal.butter(2, cutoff, btype="highpass", fs=sr, output="sos")
        out = signal.sosfilt(sos, y)
    elif mode == "bandpass_phone":
        # 전화 대역 (협소)
        sos = signal.butter(4, [300, 3400], btype="bandpass", fs=sr, output="sos")
        out = signal.sosfilt(sos, y)
    else:  # bandpass_wide
        # 일반 마이크 대역
        low = float(rng.uniform(60, 120))
        high = float(rng.uniform(6500, 7800))
        sos = signal.butter(4, [low, high], btype="bandpass", fs=sr, output="sos")
        out = signal.sosfilt(sos, y)
    return out.astype(np.float32)


def apply_soft_clip(y: np.ndarray, rng: np.random.Generator,
                    threshold_range: tuple[float, float] = (0.85, 0.97)) -> np.ndarray:
    """non-linear convolutive: 약한 amplifier saturation (tanh)."""
    threshold = float(rng.uniform(*threshold_range))
    # tanh 기반 soft knee — threshold 위에서만 휨
    out = np.where(np.abs(y) > threshold,
                   np.sign(y) * (threshold + (1 - threshold) * np.tanh((np.abs(y) - threshold) / (1 - threshold))),
                   y)
    return out.astype(np.float32)


def add_impulses(y: np.ndarray, sr: int, rng: np.random.Generator,
                  n_range: tuple[int, int] = (3, 12),
                  amp_range: tuple[float, float] = (0.05, 0.20)) -> np.ndarray:
    """impulsive additive: sparse 클릭/팝 (키보드, 마우스 등)."""
    n_impulses = int(rng.integers(*n_range))
    out = y.copy()
    for _ in range(n_impulses):
        pos = int(rng.integers(0, len(y)))
        amp = float(rng.uniform(*amp_range)) * (1 if rng.random() < 0.5 else -1)
        # 짧은 임펄스 (1~3 sample)
        width = int(rng.integers(1, 4))
        end = min(pos + width, len(y))
        out[pos:end] += amp
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def add_real_noise(y: np.ndarray, sr: int,
                    noise_pool: list[np.ndarray], snr_db: float,
                    rng: np.random.Generator) -> np.ndarray:
    """stationary additive: real noise pool에서 segment 잘라 SNR 기준 mix."""
    if not noise_pool:
        # fallback: 합성 노이즈
        return add_synthetic_noise(y, sr, snr_db, rng)
    noise = noise_pool[int(rng.integers(0, len(noise_pool)))]
    # 길이 맞추기 (랜덤 시작점)
    if len(noise) < len(y):
        # 부족하면 wrap-around
        reps = (len(y) // len(noise)) + 1
        noise = np.tile(noise, reps)[: len(y)]
    else:
        start = int(rng.integers(0, len(noise) - len(y) + 1))
        noise = noise[start: start + len(y)]
    noise = noise.astype(np.float32)

    sig_rms = _rms(y)
    noise_rms = _rms(noise)
    if noise_rms > 0:
        target_noise_rms = sig_rms / (10 ** (snr_db / 20))
        noise = noise * (target_noise_rms / noise_rms)
    return (y + noise).astype(np.float32)


def apply_mp3_codec(y: np.ndarray, sr: int, bitrate: str = "96k") -> np.ndarray:
    """mp3 round-trip (pydub + ffmpeg). 약한 압축 노이즈."""
    import io
    import tempfile
    from pydub import AudioSegment

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fwav:
        wav_path = fwav.name
    sf.write(wav_path, np.clip(y, -1.0, 1.0), sr, subtype="PCM_16")

    seg = AudioSegment.from_wav(wav_path)
    mp3_buf = io.BytesIO()
    seg.export(mp3_buf, format="mp3", bitrate=bitrate)
    mp3_buf.seek(0)
    seg_back = AudioSegment.from_file(mp3_buf, format="mp3")

    samples = np.array(seg_back.get_array_of_samples(), dtype=np.int16).astype(np.float32) / 32768.0
    if seg_back.channels == 2:
        samples = samples.reshape(-1, 2).mean(axis=1)
    if len(samples) > len(y):
        samples = samples[: len(y)]
    elif len(samples) < len(y):
        samples = np.pad(samples, (0, len(y) - len(samples)))
    Path(wav_path).unlink(missing_ok=True)
    return samples.astype(np.float32)


def apply_volume_smooth_walk(y: np.ndarray, gain_db_range: tuple[float, float],
                              rng: np.random.Generator,
                              n_anchors: int = 6) -> np.ndarray:
    """smooth random walk envelope (step jitter 대체, 부드러운 마이크 거리 변화)."""
    n = len(y)
    anchor_pos = np.linspace(0, n, n_anchors).astype(int)
    anchor_db = rng.uniform(*gain_db_range, size=n_anchors).astype(np.float32)
    # 선형 보간으로 부드러운 envelope
    envelope_db = np.interp(np.arange(n), anchor_pos, anchor_db)
    envelope = 10 ** (envelope_db / 20)
    return (y * envelope).astype(np.float32)


# strength preset (light dominant 정책 반영)
V3_PRESETS = {
    "light": {
        "snr": (25, 35), "wet": (0.03, 0.08), "vol_db": (-2.0, 2.0),
        "p_eq": 0.5, "p_clip": 0.0, "p_impulse": 0.0,
        "p_noise": 0.7, "p_reverb": 0.4,
        "p_codec": 0.3, "codec_kind": "mp3_128",
    },
    "medium": {
        # light와 청취 격차 ↑ — SNR 더 낮춤·noise/reverb/codec 모두 100%·코덱 mp3_64로 강화
        "snr": (13, 18), "wet": (0.12, 0.20), "vol_db": (-3.5, 3.5),
        "p_eq": 0.8, "p_clip": 0.3, "p_impulse": 0.2,
        "p_noise": 1.0, "p_reverb": 0.85,
        "p_codec": 0.9, "codec_kind": "mp3_64",
    },
    "heavy": {
        "snr": (12, 18), "wet": (0.15, 0.22), "vol_db": (-4.0, 4.0),
        "p_eq": 0.7, "p_clip": 0.4, "p_impulse": 0.3,
        "p_noise": 0.9, "p_reverb": 0.6,
        "p_codec": 0.7, "codec_kind": "ulaw",
    },
}


def apply_chain_v3(
    y: np.ndarray, sr: int, strength: str,
    *, real_noise_pool: list[np.ndarray] | None,
    rir_pool: list[np.ndarray] | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[str]]:
    """v3 chain: RawBoost 4 카테고리 + 코덱 + smooth volume.

    strength: "light" | "medium" | "heavy"
    """
    cfg = V3_PRESETS[strength]
    out = y.astype(np.float32)
    applied: list[str] = []

    # 1) linear convolutive (EQ/필터 = 마이크/채널)
    if rng.random() < cfg["p_eq"]:
        out = apply_iir_eq(out, sr, rng)
        applied.append("eq")

    # 2) stationary additive noise (real if available)
    if rng.random() < cfg["p_noise"]:
        snr = float(rng.uniform(*cfg["snr"]))
        if real_noise_pool:
            out = add_real_noise(out, sr, real_noise_pool, snr, rng)
            applied.append(f"realnoise(snr={snr:.1f})")
        else:
            out = add_synthetic_noise(out, sr, snr, rng)
            applied.append(f"synthnoise(snr={snr:.1f})")

    # 3) reverb (room reflection)
    if rir_pool and rng.random() < cfg["p_reverb"]:
        wet = float(rng.uniform(*cfg["wet"]))
        out = apply_reverb(out, sr, rir_pool, wet, rng)
        applied.append(f"reverb(wet={wet:.2f})")

    # 4) impulsive additive (clicks/pops)
    if rng.random() < cfg["p_impulse"]:
        out = add_impulses(out, sr, rng)
        applied.append("impulse")

    # 5) non-linear convolutive (soft clipping = amp saturation)
    if rng.random() < cfg["p_clip"]:
        out = apply_soft_clip(out, rng)
        applied.append("clip")

    # 6) codec (mp3 또는 μ-law)
    if rng.random() < cfg["p_codec"]:
        kind = cfg["codec_kind"]
        if kind == "mp3_128":
            out = apply_mp3_codec(out, sr, bitrate="128k")
            applied.append("mp3_128")
        elif kind == "mp3_96":
            out = apply_mp3_codec(out, sr, bitrate="96k")
            applied.append("mp3_96")
        elif kind == "ulaw":
            out = apply_telephony_codec(out, sr, rng)
            applied.append("ulaw")

    # 7) smooth volume walk (마이크 거리 변화)
    out = apply_volume_smooth_walk(out, cfg["vol_db"], rng)
    applied.append("vol_smooth")

    return np.clip(out, -1.0, 1.0).astype(np.float32), applied


def load_real_noise_pool(noise_dir: Path, sr: int = 16000,
                          max_files: int = 30, max_dur_sec: float = 10.0) -> list[np.ndarray]:
    """real noise wav 풀 로드 (MUSAN noise/* 또는 freesound 등).

    max_files: 메모리 절약 위해 제한
    max_dur_sec: 한 wav 길이 제한 (필요 만큼만 segment 사용)
    """
    pool: list[np.ndarray] = []
    if not noise_dir.exists():
        return pool
    files = sorted(noise_dir.rglob("*.wav"))[:max_files]
    for p in files:
        try:
            y, _ = librosa.load(p, sr=sr, duration=max_dur_sec, mono=True)
            if len(y) > sr * 0.5:  # 0.5초 이상만
                pool.append(y.astype(np.float32))
        except Exception:
            continue
    return pool
