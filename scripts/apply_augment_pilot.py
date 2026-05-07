"""
Pilot 후처리 증강: 합성 wav × 5 변형 출력.

각 16kHz 합성 wav에 대해:
    noise / codec / reverb / volume / chain  5종 변형 wav를 만든다.
chain은 4 op를 확률적으로 순차 적용 (평균 ~2.2 op).

§ 6 양방향 증강 정책의 "합성음 only" 카테고리 검증용.

실행:
    python scripts/apply_augment_pilot.py

전제:
    scripts/synth_pilot.py 가 먼저 실행되어 manifest가 있어야 함.
    assets/rir/ 에 합성 IR이 있어야 함 (build_synthetic_rirs.py).

출력:
    data_kspon/synth_pilot_aug/<variant>/<spk>__<utt_id>.wav
    metadata/splits/synth_pilot_aug_manifest.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from lib.augment_ops import (  # noqa: E402
    add_synthetic_noise,
    apply_chain,
    apply_reverb,
    apply_telephony_codec,
    apply_volume_jitter,
    load_rir_pool,
)

INPUT_MANIFEST = Path("metadata/splits/synth_pilot_manifest.csv")
RIR_DIR = Path("assets/rir")
OUT_DIR = Path("data_kspon/synth_pilot_aug")
OUT_MANIFEST = Path("metadata/splits/synth_pilot_aug_manifest.csv")
SEED = 42
SR = 16000

CHAIN_CFG = {"noise": 0.7, "codec": 0.5, "reverb": 0.4, "volume": 0.6}

OUT_FIELDS = ["utt_id", "speaker", "variant", "applied_ops",
              "snr_db", "rir_id", "wav_path"]


def write_wav(path: Path, y: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.clip(y, -1.0, 1.0), sr, subtype="PCM_16")


def main() -> None:
    rir_pool = load_rir_pool(RIR_DIR, sr=SR)
    print(f"RIR pool: {len(rir_pool)}개 로드")

    for variant in ["noise", "codec", "reverb", "volume", "chain"]:
        (OUT_DIR / variant).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    rows_out: list[dict] = []

    with INPUT_MANIFEST.open() as f:
        rows = list(csv.DictReader(f))

    print(f"입력 wav: {len(rows)}개")

    for row in rows:
        utt_id = row["utt_id"]
        spk = row["speaker"]
        wav16_path = Path(row["wav16_path"])
        y, _ = librosa.load(wav16_path, sr=SR)

        # 1) noise
        snr = float(rng.uniform(10, 20))
        y_n = add_synthetic_noise(y, SR, snr, rng)
        out = OUT_DIR / "noise" / f"{spk}__{utt_id}.wav"
        write_wav(out, y_n, SR)
        rows_out.append({"utt_id": utt_id, "speaker": spk, "variant": "noise",
                         "applied_ops": f"noise(snr={snr:.1f})",
                         "snr_db": round(snr, 1), "rir_id": "",
                         "wav_path": str(out)})

        # 2) codec
        y_c = apply_telephony_codec(y, SR, rng)
        out = OUT_DIR / "codec" / f"{spk}__{utt_id}.wav"
        write_wav(out, y_c, SR)
        rows_out.append({"utt_id": utt_id, "speaker": spk, "variant": "codec",
                         "applied_ops": "codec(μ-law 8k round-trip)",
                         "snr_db": "", "rir_id": "",
                         "wav_path": str(out)})

        # 3) reverb
        wet = float(rng.uniform(0.10, 0.20))
        rir_idx = int(rng.integers(0, len(rir_pool)))
        y_r = apply_reverb(y, SR, [rir_pool[rir_idx]], wet, rng)
        out = OUT_DIR / "reverb" / f"{spk}__{utt_id}.wav"
        write_wav(out, y_r, SR)
        rows_out.append({"utt_id": utt_id, "speaker": spk, "variant": "reverb",
                         "applied_ops": f"reverb(wet={wet:.2f})",
                         "snr_db": "", "rir_id": f"rir_{rir_idx:02d}",
                         "wav_path": str(out)})

        # 4) volume
        y_v = apply_volume_jitter(y, n_segments=3, gain_db_range=(-4, 4), rng=rng)
        out = OUT_DIR / "volume" / f"{spk}__{utt_id}.wav"
        write_wav(out, y_v, SR)
        rows_out.append({"utt_id": utt_id, "speaker": spk, "variant": "volume",
                         "applied_ops": "volume(±4dB,3seg)",
                         "snr_db": "", "rir_id": "",
                         "wav_path": str(out)})

        # 5) chain
        y_chain, applied = apply_chain(y, SR, CHAIN_CFG, rir_pool, rng)
        out = OUT_DIR / "chain" / f"{spk}__{utt_id}.wav"
        write_wav(out, y_chain, SR)
        rows_out.append({"utt_id": utt_id, "speaker": spk, "variant": "chain",
                         "applied_ops": "+".join(applied) if applied else "none",
                         "snr_db": "", "rir_id": "",
                         "wav_path": str(out)})

        print(f"  ✓ {spk}/{utt_id}: 5 variants")

    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with OUT_MANIFEST.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"\n총 {len(rows_out)} 변형 wav 생성: {OUT_DIR}")
    print(f"manifest: {OUT_MANIFEST}")


if __name__ == "__main__":
    main()
