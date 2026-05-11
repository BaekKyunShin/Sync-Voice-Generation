"""Google TTS 본 합성: 5 화자 × 60분 = 합 5시간.

각 화자별로 텍스트 풀(synth_text_pool_clean.csv 13,180발화)에서 random sample →
Google API 호출 → wav duration 측정 → 누적 60분(3,600초) 도달 시 stop.

CLOVA 사용분(synth_full_manifest.csv)은 used_utts에 미리 로드 → 진짜·CLOVA·Google 간
utt_id 중복 0.

화자 모델별 분기:
    Neural2 / Wavenet : speaking_rate 1.0±0.10, pitch_semitones ±1.0
    Chirp3-HD         : 파라미터 미지원 — 텍스트 다양성으로만 분산

합성 직후 첫 30ms 선형 페이드인 → 시작 click 원천 제거.

이어서 실행 가능 (manifest 로드 → 화자별 진행 상태 추적, 일일 한도 초과 시 다음날).

실행:
    # 검증 (1 화자 60초)
    python scripts/synth_full_google.py --target-sec 60 --speakers ko-KR-Neural2-C

    # 본 합성
    python scripts/synth_full_google.py

출력:
    data_kspon/synth_full_google/<voice_short>/<utt_id>.wav (16kHz LINEAR16)
    metadata/splits/synth_full_google_manifest.csv
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from google.api_core import exceptions as gax_exc
from google.cloud import texttospeech

load_dotenv()

ALL_SPEAKERS = [
    "ko-KR-Chirp3-HD-Aoede",
    "ko-KR-Chirp3-HD-Charon",
    "ko-KR-Chirp3-HD-Kore",
    "ko-KR-Neural2-C",
    "ko-KR-Wavenet-C",
]

TEXT_POOL_CSV = Path("metadata/splits/synth_text_pool_clean.csv")
CLOVA_MANIFEST = Path("metadata/splits/synth_full_manifest.csv")
OUT_DIR = Path("data_kspon/synth_full_google")
MANIFEST = Path("metadata/splits/synth_full_google_manifest.csv")
SR = 16000
SEED = 43  # CLOVA(42)와 다르게 → sample 독립
TARGET_SEC_DEFAULT = 60 * 60  # 화자당 60분 = 3,600초
MIN_TEXT_LEN = 10
MAX_TEXT_LEN = 200
FADE_IN_MS = 30

MANIFEST_FIELDS = [
    "utt_id", "speaker", "text", "char_count",
    "speaking_rate", "pitch_semitones",
    "wav_path", "duration_sec",
]


def short_name(voice: str) -> str:
    return voice.replace("ko-KR-", "")


def is_chirp3(voice: str) -> bool:
    return "Chirp3" in voice


def sample_params(seed_str: str, voice: str) -> dict:
    """utt_id+speaker 결정론. Chirp3는 파라미터 미지원 → 빈값."""
    if is_chirp3(voice):
        return {"speaking_rate": None, "pitch_semitones": None}
    rng = random.Random(seed_str)
    rate = round(1.0 + rng.uniform(-0.10, 0.10), 3)
    pitch = round(rng.uniform(-1.0, 1.0), 2)
    return {"speaking_rate": rate, "pitch_semitones": pitch}


def apply_fade_in(out_path: Path, fade_ms: int = FADE_IN_MS) -> None:
    """시작 30ms 선형 fade-in — TTS wav 헤더 click 원천 제거."""
    y, file_sr = sf.read(out_path)
    n = int(file_sr * fade_ms / 1000)
    if n > 0 and len(y) > n:
        ramp = np.linspace(0.0, 1.0, n, dtype="float32")
        if y.ndim == 1:
            y[:n] = (y[:n].astype("float32") * ramp).astype(y.dtype)
        else:
            y[:n, :] = (y[:n, :].astype("float32") * ramp[:, None]).astype(y.dtype)
        sf.write(out_path, y, file_sr, subtype="PCM_16")


def synth_one(client, voice: str, text: str, params: dict,
              out_path: Path) -> tuple[bool, str]:
    """Google TTS 16kHz LINEAR16 직접. 모델별 파라미터 분기."""
    audio_kwargs = dict(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        sample_rate_hertz=SR,
    )
    # Chirp3는 speaking_rate / pitch 미지원
    if not is_chirp3(voice) and params.get("speaking_rate") is not None:
        audio_kwargs["speaking_rate"] = params["speaking_rate"]
        audio_kwargs["pitch"] = params["pitch_semitones"]
    try:
        response = client.synthesize_speech(
            input=texttospeech.SynthesisInput(text=text),
            voice=texttospeech.VoiceSelectionParams(
                language_code="ko-KR", name=voice,
            ),
            audio_config=texttospeech.AudioConfig(**audio_kwargs),
        )
    except gax_exc.GoogleAPIError as e:
        return False, f"api: {str(e)[:120]}"
    except Exception as e:  # noqa: BLE001
        return False, f"err: {str(e)[:120]}"
    out_path.write_bytes(response.audio_content)
    apply_fade_in(out_path)
    return True, ""


def measure_duration(path: Path) -> float:
    info = sf.info(str(path))
    return info.frames / info.samplerate


def load_text_pool(path: Path, min_len: int, max_len: int) -> list[dict]:
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            t = r["text_clean"]
            if min_len <= len(t) <= max_len:
                rows.append({"utt_id": r["utt_id"], "text": t})
    return rows


def write_manifest(rows: list[dict]) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)


def _row(utt_id, voice, text, params, out_path, dur) -> dict:
    return {
        "utt_id": utt_id, "speaker": voice, "text": text, "char_count": len(text),
        "speaking_rate": params["speaking_rate"]
            if params["speaking_rate"] is not None else "",
        "pitch_semitones": params["pitch_semitones"]
            if params["pitch_semitones"] is not None else "",
        "wav_path": str(out_path), "duration_sec": round(dur, 3),
    }


def synth_speaker(client, voice: str, target_sec: float,
                   pool: list[dict], used_utts: set[str],
                   log_every: int = 25,
                   max_consecutive_fail: int = 30) -> list[dict]:
    """한 화자 합성 — target_sec 도달까지. used_utts 제외.

    연속 max_consecutive_fail 건 실패 시 60초 대기 + 1회 재시도.
    """
    short = short_name(voice)
    spk_dir = OUT_DIR / short
    spk_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED + abs(hash(voice)) % (10**8))
    candidates = [r for r in pool if r["utt_id"] not in used_utts]
    rng.shuffle(candidates)

    rows: list[dict] = []
    total_dur = 0.0
    fail = 0
    consec_fail = 0

    for c in candidates:
        if total_dur >= target_sec:
            break
        utt_id, text = c["utt_id"], c["text"]
        out_path = spk_dir / f"{utt_id}.wav"
        params = sample_params(f"{utt_id}__{voice}", voice)

        if out_path.exists():
            try:
                dur = measure_duration(out_path)
                total_dur += dur
                used_utts.add(utt_id)
                rows.append(_row(utt_id, voice, text, params, out_path, dur))
                consec_fail = 0
                continue
            except Exception:
                out_path.unlink(missing_ok=True)

        success, err = synth_one(client, voice, text, params, out_path)
        if not success:
            fail += 1
            consec_fail += 1
            if consec_fail >= max_consecutive_fail:
                print(f"  ⚠ {short}: 연속 {consec_fail}건 실패 — 60초 대기 후 재시도 "
                      f"(최근: {err[:80]})")
                time.sleep(60)
                success_retry, err_retry = synth_one(client, voice, text, params, out_path)
                if not success_retry:
                    print(f"  ✗ {short}: 재시도도 실패 → 화자 break "
                          f"(누적 fail {fail}, 진행 {total_dur/60:.1f}분)")
                    return rows
                consec_fail = 0
                try:
                    dur = measure_duration(out_path)
                except Exception:
                    out_path.unlink(missing_ok=True)
                    continue
                total_dur += dur
                used_utts.add(utt_id)
                rows.append(_row(utt_id, voice, text, params, out_path, dur))
                continue
            if fail % 25 == 0:
                print(f"  ⚠ {short}: 실패 누적 {fail}건 (최근: {err[:80]})")
            continue

        consec_fail = 0
        try:
            dur = measure_duration(out_path)
        except Exception as e:
            fail += 1
            print(f"  ✗ duration fail {short}/{utt_id}: {e}")
            out_path.unlink(missing_ok=True)
            continue

        total_dur += dur
        used_utts.add(utt_id)
        rows.append(_row(utt_id, voice, text, params, out_path, dur))

        if len(rows) % log_every == 0 or total_dur >= target_sec:
            print(
                f"  [{short}] {len(rows)}발화 누적 "
                f"{total_dur/60:.1f}분/{target_sec/60:.0f}분 (fail {fail})"
            )

    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-sec", type=float, default=TARGET_SEC_DEFAULT,
                    help=f"화자당 목표 초 (default {TARGET_SEC_DEFAULT}=60분)")
    ap.add_argument("--speakers", nargs="*", default=None,
                    help="처리할 화자 (default: 전체 5명)")
    ap.add_argument("--max-text-len", type=int, default=MAX_TEXT_LEN)
    ap.add_argument("--min-text-len", type=int, default=MIN_TEXT_LEN)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pool = load_text_pool(TEXT_POOL_CSV, args.min_text_len, args.max_text_len)
    print(f"텍스트 풀: {len(pool)} 발화 ({args.min_text_len}~{args.max_text_len}자)")

    speakers_to_do = ALL_SPEAKERS
    if args.speakers:
        wanted = set(args.speakers)
        speakers_to_do = [v for v in ALL_SPEAKERS if v in wanted]
        if not speakers_to_do:
            print(f"⚠ 매칭 화자 없음: {args.speakers}")
            sys.exit(1)

    # used_utts: CLOVA 사용분 + Google 기존 진행분 모두 제외
    used_utts: set[str] = set()
    if CLOVA_MANIFEST.exists():
        with CLOVA_MANIFEST.open() as f:
            for r in csv.DictReader(f):
                used_utts.add(r["utt_id"])
        print(f"CLOVA 사용 utt 제외: {len(used_utts)}개")

    all_rows: list[dict] = []
    if MANIFEST.exists():
        with MANIFEST.open() as f:
            all_rows = list(csv.DictReader(f))
        prev = {r["utt_id"] for r in all_rows}
        used_utts |= prev
        print(f"기존 Google manifest: {len(all_rows)}행 ({len(prev)} utt 추가 제외)")

    client = texttospeech.TextToSpeechClient()

    for voice in speakers_to_do:
        short = short_name(voice)
        spk_existing = [r for r in all_rows if r["speaker"] == voice]
        spk_dur = sum(float(r["duration_sec"]) for r in spk_existing)
        if spk_dur >= args.target_sec:
            print(f"[{short}] 이미 {spk_dur/60:.1f}분 도달 → skip")
            continue
        remaining = args.target_sec - spk_dur
        print(f"\n[{short}] 시작 — 목표 {remaining/60:.1f}분 남음 "
              f"(기존 {spk_dur/60:.1f}분)")

        new_rows = synth_speaker(client, voice, remaining, pool, used_utts)
        all_rows.extend(new_rows)
        added_dur = sum(r["duration_sec"] for r in new_rows)
        write_manifest(all_rows)
        print(f"[{short}] 완료: +{len(new_rows)}발화 +{added_dur/60:.1f}분 "
              f"(누적 {(spk_dur + added_dur)/60:.1f}분)")

    print("\n=== 최종 요약 ===")
    by_spk: dict[str, dict] = {}
    for r in all_rows:
        by_spk.setdefault(r["speaker"], {"n": 0, "dur": 0.0})
        by_spk[r["speaker"]]["n"] += 1
        by_spk[r["speaker"]]["dur"] += float(r["duration_sec"])
    total_n, total_dur = 0, 0.0
    for s, info in by_spk.items():
        print(f"  {short_name(s):28s} {info['n']:5d}발화 {info['dur']/60:6.1f}분")
        total_n += info["n"]
        total_dur += info["dur"]
    print(f"  {'합계':28s} {total_n:5d}발화 {total_dur/60:.1f}분 ({total_dur/3600:.2f}h)")
    print(f"manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
