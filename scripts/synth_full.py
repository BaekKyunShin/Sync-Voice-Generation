"""
CLOVA 본 합성: 8 화자 × 53분 = 합 ~7시간.

각 화자별로 텍스트 풀(synth_text_pool_clean.csv 13,180발화)에서 random sample →
CLOVA API 호출 → wav duration 측정 → 누적이 53분(3,180초) 도달 시 stop.

TTS 파라미터는 발화 단위 random:
  speed/pitch/volume/alpha ±3, end-pitch ±2,
  emotion: 80% neutral / 20% 약한 기쁨(strength=1)

화자 간 텍스트 중복 0 (이미 다른 화자가 쓴 utt_id 제외).
이어서 실행 가능 (기존 manifest 로드 → 화자별 진행 상태 추적).

실행:
    # 검증용 소규모 (1 화자 60초)
    python scripts/synth_full.py --target-sec 60 --speakers vdonghyun

    # 본격 합성 (8 화자 × 53분)
    python scripts/synth_full.py

출력:
    data_kspon/synth_full/<speaker>/<utt_id>.wav (16kHz)
    metadata/splits/synth_full_manifest.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path

import requests
import soundfile as sf
from dotenv import load_dotenv

load_dotenv()

URL = "https://naveropenapi.apigw.ntruss.com/tts-premium/v1/tts"
HEADERS = {
    "X-NCP-APIGW-API-KEY-ID": os.environ["CLOVA_CLIENT_ID"],
    "X-NCP-APIGW-API-KEY": os.environ["CLOVA_CLIENT_SECRET"],
    "Content-Type": "application/x-www-form-urlencoded",
}

ALL_SPEAKERS = [
    ("동현 Pro", "vdonghyun"),
    ("유나 Pro", "vyuna"),
    ("혜리 Pro", "vhyeri"),
    ("드림", "njangj"),
    ("박리뷰", "nreview"),
    ("상도", "nsangdo"),
    ("승표", "nseungpyo"),
    ("지환", "njihwan"),
]

TEXT_POOL_CSV = Path("metadata/splits/synth_text_pool_clean.csv")
OUT_DIR = Path("data_kspon/synth_full")
MANIFEST = Path("metadata/splits/synth_full_manifest.csv")
SEED = 42
TARGET_SEC_DEFAULT = 53 * 60  # 화자당 3,180초
MIN_TEXT_LEN = 10   # 너무 짧으면 다양성 ↓
MAX_TEXT_LEN = 200  # CLOVA 한 호출 한도·짧은 발화 분포 매칭

MANIFEST_FIELDS = [
    "utt_id", "speaker", "text", "char_count",
    "speed", "pitch", "volume", "alpha", "end_pitch",
    "emotion", "emotion_strength",
    "wav_path", "duration_sec",
]


def sample_params(seed_str: str) -> dict:
    """utt_id+speaker 기반 결정론."""
    rng = random.Random(seed_str)
    speed = rng.randint(-3, 3)
    pitch = rng.randint(-3, 3)
    volume = rng.randint(-3, 3)
    alpha = rng.randint(-3, 3)
    end_pitch = rng.randint(-2, 2)
    if rng.random() < 0.2:
        emotion, emotion_strength = 2, 1
    else:
        emotion, emotion_strength = 0, None
    return {
        "speed": speed, "pitch": pitch, "volume": volume,
        "alpha": alpha, "end_pitch": end_pitch,
        "emotion": emotion, "emotion_strength": emotion_strength,
    }


def synth_one(speaker: str, text: str, params: dict, out_path: Path) -> tuple[bool, str]:
    """CLOVA Premium 16kHz wav 직접. emotion-strength는 emotion≠0일 때만."""
    data = {
        "speaker": speaker, "text": text, "format": "wav",
        "sampling-rate": "16000",
        "speed": str(params["speed"]),
        "pitch": str(params["pitch"]),
        "volume": str(params["volume"]),
        "alpha": str(params["alpha"]),
        "end-pitch": str(params["end_pitch"]),
    }
    if params["emotion"] != 0:
        data["emotion"] = str(params["emotion"])
        if params["emotion_strength"] is not None:
            data["emotion-strength"] = str(params["emotion_strength"])
    try:
        res = requests.post(URL, headers=HEADERS, data=data, timeout=30)
    except requests.RequestException as e:
        return False, f"network: {e}"
    if res.status_code != 200:
        return False, f"{res.status_code} {res.text[:120]}"
    out_path.write_bytes(res.content)
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


def synth_speaker(spk: str, label: str, target_sec: float,
                   pool: list[dict], used_utts: set[str],
                   log_every: int = 25,
                   max_consecutive_fail: int = 30) -> list[dict]:
    """한 화자 합성 — target_sec 도달까지. 다른 화자 사용 utt 제외.

    연속 max_consecutive_fail 건 실패 시 네트워크 이상으로 판단,
    60초 대기 후 1회 재시도. 그래도 실패면 화자 break (다음 화자로).
    """
    import time

    spk_dir = OUT_DIR / spk
    spk_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED + abs(hash(spk)) % (10**8))
    candidates = [r for r in pool if r["utt_id"] not in used_utts]
    rng.shuffle(candidates)

    rows: list[dict] = []
    total_dur = 0.0
    fail = 0
    consec_fail = 0

    for i, c in enumerate(candidates):
        if total_dur >= target_sec:
            break
        utt_id, text = c["utt_id"], c["text"]
        out_path = spk_dir / f"{utt_id}.wav"
        if out_path.exists():
            try:
                dur = measure_duration(out_path)
                total_dur += dur
                used_utts.add(utt_id)
                params = sample_params(f"{utt_id}__{spk}")
                rows.append(_row(utt_id, spk, text, params, out_path, dur))
                consec_fail = 0
                continue
            except Exception:
                out_path.unlink(missing_ok=True)

        params = sample_params(f"{utt_id}__{spk}")
        success, err = synth_one(spk, text, params, out_path)
        if not success:
            fail += 1
            consec_fail += 1
            if consec_fail >= max_consecutive_fail:
                print(f"  ⚠ {label} {spk}: 연속 {consec_fail}건 실패 — "
                      f"60초 대기 후 1회 재시도 (최근: {err[:80]})")
                time.sleep(60)
                # 재시도 1회
                success_retry, err_retry = synth_one(spk, text, params, out_path)
                if not success_retry:
                    print(f"  ✗ {label} {spk}: 재시도도 실패 → 화자 break "
                          f"(누적 fail {fail}, 진행 분량 {total_dur/60:.1f}분)")
                    return rows
                # 성공 → 처리
                consec_fail = 0
                try:
                    dur = measure_duration(out_path)
                except Exception:
                    out_path.unlink(missing_ok=True)
                    continue
                total_dur += dur
                used_utts.add(utt_id)
                rows.append(_row(utt_id, spk, text, params, out_path, dur))
                continue
            if fail % 25 == 0:
                print(f"  ⚠ {label} {spk}: 실패 누적 {fail}건 (최근: {err[:80]})")
            continue

        consec_fail = 0
        try:
            dur = measure_duration(out_path)
        except Exception as e:
            fail += 1
            print(f"  ✗ duration fail {spk}/{utt_id}: {e}")
            out_path.unlink(missing_ok=True)
            continue

        total_dur += dur
        used_utts.add(utt_id)
        rows.append(_row(utt_id, spk, text, params, out_path, dur))

        if (len(rows)) % log_every == 0 or total_dur >= target_sec:
            print(
                f"  [{label} {spk}] {len(rows)}발화 누적 "
                f"{total_dur/60:.1f}분/{target_sec/60:.0f}분 (fail {fail})"
            )

    return rows


def _row(utt_id, spk, text, params, out_path, dur) -> dict:
    return {
        "utt_id": utt_id, "speaker": spk, "text": text, "char_count": len(text),
        "speed": params["speed"], "pitch": params["pitch"], "volume": params["volume"],
        "alpha": params["alpha"], "end_pitch": params["end_pitch"],
        "emotion": params["emotion"],
        "emotion_strength": params["emotion_strength"]
        if params["emotion_strength"] is not None else "",
        "wav_path": str(out_path), "duration_sec": round(dur, 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-sec", type=float, default=TARGET_SEC_DEFAULT,
                    help=f"화자당 목표 초 (default {TARGET_SEC_DEFAULT}=53분)")
    ap.add_argument("--speakers", nargs="*", default=None,
                    help="처리할 화자 ID (default: 전체 8명)")
    ap.add_argument("--max-text-len", type=int, default=MAX_TEXT_LEN)
    ap.add_argument("--min-text-len", type=int, default=MIN_TEXT_LEN)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pool = load_text_pool(TEXT_POOL_CSV, args.min_text_len, args.max_text_len)
    print(f"텍스트 풀: {len(pool)} 발화 ({args.min_text_len}~{args.max_text_len}자)")

    speakers_to_do = ALL_SPEAKERS
    if args.speakers:
        wanted = set(args.speakers)
        speakers_to_do = [(l, s) for l, s in ALL_SPEAKERS if s in wanted]
        if not speakers_to_do:
            print(f"⚠ 매칭 화자 없음: {args.speakers}")
            sys.exit(1)

    # 기존 manifest 로드 (resume 지원)
    all_rows: list[dict] = []
    used_utts: set[str] = set()
    if MANIFEST.exists():
        with MANIFEST.open() as f:
            all_rows = list(csv.DictReader(f))
        used_utts = {r["utt_id"] for r in all_rows}
        print(f"기존 manifest: {len(all_rows)}행 (이미 사용 utt {len(used_utts)}개)")

    for label, spk in speakers_to_do:
        spk_existing = [r for r in all_rows if r["speaker"] == spk]
        spk_dur = sum(float(r["duration_sec"]) for r in spk_existing)
        if spk_dur >= args.target_sec:
            print(f"[{label} {spk}] 이미 {spk_dur/60:.1f}분 도달 → skip")
            continue
        remaining = args.target_sec - spk_dur
        print(
            f"\n[{label} {spk}] 시작 — 목표 {remaining/60:.1f}분 남음 "
            f"(기존 {spk_dur/60:.1f}분)"
        )

        new_rows = synth_speaker(spk, label, remaining, pool, used_utts)
        all_rows.extend(new_rows)
        added_dur = sum(r["duration_sec"] for r in new_rows)
        # 화자 끝날 때마다 manifest 즉시 저장 (실패 대비 안전망)
        write_manifest(all_rows)
        print(f"[{label} {spk}] 완료: +{len(new_rows)}발화 +{added_dur/60:.1f}분 "
              f"(누적 화자분 {(spk_dur + added_dur)/60:.1f}분)")

    # 최종 요약
    print("\n=== 최종 요약 ===")
    by_spk: dict[str, dict] = {}
    for r in all_rows:
        by_spk.setdefault(r["speaker"], {"n": 0, "dur": 0.0})
        by_spk[r["speaker"]]["n"] += 1
        by_spk[r["speaker"]]["dur"] += float(r["duration_sec"])
    total_n, total_dur = 0, 0.0
    for s, info in by_spk.items():
        print(f"  {s:12s} {info['n']:5d}발화 {info['dur']/60:6.1f}분")
        total_n += info["n"]
        total_dur += info["dur"]
    print(f"  {'합계':12s} {total_n:5d}발화 {total_dur/60:.1f}분 ({total_dur/3600:.2f}h)")
    print(f"manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
