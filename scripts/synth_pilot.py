"""
Pilot 합성: vdonghyun + vhyeri 두 화자 × 25발화 = 50 wav.

§ 5(자연스러운 일상 대화 톤)을 통과한 두 화자만으로
TTS 옵션 다양화(speed/pitch/volume/emotion) + 망설임 텍스트 변형을 적용해
본 합성(7h × 2 화자) 진입 전 작은 배치를 검증.

실행:
    python scripts/synth_pilot.py

출력:
    data_kspon/synth_pilot/<spk>/<utt_id>.wav         (24kHz CLOVA 원본)
    data_kspon/synth_pilot_16k/<spk>/<utt_id>.wav     (16kHz, 증강 입력)
    metadata/splits/synth_pilot_manifest.csv          (TTS 파라미터 + 망설임 로그)
"""
from __future__ import annotations

import csv
import os
import random
import sys
from pathlib import Path

import librosa
import requests
import soundfile as sf
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from lib.hesitation_inject import maybe_inject_hesitation  # noqa: E402

load_dotenv()
CLIENT_ID = os.environ["CLOVA_CLIENT_ID"]
CLIENT_SECRET = os.environ["CLOVA_CLIENT_SECRET"]

URL = "https://naveropenapi.apigw.ntruss.com/tts-premium/v1/tts"
HEADERS = {
    "X-NCP-APIGW-API-KEY-ID": CLIENT_ID,
    "X-NCP-APIGW-API-KEY": CLIENT_SECRET,
    "Content-Type": "application/x-www-form-urlencoded",
}

SPEAKERS = ["vdonghyun", "vhyeri"]
N_PER_SPEAKER = 25
SEED = 42

TEXT_POOL_CSV = Path("metadata/splits/synth_text_pool_clean.csv")
OUT_24K = Path("data_kspon/synth_pilot")
OUT_16K = Path("data_kspon/synth_pilot_16k")
MANIFEST = Path("metadata/splits/synth_pilot_manifest.csv")

MANIFEST_FIELDS = [
    "utt_id", "speaker", "text_clean", "text_synth", "hesitation_applied",
    "speed", "pitch", "volume", "emotion", "emotion_strength",
    "wav24_path", "wav16_path", "duration_sec",
]


def sample_texts() -> list[dict]:
    """text_clean 길이 15~120자에서 랜덤 N_PER_SPEAKER 추출 (seed=42)."""
    rng = random.Random(SEED)
    rows = []
    with TEXT_POOL_CSV.open() as f:
        for row in csv.DictReader(f):
            if 15 <= len(row["text_clean"]) <= 120:
                rows.append(row)
    rng.shuffle(rows)
    return rows[:N_PER_SPEAKER]


def sample_params(seed_str: str) -> dict:
    """utt_id+speaker 기반 결정론. emotion≠0일 때만 strength 부여 (CLOVA 제약)."""
    rng = random.Random(seed_str)
    speed = rng.choice([-2, -1, 0, 1, 2])
    pitch = rng.choice([-1, 0, 1])
    volume = rng.choice([-1, 0, 1])
    if rng.random() < 0.2:
        # 약한 기쁨 (strength=1, 너무 들뜨지 않게)
        emotion, emotion_strength = 2, 1
    else:
        # 중립 — emotion 파라미터 자체 생략
        emotion, emotion_strength = 0, None
    return {
        "speed": speed, "pitch": pitch, "volume": volume,
        "emotion": emotion, "emotion_strength": emotion_strength,
    }


def synth_one(speaker: str, text: str, params: dict, out_path: Path) -> tuple[bool, str]:
    data = {"speaker": speaker, "text": text, "format": "wav",
            "speed": str(params["speed"]),
            "pitch": str(params["pitch"]),
            "volume": str(params["volume"])}
    # emotion은 ≠0일 때만 보냄. strength도 그때만.
    if params["emotion"] != 0:
        data["emotion"] = str(params["emotion"])
        if params["emotion_strength"] is not None:
            data["emotion-strength"] = str(params["emotion_strength"])
    res = requests.post(URL, headers=HEADERS, data=data, timeout=30)
    if res.status_code != 200:
        return False, f"{res.status_code} {res.text[:120]}"
    out_path.write_bytes(res.content)
    return True, ""


def to_16k(src: Path, dst: Path) -> float:
    y, _ = librosa.load(src, sr=16000)
    sf.write(dst, y, 16000, subtype="PCM_16")
    return len(y) / 16000


def main() -> None:
    OUT_24K.mkdir(parents=True, exist_ok=True)
    OUT_16K.mkdir(parents=True, exist_ok=True)
    for spk in SPEAKERS:
        (OUT_24K / spk).mkdir(parents=True, exist_ok=True)
        (OUT_16K / spk).mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    texts = sample_texts()
    print(f"샘플 텍스트 {len(texts)}개 추출 (text_clean 15-120자, seed={SEED})")

    rng_hes = random.Random(SEED)

    rows_out: list[dict] = []
    ok, fail = 0, 0
    total_chars = 0

    for spk in SPEAKERS:
        for row in texts:
            utt_id = row["utt_id"]
            text_clean = row["text_clean"]

            text_synth, hesitation_applied = maybe_inject_hesitation(
                text_clean, rng_hes, apply_rate=0.5
            )
            params = sample_params(f"{utt_id}__{spk}")

            f24 = OUT_24K / spk / f"{utt_id}.wav"
            f16 = OUT_16K / spk / f"{utt_id}.wav"
            if f16.exists():
                print(f"  - skip {spk}/{utt_id}")
                continue

            success, err = synth_one(spk, text_synth, params, f24)
            if not success:
                print(f"  ✗ {spk}/{utt_id}: {err}")
                fail += 1
                continue

            duration = to_16k(f24, f16)
            ok += 1
            total_chars += len(text_synth)
            print(
                f"  ✓ {spk}/{utt_id} ({duration:.2f}s, "
                f"hes={int(hesitation_applied)}, "
                f"sp={params['speed']:+d} pi={params['pitch']:+d} "
                f"vo={params['volume']:+d} em={params['emotion']})"
            )

            rows_out.append({
                "utt_id": utt_id,
                "speaker": spk,
                "text_clean": text_clean,
                "text_synth": text_synth,
                "hesitation_applied": int(hesitation_applied),
                "speed": params["speed"],
                "pitch": params["pitch"],
                "volume": params["volume"],
                "emotion": params["emotion"],
                "emotion_strength": params["emotion_strength"] if params["emotion_strength"] is not None else "",
                "wav24_path": str(f24),
                "wav16_path": str(f16),
                "duration_sec": round(duration, 3),
            })

    if rows_out:
        with MANIFEST.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(rows_out)

    print(f"\n결과: 성공 {ok} / 실패 {fail}")
    print(f"총 합성 글자수: {total_chars} (CLOVA 무료 한도 100만자의 {total_chars/10000:.2f}%)")
    print(f"24kHz: {OUT_24K}/<spk>/<utt_id>.wav")
    print(f"16kHz: {OUT_16K}/<spk>/<utt_id>.wav")
    print(f"manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
