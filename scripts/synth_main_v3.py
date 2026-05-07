"""
본 합성 v3 (pilot에서 sanity 통과 후): P1 페어 disfluent 텍스트 × 9 화자 = 18 wav.

P1 disfluent 텍스트 (모든 화자 동일 — 화자 간 비교 공정성):
    A: "어, 어제 말씀하신 거 있잖아요오. 그게 좀, 그게 급한 거 같아서...
        혹시 통화 가능하세요?"
        — filler + lengthening + repetition + mid-pause
    B: "아... 진짜 너무 힘드네요오. 어떻게 해야 좋을지... 모르겠어요."
        — filler + mid-pause + lengthening

CLOVA `sampling-rate=16000` 직접 사용 → 24kHz→16kHz 리샘플 단계 생략.

실행:
    python scripts/synth_main_v3.py

출력:
    data_kspon/synth_main_v3/<spk>__<varA|varB>.wav   (16kHz CLOVA 직접)
    metadata/splits/synth_main_v3_manifest.csv
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()
URL = "https://naveropenapi.apigw.ntruss.com/tts-premium/v1/tts"
HEADERS = {
    "X-NCP-APIGW-API-KEY-ID": os.environ["CLOVA_CLIENT_ID"],
    "X-NCP-APIGW-API-KEY": os.environ["CLOVA_CLIENT_SECRET"],
    "Content-Type": "application/x-www-form-urlencoded",
}

SPEAKERS = [
    ("동현 Pro",  "vdonghyun"),
    ("유나 Pro",  "vyuna"),
    ("혜리 Pro",  "vhyeri"),
    ("드림",      "njangj"),
    ("박리뷰",    "nreview"),
    ("상도",      "nsangdo"),
    ("선희",      "nsunhee"),
    ("승표",      "nseungpyo"),
    ("지환",      "njihwan"),
]

# P1 disfluent 변형 (학술 검증 disfluency: filler·lengthening·repetition·mid-pause)
VARIANTS = [
    ("varA",
     "어, 어제 말씀하신 거 있잖아요오. 그게 좀, 그게 급한 거 같아서... 혹시 통화 가능하세요?"),
    ("varB",
     "아... 진짜 너무 힘드네요오. 어떻게 해야 좋을지... 모르겠어요."),
]

OUT_DIR = Path("data_kspon/synth_main_v3")
MANIFEST = Path("metadata/splits/synth_main_v3_manifest.csv")


def synth_16k(speaker: str, text: str, out_path: Path) -> tuple[bool, str]:
    """CLOVA에서 16kHz wav 직접 받기."""
    res = requests.post(
        URL, headers=HEADERS,
        data={"speaker": speaker, "text": text, "format": "wav",
              "sampling-rate": "16000"},
        timeout=30,
    )
    if res.status_code != 200:
        return False, f"{res.status_code} {res.text[:120]}"
    out_path.write_bytes(res.content)
    return True, ""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    ok, fail = 0, 0
    total_chars = 0

    for label, spk in SPEAKERS:
        for var_name, text in VARIANTS:
            out = OUT_DIR / f"{spk}__{var_name}.wav"
            if out.exists():
                print(f"  - skip {spk}/{var_name}")
                continue
            success, err = synth_16k(spk, text, out)
            if not success:
                print(f"  ✗ {label}({spk})/{var_name}: {err}")
                fail += 1
                continue
            ok += 1
            total_chars += len(text)
            print(f"  ✓ {label:8s} {spk:12s} {var_name} ({len(text)}자)")
            rows.append({
                "speaker_label": label,
                "speaker": spk,
                "variant": var_name,
                "text": text,
                "char_count": len(text),
                "wav_path": str(out),
            })

    if rows:
        with MANIFEST.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"\n결과: 성공 {ok} / 실패 {fail}")
    print(f"총 {total_chars}자 (CLOVA 무료 한도의 {total_chars/10000:.3f}%)")
    print(f"출력: {OUT_DIR}")
    print(f"manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
