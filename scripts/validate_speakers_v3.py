"""
9 화자 ID validate — 1 발화씩만 합성해서 어떤 ID가 valid한지 확인.

대상 9 화자 (사용자가 clova.ai 데모에서 미리듣기로 선정):
    동현(vdonghyun), 유나Pro(vyuna), 혜리Pro(vhyeri),
    드림(njangj), 박리뷰(nreview),
    상도(nsangdo), 선희(nsunhee), 승표(nseungpyo), 지환(njihwan)

vdonghyun/vhyeri/njihwan 은 이전 sanity에서 valid 확인됨.
나머지 6명은 이번에 검증.

실행:
    python scripts/validate_speakers_v3.py

출력:
    data_kspon/validate_v3/<spk>.wav  (16kHz)  — valid 화자만
    표준출력에 valid/invalid 결과
"""
from __future__ import annotations

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
    ("동현 Pro",   "vdonghyun"),
    ("유나 Pro",   "vyuna"),
    ("혜리 Pro",   "vhyeri"),
    ("드림",       "njangj"),
    ("박리뷰",     "nreview"),
    ("상도",       "nsangdo"),
    ("선희",       "nsunhee"),
    ("승표",       "nseungpyo"),
    ("지환",       "njihwan"),
]

TEXT = "안녕하세요. 음성 합성 테스트입니다."
OUT_DIR = Path("data_kspon/validate_v3")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    valid: list[tuple[str, str]] = []
    invalid: list[tuple[str, str, str]] = []

    for label, spk in SPEAKERS:
        # CLOVA 16kHz wav 직접 받기 (sampling-rate 파라미터)
        res = requests.post(
            URL, headers=HEADERS,
            data={"speaker": spk, "text": TEXT, "format": "wav",
                  "sampling-rate": "16000"},
            timeout=30,
        )
        if res.status_code == 200:
            (OUT_DIR / f"{spk}.wav").write_bytes(res.content)
            print(f"  ✓ {label:10s} = {spk}")
            valid.append((label, spk))
        else:
            err = res.text[:100]
            print(f"  ✗ {label:10s} = {spk}: {res.status_code} {err}")
            invalid.append((label, spk, err))

    print(f"\n결과: valid {len(valid)} / invalid {len(invalid)}")
    if invalid:
        print("\n[invalid 화자 → 콘솔에서 정확한 ID 확인 필요]")
        for label, spk, err in invalid:
            print(f"  - {label} (시도: {spk})")


if __name__ == "__main__":
    main()
