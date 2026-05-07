"""
CLOVA Voice 후보 화자 sanity check.

각 후보 화자(12명)에 대해 톤이 다른 텍스트 3개씩 합성 → wav 36개 생성.
16kHz로 다운샘플해서 진짜 데이터(KsponSpeech)와 동일 포맷으로 통일.

목적: 본 합성 들어가기 전에 톤·발화 품질을 청취 비교해 § 5 가이드대로
      자연스러운 일상 대화 톤 화자 4~6명을 추리는 단계.

실행:
  python scripts/synth_sanity_check.py

출력:
  data_kspon/sanity_check/<speaker>_<tag>.wav        (원본, 24kHz)
  data_kspon/sanity_check_16k/<speaker>_<tag>.wav    (청취·비교용, 16kHz)
"""
from __future__ import annotations

import os
from pathlib import Path

import librosa
import requests
import soundfile as sf
from dotenv import load_dotenv

load_dotenv()
CLIENT_ID = os.environ["CLOVA_CLIENT_ID"]
CLIENT_SECRET = os.environ["CLOVA_CLIENT_SECRET"]

URL = "https://naveropenapi.apigw.ntruss.com/tts-premium/v1/tts"
HEADERS = {
    "X-NCP-APIGW-API-KEY-ID": CLIENT_ID,
    "X-NCP-APIGW-API-KEY": CLIENT_SECRET,
    "Content-Type": "application/x-www-form-urlencoded",
}

# 후보 화자 12명 (콘솔 ID 기준, NES 화자는 'n' + 이름 형식으로 정정)
SPEAKERS = [
    # Pro 라인 (여)
    "vara", "vmikyung", "vyuna", "vhyeri", "vgoeun",
    # Pro 라인 (남)
    "vdaeseong", "vian", "vdonghyun",
    # NES 일반 (남)
    "njihun", "njihwan",
    # NES 일반 (여)  ※ 콘솔에서 ID 한 번 재확인 권장
    "nhyeri", "nmikyung",
]

# 톤 분리용 3종 텍스트
TEXTS = {
    "casual":   "어, 그 어제 말씀하신 거 있잖아요. 그게 좀 급한 거 같아서 연락 드렸는데, 혹시 통화 가능하세요?",
    "question": "그래서, 이거 진짜 맞아요? 좀 이상하지 않아요?",
    "long":     "지난주에 보낸 메일 보셨나요? 일정이 좀 바뀌어서, 다음 주 화요일 오후 두 시쯤으로 옮겼으면 하는데요.",
}

OUT_24K = Path("data_kspon/sanity_check")
OUT_16K = Path("data_kspon/sanity_check_16k")
OUT_24K.mkdir(parents=True, exist_ok=True)
OUT_16K.mkdir(parents=True, exist_ok=True)


def synth_one(speaker: str, text: str, out_path: Path) -> tuple[bool, str]:
    res = requests.post(
        URL, headers=HEADERS,
        data={"speaker": speaker, "text": text, "format": "wav"},
        timeout=30,
    )
    if res.status_code != 200:
        return False, f"{res.status_code} {res.text[:120]}"
    out_path.write_bytes(res.content)
    return True, ""


def to_16k(src: Path, dst: Path) -> None:
    y, _ = librosa.load(src, sr=16000)
    sf.write(dst, y, 16000, subtype="PCM_16")


def main() -> None:
    ok, fail, skip = 0, 0, 0
    for spk in SPEAKERS:
        for tag, text in TEXTS.items():
            f24 = OUT_24K / f"{spk}_{tag}.wav"
            f16 = OUT_16K / f"{spk}_{tag}.wav"
            if f16.exists():
                print(f"  - skip {spk}_{tag}")
                skip += 1
                continue
            success, err = synth_one(spk, text, f24)
            if not success:
                print(f"  ✗ {spk}_{tag}: {err}")
                fail += 1
                continue
            to_16k(f24, f16)
            print(f"  ✓ {spk}_{tag}")
            ok += 1

    print(f"\n결과: 성공 {ok} / 실패 {fail} / 건너뜀 {skip}")
    print(f"24kHz 원본:  {OUT_24K}")
    print(f"16kHz 통일:  {OUT_16K}  ← 이걸 청취·비교용으로 사용")


if __name__ == "__main__":
    main()
