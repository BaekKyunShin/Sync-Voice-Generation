"""Google Cloud TTS sanity: 13 후보 화자 × P1 disfluent A/B = 26 wav (raw, 후처리 X).

CLOVA `synth_main_v3.py` 패턴 그대로 — 같은 텍스트로 화자별 자연스러움만 비교.

후보:
    Neural2  3명 (A·B·C)              — § 4.5 권장
    Wavenet  4명 (A·B·C·D)            — 검증된 구세대
    Chirp3-HD 6명                     — 최신, 다양성 확보

실행:
    python scripts/synth_google_sanity.py

출력:
    data_kspon/synth_google_sanity/<voice>__<varA|varB>.wav   (16kHz LINEAR16)
    metadata/splits/synth_google_sanity_manifest.csv
"""
from __future__ import annotations

import csv
from pathlib import Path

from dotenv import load_dotenv
from google.cloud import texttospeech

load_dotenv()

CANDIDATES = [
    # Neural2
    "ko-KR-Neural2-A", "ko-KR-Neural2-B", "ko-KR-Neural2-C",
    # Wavenet
    "ko-KR-Wavenet-A", "ko-KR-Wavenet-B", "ko-KR-Wavenet-C", "ko-KR-Wavenet-D",
    # Chirp3-HD (성별·인상 mix)
    "ko-KR-Chirp3-HD-Achernar",
    "ko-KR-Chirp3-HD-Aoede",
    "ko-KR-Chirp3-HD-Kore",
    "ko-KR-Chirp3-HD-Charon",
    "ko-KR-Chirp3-HD-Puck",
    "ko-KR-Chirp3-HD-Zephyr",
]

VARIANTS = [
    ("varA",
     "어제 말씀하신 거, 그 있잖아요. 그게 좀, 그게 급한 거 같아서... 혹시 통화 가능하세요?"),
    ("varB",
     "진짜 너무 힘드네요. 어떻게 해야 좋을지, 그게... 모르겠어요."),
]

OUT_DIR = Path("data_kspon/synth_google_sanity")
MANIFEST = Path("metadata/splits/synth_google_sanity_manifest.csv")
SR = 16000


def synth(client, voice_name: str, text: str, out_path: Path) -> tuple[bool, str]:
    try:
        response = client.synthesize_speech(
            input=texttospeech.SynthesisInput(text=text),
            voice=texttospeech.VoiceSelectionParams(
                language_code="ko-KR", name=voice_name,
            ),
            audio_config=texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                sample_rate_hertz=SR,
            ),
        )
    except Exception as e:
        return False, str(e)[:160]
    out_path.write_bytes(response.audio_content)
    return True, ""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    client = texttospeech.TextToSpeechClient()
    rows: list[dict] = []
    ok, fail = 0, 0
    total_chars = 0

    for voice in CANDIDATES:
        short = voice.replace("ko-KR-", "")
        for var_name, text in VARIANTS:
            out = OUT_DIR / f"{short}__{var_name}.wav"
            if out.exists():
                print(f"  - skip {short}/{var_name}")
                continue
            success, err = synth(client, voice, text, out)
            if not success:
                print(f"  ✗ {short}/{var_name}: {err}")
                fail += 1
                continue
            ok += 1
            total_chars += len(text)
            print(f"  ✓ {short:30s} {var_name} ({len(text)}자)")
            rows.append({
                "voice": voice,
                "voice_short": short,
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
    print(f"총 {total_chars}자 (Google Neural2 월 무료 1M의 {total_chars/10000:.3f}%)")
    print(f"출력: {OUT_DIR}")
    print(f"manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
