"""
Disfluency 카테고리 + Heavy Chain 후처리 sanity v2.

배경 (학술적 검증):
- arxiv 2412.12710 "Enhancing Naturalness in LLM-Generated Utterances through
  Disfluency Insertion"은 filler·repetition·lengthening·pause 모두 자연성을
  유의미하게 향상시킨다고 보고.
- 한국어 prosody 연구는 phrase-final lengthening(어절 끝 모음 늘림)을
  가장 검증된 자연화 기법으로 본다.
- CLOVA Premium은 SSML 미지원 → 텍스트로만 자연화 가능.

이 스크립트는 한 base 문장을 5 disfluency 카테고리로 변형해 두 화자에 합성하고,
§ 6 양방향 증강 정책의 chain을 더 강하게 적용한 heavy chain까지 만들어
청취 비교용 폴더로 정리한다.

실행:
    python scripts/disfluency_sanity_v2.py

출력:
    data_kspon/disfluency_v2/dry/<spk>__<varN>_<cat>.wav        (10 wav)
    data_kspon/disfluency_v2/heavy/<spk>__<varN>_<cat>.wav      (10 wav)
    metadata/splits/disfluency_v2_manifest.csv
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import requests
import soundfile as sf
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from lib.augment_ops import (  # noqa: E402
    add_synthetic_noise,
    apply_reverb,
    apply_telephony_codec,
    apply_volume_jitter,
    load_rir_pool,
)

load_dotenv()
URL = "https://naveropenapi.apigw.ntruss.com/tts-premium/v1/tts"
HEADERS = {
    "X-NCP-APIGW-API-KEY-ID": os.environ["CLOVA_CLIENT_ID"],
    "X-NCP-APIGW-API-KEY": os.environ["CLOVA_CLIENT_SECRET"],
    "Content-Type": "application/x-www-form-urlencoded",
}

SPEAKERS = ["vdonghyun", "vhyeri"]
SR = 16000
SEED = 42
RIR_DIR = Path("assets/rir")

OUT_DRY = Path("data_kspon/disfluency_v2/dry")
OUT_HEAVY = Path("data_kspon/disfluency_v2/heavy")
MANIFEST = Path("metadata/splits/disfluency_v2_manifest.csv")

# 같은 base 의미로 5 변형. 의미·길이는 비슷하게 유지해서 비교 공정성 확보.
VARIANTS = [
    ("var1_plain",
     "어제 말씀하신 거 있잖아요. 좀 급한 거 같아서 연락 드렸는데, 혹시 통화 가능하세요?"),
    ("var2_filler_lengthen",  # filler + phrase-final lengthening
     "어, 어제 말씀하신 거 있잖아요오. 좀 급한 거 같아서, 연락 드렸는데에. 혹시 통화 가능하세요?"),
    ("var3_repetition",  # word repetition
     "어제, 어제 말씀하신 거 있잖아요. 그 그 좀 급한 거 같아서 연락 드렸는데, 혹시 통화 가능하세요?"),
    ("var4_pause",  # mid-pause via 말줄임표
     "어제 말씀하신 거... 있잖아요. 좀 급한 거 같아서... 연락 드렸는데, 혹시 통화 가능하세요?"),
    ("var5_hedge_restart",  # hedge + false start
     "그러니까, 어제 말씀하신 거 있잖아요. 그게 뭐랄까, 좀 급한 거 같아서 연락 드렸는데, 혹시 통화 가능하세요?"),
]


def synth(speaker: str, text: str, out_path: Path) -> bool:
    res = requests.post(URL, headers=HEADERS,
                        data={"speaker": speaker, "text": text, "format": "wav"},
                        timeout=30)
    if res.status_code != 200:
        print(f"  ✗ {out_path.name}: {res.status_code} {res.text[:120]}")
        return False
    out_path.write_bytes(res.content)
    return True


def to_16k(src: Path, dst: Path) -> None:
    y, _ = librosa.load(src, sr=SR)
    sf.write(dst, y, SR, subtype="PCM_16")


def heavy_chain(y: np.ndarray, rir_pool, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    """모든 4 op를 강한 강도로 순차 적용."""
    snr = float(rng.uniform(5, 10))
    out = add_synthetic_noise(y, SR, snr, rng)

    out = apply_telephony_codec(out, SR, rng)

    wet = float(rng.uniform(0.25, 0.40))
    out = apply_reverb(out, SR, rir_pool, wet, rng)

    out = apply_volume_jitter(out, n_segments=3, gain_db_range=(-6, 6), rng=rng)

    desc = f"noise(snr={snr:.1f})+codec+reverb(wet={wet:.2f})+volume(±6dB)"
    return out, desc


def main() -> None:
    OUT_DRY.mkdir(parents=True, exist_ok=True)
    OUT_HEAVY.mkdir(parents=True, exist_ok=True)
    tmp_24k = Path("data_kspon/disfluency_v2/_tmp_24k")
    tmp_24k.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    rir_pool = load_rir_pool(RIR_DIR, sr=SR)
    print(f"RIR pool: {len(rir_pool)}")

    rng = np.random.default_rng(SEED)
    rows: list[dict] = []
    total_chars = 0
    ok = 0

    for spk in SPEAKERS:
        for var_name, text in VARIANTS:
            base = f"{spk}__{var_name}"
            f24 = tmp_24k / f"{base}.wav"
            f_dry = OUT_DRY / f"{base}.wav"
            f_heavy = OUT_HEAVY / f"{base}.wav"

            # 1) 합성 (24kHz)
            if not synth(spk, text, f24):
                continue
            total_chars += len(text)

            # 2) 16kHz 변환 (dry)
            to_16k(f24, f_dry)

            # 3) heavy chain
            y, _ = librosa.load(f_dry, sr=SR)
            y_h, desc = heavy_chain(y, rir_pool, rng)
            sf.write(f_heavy, np.clip(y_h, -1.0, 1.0), SR, subtype="PCM_16")

            ok += 1
            print(f"  ✓ {base} ({len(text)}자) — {desc}")

            rows.append({
                "speaker": spk,
                "variant": var_name,
                "text": text,
                "char_count": len(text),
                "dry_path": str(f_dry),
                "heavy_path": str(f_heavy),
                "heavy_desc": desc,
            })

    if rows:
        with MANIFEST.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"\n결과: {ok} 합성 / {len(rows) * 2} 출력 wav (dry + heavy)")
    print(f"총 글자수: {total_chars} (CLOVA 무료 한도의 {total_chars/10000:.3f}%)")
    print(f"dry:    {OUT_DRY}")
    print(f"heavy:  {OUT_HEAVY}")
    print(f"manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
