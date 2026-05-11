"""본 합성 wav에 v3 후처리 mix 적용 — 학습 데이터 완성.

Mix 비율 (사용자 결정):
  - light    70% — 조용한 실내 톤 (SNR 25~35dB, wet 0.03~0.08)
  - medium   25% — 일반 통화·녹음 (SNR 18~25dB, wet 0.08~0.13)
  - original  5% — 후처리 X (합성티만, dry)
  - heavy     0% (제외)

각 light/medium에는 75% 확률로 silence trim + 합성 호흡음 prepend (시작 자연화).
RawBoost 4 카테고리 (linear conv EQ, soft clipping, impulsive, real MUSAN noise) +
mp3/μ-law codec + smooth volume walk이 발화 단위 random 적용.

실행:
    # CLOVA (기본값)
    python scripts/apply_augment_full_mix.py

    # Google
    python scripts/apply_augment_full_mix.py \\
        --input metadata/splits/synth_full_google_manifest.csv \\
        --out-dir data_kspon/synth_full_google_aug \\
        --out-manifest metadata/splits/synth_full_google_aug_manifest.csv \\
        --seed 43

전제:
    --input manifest 존재
    assets/rir/ (16개 합성 IR)
    assets/musan/musan/noise/ (930개 real noise)

출력:
    --out-dir/<speaker>/<utt_id>.wav (16kHz, 1 강도)
    --out-manifest CSV
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from lib.augment_ops import (  # noqa: E402
    apply_chain_v3,
    load_real_noise_pool,
    load_rir_pool,
    prepend_breath_natural,
)

DEFAULT_INPUT = Path("metadata/splits/synth_full_manifest.csv")
DEFAULT_OUT_DIR = Path("data_kspon/synth_full_aug")
DEFAULT_OUT_MANIFEST = Path("metadata/splits/synth_full_aug_manifest.csv")
RIR_DIR = Path("assets/rir")
MUSAN_NOISE_DIR = Path("assets/musan/musan/noise")
SR = 16000
DEFAULT_SEED = 42

MIX_WEIGHTS = [("light", 0.70), ("medium", 0.25), ("original", 0.05)]
BREATH_APPLY_PROB = 0.75

# spoof split (build_manifest_full.py와 동기화 필수)
# 출력 경로 패턴: <out-dir>/<split>/<speaker>/<utt_id>.wav
SPOOF_SPLIT = {
    "vdonghyun": "train", "vyuna": "train", "vhyeri": "train",
    "njangj": "train", "nreview": "train", "nsangdo": "train",
    "nseungpyo": "val", "njihwan": "test",
    "ko-KR-Chirp3-HD-Aoede": "train",
    "ko-KR-Chirp3-HD-Charon": "train",
    "ko-KR-Chirp3-HD-Kore": "train",
    "ko-KR-Neural2-C": "val",
    "ko-KR-Wavenet-C": "test",
}

OUT_FIELDS = ["utt_id", "speaker", "strength", "applied_ops",
              "wav_path", "duration_sec"]


def sample_strength(rng: np.random.Generator) -> str:
    r = float(rng.random())
    cum = 0.0
    for name, w in MIX_WEIGHTS:
        cum += w
        if r < cum:
            return name
    return MIX_WEIGHTS[0][0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--out-manifest", type=Path, default=DEFAULT_OUT_MANIFEST)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    print(f"input:        {args.input}")
    print(f"out-dir:      {args.out_dir}")
    print(f"out-manifest: {args.out_manifest}")
    print(f"seed:         {args.seed}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rir_pool = load_rir_pool(RIR_DIR, sr=SR)
    real_noise = load_real_noise_pool(MUSAN_NOISE_DIR, sr=SR, max_files=30)
    print(f"RIR {len(rir_pool)}, real noise {len(real_noise)}")

    rng = np.random.default_rng(args.seed)

    with args.input.open() as f:
        rows_in = list(csv.DictReader(f))
    print(f"입력 wav: {len(rows_in)}")

    rows_out: list[dict] = []
    by_strength = {"light": 0, "medium": 0, "original": 0}

    for i, row in enumerate(rows_in):
        spk = row["speaker"]
        utt_id = row["utt_id"]
        wav_in = Path(row["wav_path"])
        if spk not in SPOOF_SPLIT:
            raise SystemExit(f"unknown speaker {spk!r} — SPOOF_SPLIT 갱신 필요")
        spk_dir = args.out_dir / SPOOF_SPLIT[spk] / spk
        spk_dir.mkdir(parents=True, exist_ok=True)
        wav_out = spk_dir / f"{utt_id}.wav"

        strength = sample_strength(rng)
        by_strength[strength] += 1

        if strength == "original":
            shutil.copy(wav_in, wav_out)
            applied = "original"
        else:
            y, _ = librosa.load(wav_in, sr=SR)
            y, breath_desc = prepend_breath_natural(
                y, SR, rng, apply_prob=BREATH_APPLY_PROB,
            )
            y, ops = apply_chain_v3(
                y, SR, strength,
                real_noise_pool=real_noise, rir_pool=rir_pool, rng=rng,
            )
            sf.write(wav_out, np.clip(y, -1.0, 1.0), SR, subtype="PCM_16")
            applied = "+".join([breath_desc, *ops])

        info = sf.info(str(wav_out))
        dur = info.frames / info.samplerate
        rows_out.append({
            "utt_id": utt_id, "speaker": spk, "strength": strength,
            "applied_ops": applied, "wav_path": str(wav_out),
            "duration_sec": round(dur, 3),
        })

        if (i + 1) % 250 == 0:
            print(
                f"  진행 {i+1}/{len(rows_in)} | "
                f"light/medium/original = "
                f"{by_strength['light']}/{by_strength['medium']}/{by_strength['original']}"
            )

    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.out_manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        w.writeheader()
        w.writerows(rows_out)

    total = len(rows_out)
    print("\n=== 완료 ===")
    for name, _ in MIX_WEIGHTS:
        n = by_strength[name]
        print(f"  {name:9s}: {n:5d} ({n/total*100:.1f}%)")
    total_dur = sum(r["duration_sec"] for r in rows_out)
    print(f"  총 분량: {total_dur/3600:.2f}h")
    print(f"manifest: {args.out_manifest}")


if __name__ == "__main__":
    main()
