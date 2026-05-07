"""
v3 후처리 적용: 18 합성 wav × 3 강도(light/medium/heavy) = 54 wav.

학술 근거: RawBoost (Tak et al. 2022) — linear convolutive(EQ),
non-linear(soft clipping), impulsive, stationary additive 4 카테고리 + 코덱.
강도는 light dominant 정책 (자연스러움 우선).

청취 폴더는 평면 배치, 같은 발화의 3 강도가 정렬 시 인접.

실행:
    python scripts/apply_augment_v3.py

전제:
    scripts/synth_main_v3.py 실행 후 manifest 존재.
    assets/rir/ 에 합성 IR 16개 존재.
    (선택) assets/musan/musan/noise/ 에 MUSAN 노이즈 풀 → 있으면 real noise 사용.

출력:
    data_kspon/synth_main_v3_aug/<spk>__<var>__<strength>.wav  (54 wav)
    data_kspon/listening_compare_v3/<spk>__<var>__<N_strength>.wav  (평면 정렬)
    metadata/splits/synth_main_v3_aug_manifest.csv
"""
from __future__ import annotations

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
)

INPUT_MANIFEST = Path("metadata/splits/synth_main_v3_manifest.csv")
RIR_DIR = Path("assets/rir")
MUSAN_NOISE_DIR = Path("assets/musan/musan/noise")  # 추출 후 위치
OUT_AUG = Path("data_kspon/synth_main_v3_aug")
OUT_COMPARE = Path("data_kspon/listening_compare_v3")
OUT_MANIFEST = Path("metadata/splits/synth_main_v3_aug_manifest.csv")
SR = 16000
SEED = 42

STRENGTH_ORDER = [(1, "light"), (2, "medium"), (3, "heavy")]


def main() -> None:
    OUT_AUG.mkdir(parents=True, exist_ok=True)
    OUT_COMPARE.mkdir(parents=True, exist_ok=True)
    # 청취 폴더 비우기 (set 일관성)
    for old in OUT_COMPARE.glob("*.wav"):
        old.unlink()

    rir_pool = load_rir_pool(RIR_DIR, sr=SR)
    print(f"RIR pool: {len(rir_pool)}개")

    real_noise_pool = load_real_noise_pool(MUSAN_NOISE_DIR, sr=SR, max_files=30)
    if real_noise_pool:
        print(f"MUSAN real noise: {len(real_noise_pool)}개 로드")
    else:
        print("MUSAN 미준비 → 합성 noise(가우시안+핑크) fallback")

    rng = np.random.default_rng(SEED)
    rows_out: list[dict] = []

    with INPUT_MANIFEST.open() as f:
        rows = list(csv.DictReader(f))
    print(f"입력 합성 wav: {len(rows)}개\n")

    for row in rows:
        spk = row["speaker"]
        var = row["variant"]
        wav_path = Path(row["wav_path"])
        y, _ = librosa.load(wav_path, sr=SR)

        for n_idx, strength in STRENGTH_ORDER:
            y_out, applied = apply_chain_v3(
                y, SR, strength,
                real_noise_pool=real_noise_pool, rir_pool=rir_pool, rng=rng,
            )

            # 1) augmentation 폴더에 저장
            aug_path = OUT_AUG / f"{spk}__{var}__{strength}.wav"
            sf.write(aug_path, y_out, SR, subtype="PCM_16")

            # 2) 청취 비교 폴더에 평면 복사 (정렬용 prefix)
            cmp_path = OUT_COMPARE / f"{spk}__{var}__{n_idx}_{strength}.wav"
            shutil.copy(aug_path, cmp_path)

            ops_str = "+".join(applied)
            rows_out.append({
                "speaker": spk,
                "variant": var,
                "strength": strength,
                "applied_ops": ops_str,
                "wav_path": str(aug_path),
                "compare_path": str(cmp_path),
            })
            print(f"  ✓ {spk}/{var}/{strength}: {ops_str}")

    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with OUT_MANIFEST.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"\n총 {len(rows_out)} 후처리 wav 생성")
    print(f"증강 출력: {OUT_AUG}")
    print(f"청취 폴더: {OUT_COMPARE}")
    print(f"manifest:  {OUT_MANIFEST}")


if __name__ == "__main__":
    main()
