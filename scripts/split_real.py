"""train_01 메타 → speaker-disjoint train/val/test 분할.

화자 11/3/3 (0001~0011 train, 0012~0014 val, 0015~0017 test).
화자당 N발화(기본 570)만 진짜로 사용 → 진짜 음성 약 15h.
나머지 발화는 합성용 텍스트 풀에 저장 (텍스트 누설 방지).

사용:
  python scripts/split_real.py \\
    --input metadata/train_01.csv \\
    --out-dir metadata/splits \\
    --utts-per-speaker 570 \\
    --seed 42
"""
import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

# 22 화자: _01 17명(0001~0017) + _02 5명(0125~0129) → train 14 / val 4 / test 4
TRAIN_SPEAKERS = (
    [f"KsponSpeech_{i:04d}" for i in range(1, 12)]    # 0001~0011 (_01 11명)
    + [f"KsponSpeech_{i:04d}" for i in range(125, 128)]  # 0125~0127 (_02 3명)
)  # 14명
VAL_SPEAKERS = (
    [f"KsponSpeech_{i:04d}" for i in range(12, 15)]   # 0012~0014 (_01 3명)
    + ["KsponSpeech_0128"]                            # _02 1명
)  # 4명
TEST_SPEAKERS = (
    [f"KsponSpeech_{i:04d}" for i in range(15, 18)]   # 0015~0017 (_01 3명)
    + ["KsponSpeech_0129"]                            # _02 1명
)  # 4명


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--utts-per-speaker", type=int, default=570)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    by_speaker: dict[str, list[dict]] = defaultdict(list)
    with args.input.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_speaker[row["speaker"]].append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    leftover: list[dict] = []

    for split_name, speakers in [
        ("train", TRAIN_SPEAKERS),
        ("val", VAL_SPEAKERS),
        ("test", TEST_SPEAKERS),
    ]:
        rows: list[dict] = []
        for spk in speakers:
            utts = by_speaker.get(spk, [])
            if len(utts) < args.utts_per_speaker:
                raise SystemExit(f"화자 {spk}: 발화 {len(utts)} < 요청 {args.utts_per_speaker}")
            shuffled = utts[:]
            rng.shuffle(shuffled)
            chosen = shuffled[: args.utts_per_speaker]
            rest = shuffled[args.utts_per_speaker:]
            for r in chosen:
                r2 = dict(r); r2["split"] = split_name; r2["label"] = "real"
                rows.append(r2)
            leftover.extend(rest)

        out = args.out_dir / f"real_{split_name}.csv"
        with out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        total_dur = sum(float(r["duration_sec"]) for r in rows)
        print(f"{split_name}: {len(speakers)}화자 {len(rows):,}발화 {total_dur/3600:.2f}h → {out.name}")

    leftover_path = args.out_dir / "synth_text_pool_train_01.csv"
    with leftover_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(leftover[0].keys()))
        w.writeheader(); w.writerows(leftover)
    leftover_dur = sum(float(r["duration_sec"]) for r in leftover)
    print(f"\n합성용 풀(train_01 잔여): {len(leftover):,}발화 {leftover_dur/3600:.2f}h → {leftover_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
