"""진짜 split에서 비언어 마커(b/n/l) 많은 발화 제외 — § 5.4 보완책.

진짜에 b/(숨)·n/(잡음)·l/(웃음) 마커가 많은 발화 = 비언어 음향이 많이 섞임.
가짜는 그런 음향이 없어서, 모델이 이를 단서로 "진짜 vs 가짜" 구분 학습할 위험.
대책: 마커 N개 이상인 발화는 학습/평가 풀에서 제외.

사용:
  python scripts/filter_real_clean.py \\
    --splits-dir metadata/splits \\
    --max-nonverbal 2
"""
import argparse
import csv
import re
from pathlib import Path

RE_NONVERBAL = re.compile(r"\b[bnlBNL]/")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-dir", type=Path, required=True)
    ap.add_argument("--max-nonverbal", type=int, default=2,
                    help="허용 마커 최대 개수 (이 값 초과 시 제외)")
    args = ap.parse_args()

    for split in ("train", "val", "test"):
        src = args.splits_dir / f"real_{split}_with_wav.csv"
        rows = list(csv.DictReader(src.open(encoding="utf-8")))
        kept: list[dict] = []
        for r in rows:
            cnt = len(RE_NONVERBAL.findall(r["text"]))
            if cnt <= args.max_nonverbal:
                kept.append(r)
        out = args.splits_dir / f"real_{split}_clean.csv"
        with out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(kept[0].keys()))
            w.writeheader(); w.writerows(kept)
        dur = sum(float(r["duration_sec"]) for r in kept)
        dropped = len(rows) - len(kept)
        print(f"{split}: {len(rows):,} → {len(kept):,} "
              f"(제외 {dropped:,}, {dropped/len(rows)*100:.1f}%) "
              f"{dur/3600:.2f}h → {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
