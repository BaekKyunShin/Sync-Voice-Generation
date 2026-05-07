"""합성용 텍스트 풀 통합.

진짜 음성으로 안 쓴 발화의 텍스트만 모아 가짜 합성용 풀 구성.
출처: train_01 잔여(splits/synth_text_pool_train_01.csv) + eval_clean + eval_other.

사용:
  python scripts/build_synth_pool.py
"""
import csv
from pathlib import Path

ROOT = Path("/Users/baekkyunshin/Desktop/deeplearning_project")
SOURCES = [
    ROOT / "metadata" / "splits" / "synth_text_pool_train_01.csv",
    ROOT / "metadata" / "eval_clean.csv",
    ROOT / "metadata" / "eval_other.csv",
]
OUT = ROOT / "metadata" / "splits" / "synth_text_pool.csv"


def main() -> int:
    rows: list[dict] = []
    for src in SOURCES:
        with src.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append({
                    "utt_id": r["utt_id"],
                    "source": src.stem,
                    "text": r["text"],
                    "ref_speaker": r.get("speaker", "unknown"),
                    "ref_duration_sec": r["duration_sec"],
                })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # 통계
    n = len(rows)
    by_source: dict[str, int] = {}
    char_total = 0
    for r in rows:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
        char_total += len(r["text"])
    print(f"총 {n:,}발화 → {OUT.name}")
    for s, c in sorted(by_source.items()):
        print(f"  {s}: {c:,}")
    print(f"텍스트 총 글자수: {char_total:,} (평균 {char_total/n:.1f}자/발화)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
