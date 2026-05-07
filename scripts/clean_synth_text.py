"""KsponSpeech 텍스트 정제 — § 5.4 정책 적용.

룰:
1. (A)/(B) → B (발음형 사용)
2. 어/ 아/ 음/ 등 한글+/ → 슬래시만 제거 (망설임 보존)
3. b/ n/ o/ l/ → 제거 (비언어 마커, 합성 불가)
4. + * 단독 → 제거
5. 너무 짧음(<3자)·긺(>200자) → 풀에서 제외

사용:
  python scripts/clean_synth_text.py \\
    --input metadata/splits/synth_text_pool.csv \\
    --output metadata/splits/synth_text_pool_clean.csv
"""
import argparse
import csv
import re
from pathlib import Path

# (A)/(B) — 표기형/발음형 이중 표기, B(발음형) 사용
RE_DUAL = re.compile(r"\(([^)]+)\)/\(([^)]+)\)")
# 한글 + / — 망설임 마커, 슬래시만 제거 (글자 보존)
RE_HESITATE = re.compile(r"([가-힣])/")
# 영문 1글자 + / — 비언어 마커 (b/=숨, n/=잡음, o/=겹침, l/=웃음)
RE_NONVERBAL = re.compile(r"\b[a-zA-Z]/")
# 잘림·부정확 마커
RE_PLUS_STAR = re.compile(r"[+*]")
# 정리용 다중 공백
RE_MULTI_SPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = RE_DUAL.sub(r"\2", text)        # (A)/(B) → B
    text = RE_HESITATE.sub(r"\1 ", text)   # 어/ → 어 (글자 보존, 공백 추가)
    text = RE_NONVERBAL.sub("", text)      # b/ n/ o/ l/ 제거
    text = RE_PLUS_STAR.sub("", text)      # + * 제거
    return RE_MULTI_SPACE.sub(" ", text).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--min-chars", type=int, default=3)
    ap.add_argument("--max-chars", type=int, default=200)
    args = ap.parse_args()

    rows_in = list(csv.DictReader(args.input.open(encoding="utf-8")))
    out: list[dict] = []
    skip_short = skip_long = 0
    for r in rows_in:
        cleaned = clean_text(r["text"])
        n = len(cleaned)
        if n < args.min_chars:
            skip_short += 1; continue
        if n > args.max_chars:
            skip_long += 1; continue
        r2 = dict(r); r2["text_clean"] = cleaned
        out.append(r2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)

    print(f"입력 {len(rows_in):,} → 정제 통과 {len(out):,}")
    print(f"  제외(짧음 <{args.min_chars}자): {skip_short:,}")
    print(f"  제외(긺 >{args.max_chars}자): {skip_long:,}")
    avg = sum(len(r["text_clean"]) for r in out) / len(out)
    total = sum(len(r["text_clean"]) for r in out)
    print(f"  정제본 평균 {avg:.1f}자, 총 {total:,}자")
    print(f"→ {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
