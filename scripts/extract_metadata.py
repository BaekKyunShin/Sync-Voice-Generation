"""KsponSpeech .trn + PCM 디렉터리 → 메타 CSV.

추출 컬럼: utt_id, speaker, pcm_path, text, samples, duration_sec

사용:
  python scripts/extract_metadata.py \\
    --trn data_kspon/scripts/train.trn \\
    --pcm-root data_kspon/10.한국어음성 \\
    --subset KsponSpeech_01 \\
    --out metadata/train_01.csv

  # eval (화자 ID 없음 → speaker 컬럼 'unknown')
  python scripts/extract_metadata.py \\
    --trn data_kspon/scripts/eval_clean.trn \\
    --pcm-root data_kspon/10.한국어음성 \\
    --subset eval_clean \\
    --out metadata/eval_clean.csv

.trn 라인 형식: <relpath>.pcm :: <전사문>
  train: KsponSpeech_01/KsponSpeech_0001/KsponSpeech_000001.pcm :: ...
  eval : KsponSpeech_eval/eval_clean/KsponSpeech_E00001.pcm :: ...
"""
import argparse
import csv
import sys
from pathlib import Path

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2  # 16-bit


def parse_trn_line(line: str) -> tuple[str, str] | None:
    """`<path> :: <text>` → (path, text). 형식 안 맞으면 None."""
    if " :: " not in line:
        return None
    path, text = line.split(" :: ", 1)
    return path.strip(), text.strip()


def extract_speaker(rel_path: str) -> str:
    """train: KsponSpeech_01/KsponSpeech_0001/... → KsponSpeech_0001
    eval: KsponSpeech_eval/eval_clean/... → 'unknown'"""
    parts = rel_path.split("/")
    if len(parts) >= 3 and parts[1].startswith("KsponSpeech_") and parts[1][12:].isdigit():
        return parts[1]
    return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trn", type=Path, required=True)
    ap.add_argument("--pcm-root", type=Path, required=True,
                    help=".trn의 path를 이 디렉터리 기준으로 해석. "
                         "예: data_kspon/10.한국어음성 (KsponSpeech_01/... 가 그 안에 있어야 함)")
    ap.add_argument("--subset", type=str, default=None,
                    help="이 prefix로 시작하는 라인만 추출 (예: KsponSpeech_01)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-check-exists", action="store_true",
                    help="실제 PCM 파일 존재 여부 체크 생략 (빠름, .trn만 읽음)")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    lines = args.trn.read_text(encoding="utf-8").splitlines()
    if args.subset:
        lines = [l for l in lines if l.startswith(args.subset + "/")]
    print(f"라인 수: {len(lines):,}", file=sys.stderr)

    # eval은 trn 경로 prefix가 'KsponSpeech_eval/...'인데 디스크엔 그 prefix가 없음
    # → trn path에서 첫 컴포넌트 제거 후 pcm_root와 join 해야 매칭됨
    rows = []
    skipped = 0
    total_dur = 0.0
    for line in lines:
        parsed = parse_trn_line(line)
        if parsed is None:
            continue
        rel_path, text = parsed
        speaker = extract_speaker(rel_path)

        # 디스크 경로 후보: pcm_root / rel_path 그대로, 또는 첫 컴포넌트 제거 후
        candidates = [
            args.pcm_root / rel_path,
            args.pcm_root / "/".join(rel_path.split("/")[1:]),
        ]
        pcm_path = next((c for c in candidates if c.exists()), candidates[0])

        utt_id = Path(rel_path).stem
        if args.no_check_exists:
            samples = -1
            dur = -1.0
        else:
            if not pcm_path.exists():
                skipped += 1
                continue
            sz = pcm_path.stat().st_size
            samples = sz // SAMPLE_WIDTH
            dur = samples / SAMPLE_RATE

        rows.append({
            "utt_id": utt_id,
            "speaker": speaker,
            "pcm_path": str(pcm_path),
            "text": text,
            "samples": samples,
            "duration_sec": f"{dur:.3f}" if dur >= 0 else "",
        })
        if dur > 0:
            total_dur += dur

    with args.out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"저장: {args.out}", file=sys.stderr)
    print(f"발화 {len(rows):,}개 / 누락 {skipped}개", file=sys.stderr)
    if total_dur:
        print(f"총 시간 {total_dur/3600:.2f}h (평균 {total_dur/len(rows):.2f}s)", file=sys.stderr)
    speakers = {r["speaker"] for r in rows}
    print(f"고유 화자 수: {len(speakers)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
