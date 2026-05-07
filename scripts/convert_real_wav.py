"""split CSV의 진짜 음성을 PCM → WAV로 일괄 변환.

출력 구조: data_kspon/wav_real/{train,val,test}/<speaker>/<utt_id>.wav
CSV에 wav_path 컬럼 추가해 새 위치에 저장.

사용:
  python scripts/convert_real_wav.py \\
    --splits-dir metadata/splits \\
    --wav-root data_kspon/wav_real
"""
import argparse
import csv
import sys
import wave
from pathlib import Path

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2


def pcm_to_wav(pcm_path: Path, wav_path: Path) -> int:
    raw = pcm_path.read_bytes()
    if len(raw) % SAMPLE_WIDTH:
        raw = raw[: len(raw) - (len(raw) % SAMPLE_WIDTH)]
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(SAMPLE_WIDTH)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(raw)
    return len(raw) // SAMPLE_WIDTH


def convert_split(csv_in: Path, csv_out: Path, wav_root: Path, split: str) -> None:
    rows = list(csv.DictReader(csv_in.open(encoding="utf-8")))
    fieldnames = list(rows[0].keys()) + ["wav_path"]
    total = 0
    for i, r in enumerate(rows, 1):
        pcm = Path(r["pcm_path"])
        wav = wav_root / split / r["speaker"] / (r["utt_id"] + ".wav")
        pcm_to_wav(pcm, wav)
        r["wav_path"] = str(wav)
        total += int(r["samples"])
        if i % 1000 == 0 or i == len(rows):
            print(f"  {split}: {i}/{len(rows)}", file=sys.stderr)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)
    print(f"{split}: {len(rows)}개 변환 ({total/SAMPLE_RATE/3600:.2f}h) → {csv_out.name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-dir", type=Path, required=True)
    ap.add_argument("--wav-root", type=Path, required=True)
    args = ap.parse_args()

    for split in ("train", "val", "test"):
        csv_in = args.splits_dir / f"real_{split}.csv"
        csv_out = args.splits_dir / f"real_{split}_with_wav.csv"
        convert_split(csv_in, csv_out, args.wav_root, split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
