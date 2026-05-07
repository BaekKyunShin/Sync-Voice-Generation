"""KsponSpeech raw PCM(16kHz/16bit/mono LE) → WAV 변환.

사용:
  python scripts/pcm_to_wav.py <input_dir> <output_dir>
  python scripts/pcm_to_wav.py data_kspon/10.한국어음성/eval_clean data_kspon/wav_eval/eval_clean
"""
import argparse
import sys
import wave
from pathlib import Path

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2  # 16bit
CHANNELS = 1


def pcm_to_wav(pcm_path: Path, wav_path: Path) -> tuple[int, float]:
    """raw PCM을 WAV로 감쌈. (samples, duration_sec) 반환."""
    raw = pcm_path.read_bytes()
    if len(raw) % SAMPLE_WIDTH:
        raw = raw[: len(raw) - (len(raw) % SAMPLE_WIDTH)]
    samples = len(raw) // SAMPLE_WIDTH
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(CHANNELS)
        w.setsampwidth(SAMPLE_WIDTH)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(raw)
    return samples, samples / SAMPLE_RATE


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir", type=Path)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--limit", type=int, default=None, help="최대 변환 개수 (테스트용)")
    args = ap.parse_args()

    pcms = sorted(args.input_dir.glob("*.pcm"))
    if args.limit:
        pcms = pcms[: args.limit]

    total_dur = 0.0
    for i, pcm in enumerate(pcms, 1):
        wav = args.output_dir / (pcm.stem + ".wav")
        _, dur = pcm_to_wav(pcm, wav)
        total_dur += dur
        if i % 500 == 0 or i == len(pcms):
            print(f"  {i}/{len(pcms)} ({total_dur/60:.1f}분 누적)", file=sys.stderr)
    print(f"완료: {len(pcms)}개, 총 {total_dur/60:.1f}분 ({total_dur/3600:.2f}h)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
