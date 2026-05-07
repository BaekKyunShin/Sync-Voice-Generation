"""
청취 비교 세트 생성.

6세트 × 5 wav = 30 wav 를 한 폴더에 평면 배치.
세트 선정: 화자 3:3, 길이 분산 (짧음/중간/긺), 망설임 유/무 mix.

파일명 규약 (재생기 정렬):
    set01_<spk>_<short_utt>__01_orig.wav
    set01_<spk>_<short_utt>__02_noise.wav
    set01_<spk>_<short_utt>__03_codec.wav
    set01_<spk>_<short_utt>__04_reverb.wav
    set01_<spk>_<short_utt>__05_chain.wav

→ Finder/VLC 정렬만으로 같은 발화 5개가 인접.

실행:
    python scripts/make_listening_compare.py
"""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

PILOT_MANIFEST = Path("metadata/splits/synth_pilot_manifest.csv")
AUG_MANIFEST = Path("metadata/splits/synth_pilot_aug_manifest.csv")
OUT_DIR = Path("data_kspon/listening_compare")
INDEX_CSV = OUT_DIR / "_index.csv"

N_PER_SPEAKER = 3  # 총 6세트


def short_id(utt_id: str) -> str:
    """KsponSpeech_000289 → 000289 / KsponSpeech_E01448 → E01448"""
    return utt_id.split("_")[-1]


def select_sets(rows: list[dict]) -> list[dict]:
    """화자별 길이 분산 + 망설임 mix로 N_PER_SPEAKER개씩 선정."""
    by_spk: dict[str, list[dict]] = {}
    for r in rows:
        by_spk.setdefault(r["speaker"], []).append(r)

    selected: list[dict] = []
    for spk, spk_rows in by_spk.items():
        # 길이 정렬 후 균등 분포 인덱스 추출
        sorted_rows = sorted(spk_rows, key=lambda r: float(r["duration_sec"]))
        n = len(sorted_rows)
        if n == 0:
            continue
        # n=21 일 때 인덱스 0, 10, 20 → 짧음/중간/긺
        idxs = [round(i * (n - 1) / (N_PER_SPEAKER - 1)) for i in range(N_PER_SPEAKER)]
        for idx in idxs:
            selected.append(sorted_rows[idx])
    return selected


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 기존 청취 폴더 비우기 (set 번호 일관성 위해)
    for old in OUT_DIR.glob("set*.wav"):
        old.unlink()

    with PILOT_MANIFEST.open() as f:
        pilot_rows = list(csv.DictReader(f))
    print(f"pilot manifest: {len(pilot_rows)} 발화")

    with AUG_MANIFEST.open() as f:
        aug_rows = list(csv.DictReader(f))
    aug_lookup: dict[tuple, dict] = {}
    for r in aug_rows:
        aug_lookup[(r["speaker"], r["utt_id"], r["variant"])] = r

    selected = select_sets(pilot_rows)
    print(f"선정: {len(selected)}세트 (화자별 {N_PER_SPEAKER}개)")

    index_rows: list[dict] = []
    for set_idx, sel in enumerate(selected, start=1):
        spk = sel["speaker"]
        utt_id = sel["utt_id"]
        prefix = f"set{set_idx:02d}_{spk}_{short_id(utt_id)}"

        # 1. orig
        shutil.copy(sel["wav16_path"], OUT_DIR / f"{prefix}__01_orig.wav")
        # 2~5. 증강
        for n_idx, variant in [(2, "noise"), (3, "codec"), (4, "reverb"), (5, "chain")]:
            aug = aug_lookup[(spk, utt_id, variant)]
            shutil.copy(aug["wav_path"], OUT_DIR / f"{prefix}__0{n_idx}_{variant}.wav")

        index_rows.append({
            "set": f"set{set_idx:02d}",
            "speaker": spk,
            "utt_id": utt_id,
            "duration_sec": sel["duration_sec"],
            "text_clean": sel["text_clean"],
            "text_synth": sel["text_synth"],
            "hesitation_applied": sel["hesitation_applied"],
            "tts_speed": sel["speed"],
            "tts_pitch": sel["pitch"],
            "tts_volume": sel["volume"],
            "tts_emotion": sel["emotion"],
            "noise_snr_db": aug_lookup[(spk, utt_id, "noise")]["snr_db"],
            "reverb_ops": aug_lookup[(spk, utt_id, "reverb")]["applied_ops"],
            "chain_ops": aug_lookup[(spk, utt_id, "chain")]["applied_ops"],
        })

    with INDEX_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
        writer.writeheader()
        writer.writerows(index_rows)

    print(f"\n{len(selected)}세트 × 5 wav = {len(selected) * 5} wav 추출")
    print(f"폴더: {OUT_DIR}")
    print(f"index: {INDEX_CSV}")
    print("\n청취 팁:")
    print(f"  open {OUT_DIR}    # Finder에서 이름순 정렬")
    print(f"  → 같은 발화 5개(orig→noise→codec→reverb→chain)가 인접 배치됨")


if __name__ == "__main__":
    main()
