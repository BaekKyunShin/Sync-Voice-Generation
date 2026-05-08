"""통합 학습 매니페스트 빌드 (Track D).

real (KsponSpeech 22 화자, 14.62h) + spoof (CLOVA 8 화자, ~7h) 통합:
  컬럼: utt_id, speaker, label, source, text, wav_path, duration_sec, split

real split: real_{train,val,test}_clean.csv 그대로
spoof split (8 화자 → 6/1/1, 최대 다양성 + speaker-disjoint):
  train: vdonghyun, vyuna, vhyeri, njangj, nreview, nsangdo (6명)
  val:   nseungpyo (1명)
  test:  njihwan (1명)

§ 7.2 (utt_id 누설), § 7.3 (speaker 누설) 자동 검증.

실행:
    python scripts/build_manifest_full.py

출력:
    metadata/splits/manifest_full.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REAL_SPLITS_DIR = Path("metadata/splits")
SPOOF_AUG_MANIFEST = Path("metadata/splits/synth_full_aug_manifest.csv")
SPOOF_RAW_MANIFEST = Path("metadata/splits/synth_full_manifest.csv")
OUT = Path("metadata/splits/manifest_full.csv")

SPOOF_SPLIT = {
    "vdonghyun": "train", "vyuna": "train", "vhyeri": "train",
    "njangj": "train", "nreview": "train", "nsangdo": "train",
    "nseungpyo": "val",
    "njihwan": "test",
}

FIELDS = ["utt_id", "speaker", "label", "source",
           "text", "wav_path", "duration_sec", "split"]


def main() -> int:
    rows: list[dict] = []

    # real
    for split in ("train", "val", "test"):
        src = REAL_SPLITS_DIR / f"real_{split}_clean.csv"
        with src.open() as f:
            for r in csv.DictReader(f):
                rows.append({
                    "utt_id": r["utt_id"],
                    "speaker": r["speaker"],
                    "label": "real",
                    "source": "KsponSpeech",
                    "text": r["text"],
                    "wav_path": r["wav_path"],
                    "duration_sec": r["duration_sec"],
                    "split": split,
                })

    # spoof — text 컬럼은 raw manifest에서 lookup
    text_lookup = {
        r["utt_id"]: r["text"]
        for r in csv.DictReader(SPOOF_RAW_MANIFEST.open())
    }
    with SPOOF_AUG_MANIFEST.open() as f:
        for r in csv.DictReader(f):
            spk = r["speaker"]
            if spk not in SPOOF_SPLIT:
                print(f"  ⚠ unknown speaker {spk} → skip")
                continue
            rows.append({
                "utt_id": r["utt_id"],
                "speaker": spk,
                "label": "spoof",
                "source": "CLOVA",
                "text": text_lookup.get(r["utt_id"], ""),
                "wav_path": r["wav_path"],
                "duration_sec": r["duration_sec"],
                "split": SPOOF_SPLIT[spk],
            })

    # 검증 — § 7.2 utt_id 누설
    by_label = {"real": set(), "spoof": set()}
    for r in rows:
        by_label[r["label"]].add(r["utt_id"])
    inter = by_label["real"] & by_label["spoof"]
    if inter:
        print(f"  ⚠ utt_id 누설 {len(inter)}건: {list(inter)[:3]}")
        return 1

    # 검증 — § 7.3 speaker 누설 (label·split별 화자 교집합)
    spk_sets: dict = {}
    for r in rows:
        key = (r["label"], r["split"])
        spk_sets.setdefault(key, set()).add(r["speaker"])
    for label in ("real", "spoof"):
        train_s = spk_sets.get((label, "train"), set())
        val_s = spk_sets.get((label, "val"), set())
        test_s = spk_sets.get((label, "test"), set())
        for a, b, name in [(train_s, val_s, "train×val"),
                            (train_s, test_s, "train×test"),
                            (val_s, test_s, "val×test")]:
            inter_s = a & b
            if inter_s:
                print(f"  ⚠ {label} {name} speaker 누설: {inter_s}")
                return 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    # 요약
    print(f"통합 manifest: {len(rows)} 행")
    print(f"  ✓ utt_id 누설 0 (§ 7.2)")
    print(f"  ✓ speaker 누설 0 (§ 7.3)")
    print()
    by = {}
    for r in rows:
        k = (r["label"], r["split"])
        by.setdefault(k, {"n": 0, "dur": 0.0, "spk": set()})
        by[k]["n"] += 1
        by[k]["dur"] += float(r["duration_sec"])
        by[k]["spk"].add(r["speaker"])
    print(f"  {'label':6s} {'split':6s} {'화자':>5s} {'발화':>7s} {'시간':>8s}")
    print(f"  {'-'*6} {'-'*6} {'-'*5} {'-'*7} {'-'*8}")
    for label in ("real", "spoof"):
        total_n, total_d = 0, 0.0
        for split in ("train", "val", "test"):
            info = by.get((label, split), {"n": 0, "dur": 0.0, "spk": set()})
            print(f"  {label:6s} {split:6s} {len(info['spk']):>5d} "
                  f"{info['n']:>7d} {info['dur']/3600:>7.2f}h")
            total_n += info["n"]
            total_d += info["dur"]
        print(f"  {label:6s} {'합계':6s} {'':>5s} {total_n:>7d} {total_d/3600:>7.2f}h")
    print(f"\n출력: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
