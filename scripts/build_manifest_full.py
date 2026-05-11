"""통합 학습 매니페스트 빌드 (Track D).

real (KsponSpeech 22 화자) + spoof (CLOVA 8 + Google 5 = 13 화자) 통합:
  컬럼: utt_id, speaker, label, source, text, wav_path, duration_sec, split

real split: real_{train,val,test}_clean.csv 그대로
spoof split (13 화자 → 9/2/2, val·test에 CLOVA·Google 각 1명씩):
  train: vdonghyun, vyuna, vhyeri, njangj, nreview, nsangdo (CLOVA 6명)
         Chirp3-HD-Aoede, Chirp3-HD-Charon, Chirp3-HD-Kore (Google 3명)
  val:   nseungpyo (CLOVA), Neural2-C (Google)
  test:  njihwan (CLOVA), Wavenet-C (Google)

§ 7.2 (utt_id 누설), § 7.3 (speaker 누설) 자동 검증.
spoof 내부 utt_id 충돌(같은 utt_id가 CLOVA·Google 양쪽) 검출 시 Google 측 drop
(CLOVA가 먼저 합성됨 → 중복 발생 시 Google이 제거 대상).

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

# CLOVA 8 화자
CLOVA_RAW_MANIFEST = Path("metadata/splits/synth_full_manifest.csv")
CLOVA_AUG_MANIFEST = Path("metadata/splits/synth_full_aug_manifest.csv")
# Google 5 화자
GOOGLE_RAW_MANIFEST = Path("metadata/splits/synth_full_google_manifest.csv")
GOOGLE_AUG_MANIFEST = Path("metadata/splits/synth_full_google_aug_manifest.csv")

OUT = Path("metadata/splits/manifest_full.csv")

SPOOF_SPLIT = {
    # CLOVA train (6)
    "vdonghyun": "train", "vyuna": "train", "vhyeri": "train",
    "njangj": "train", "nreview": "train", "nsangdo": "train",
    # CLOVA val/test
    "nseungpyo": "val",
    "njihwan": "test",
    # Google train (3 Chirp3-HD)
    "ko-KR-Chirp3-HD-Aoede": "train",
    "ko-KR-Chirp3-HD-Charon": "train",
    "ko-KR-Chirp3-HD-Kore": "train",
    # Google val/test
    "ko-KR-Neural2-C": "val",
    "ko-KR-Wavenet-C": "test",
}

FIELDS = ["utt_id", "speaker", "label", "source",
           "text", "wav_path", "duration_sec", "split"]


def _source_from_speaker(spk: str) -> str:
    return "Google" if spk.startswith("ko-KR-") else "CLOVA"


def _load_text_lookup(*manifests: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for m in manifests:
        if not m.exists():
            print(f"  ⚠ raw manifest 없음 → skip: {m}")
            continue
        with m.open() as f:
            for r in csv.DictReader(f):
                out[r["utt_id"]] = r["text"]
    return out


def _ingest_spoof(aug_manifest: Path, text_lookup: dict[str, str],
                   already_used: set[str], rows: list[dict],
                   spoof_set: set[str]) -> int:
    """aug manifest 한 개 → rows에 append. utt_id 충돌(이미 spoof_set에 존재) 시 drop."""
    if not aug_manifest.exists():
        print(f"  ⚠ aug manifest 없음 → skip: {aug_manifest}")
        return 0
    n_added = n_drop = 0
    with aug_manifest.open() as f:
        for r in csv.DictReader(f):
            spk = r["speaker"]
            if spk not in SPOOF_SPLIT:
                print(f"  ⚠ unknown speaker {spk} → skip")
                continue
            utt_id = r["utt_id"]
            if utt_id in spoof_set:
                # 같은 utt_id가 이미 다른 spoof source에 사용됨 → 드랍
                n_drop += 1
                continue
            spoof_set.add(utt_id)
            already_used.add(utt_id)
            rows.append({
                "utt_id": utt_id,
                "speaker": spk,
                "label": "spoof",
                "source": _source_from_speaker(spk),
                "text": text_lookup.get(utt_id, ""),
                "wav_path": r["wav_path"],
                "duration_sec": r["duration_sec"],
                "split": SPOOF_SPLIT[spk],
            })
            n_added += 1
    if n_drop:
        print(f"  ⚠ {aug_manifest.name}: {n_drop}건 utt_id 중복 → drop")
    return n_added


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

    # spoof — text lookup은 두 raw manifest 합집합
    text_lookup = _load_text_lookup(CLOVA_RAW_MANIFEST, GOOGLE_RAW_MANIFEST)

    spoof_utts: set[str] = set()
    used_utts: set[str] = set()
    n_clova = _ingest_spoof(CLOVA_AUG_MANIFEST, text_lookup, used_utts, rows, spoof_utts)
    n_google = _ingest_spoof(GOOGLE_AUG_MANIFEST, text_lookup, used_utts, rows, spoof_utts)
    print(f"spoof 적재: CLOVA {n_clova}, Google {n_google}")

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
