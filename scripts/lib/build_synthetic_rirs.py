"""
약한 reverb용 합성 IR(Impulse Response) 풀 생성.

pyroomacoustics로 작은 방(3~5m × 3~5m × 2.3~2.8m) RIR을 N개 만들어
assets/rir/rir_room_NN.wav 로 저장. apply_reverb()가 이 풀에서 랜덤 샘플.

외부 IR 데이터셋 다운로드 의존성을 없애기 위함. 자연스러움 부족하면
본 합성 단계에서 OpenAIR 등으로 교체 결정.

실행:
    python scripts/lib/build_synthetic_rirs.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyroomacoustics as pra
import soundfile as sf

OUT_DIR = Path("assets/rir")
NUM_RIRS = 16
SR = 16000
SEED = 42

# RT60 0.1~0.6s 분포로 다양화 (small/medium/large 방 mix)
ROOM_PRESETS = [
    # (name, Lx range, Ly range, Lz range, abs range, max_order)
    ("small",  (3.0, 5.0),  (3.0, 5.0),  (2.3, 2.8), (0.45, 0.65), 8),
    ("medium", (5.0, 7.0),  (4.0, 6.0),  (2.5, 3.0), (0.30, 0.45), 12),
    ("large",  (7.0, 10.0), (6.0, 9.0),  (3.0, 4.0), (0.20, 0.35), 17),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # 기존 RIR 삭제 (다양화 풀로 재생성)
    for old in OUT_DIR.glob("rir_*.wav"):
        old.unlink()

    rng = np.random.default_rng(SEED)

    for i in range(NUM_RIRS):
        preset_name, lx_r, ly_r, lz_r, abs_r, max_order = ROOM_PRESETS[i % len(ROOM_PRESETS)]
        Lx = float(rng.uniform(*lx_r))
        Ly = float(rng.uniform(*ly_r))
        Lz = float(rng.uniform(*lz_r))
        e_absorption = float(rng.uniform(*abs_r))

        room = pra.ShoeBox(
            [Lx, Ly, Lz],
            fs=SR,
            materials=pra.Material(e_absorption),
            max_order=max_order,
        )

        src = [
            float(rng.uniform(0.5, Lx - 0.5)),
            float(rng.uniform(0.5, Ly - 0.5)),
            float(rng.uniform(1.0, 1.6)),
        ]
        mic = [
            float(rng.uniform(0.5, Lx - 0.5)),
            float(rng.uniform(0.5, Ly - 0.5)),
            float(rng.uniform(1.0, 1.6)),
        ]
        # source/mic 너무 가깝지 않게
        while np.linalg.norm(np.array(src) - np.array(mic)) < 0.5:
            mic = [
                float(rng.uniform(0.5, Lx - 0.5)),
                float(rng.uniform(0.5, Ly - 0.5)),
                float(rng.uniform(1.0, 1.6)),
            ]

        room.add_source(src)
        room.add_microphone(mic)
        room.compute_rir()
        rir = room.rir[0][0]

        rir = rir / (np.max(np.abs(rir)) + 1e-12)
        # RT60 추정 (rough): -60dB까지 감쇠 시간
        env = np.abs(rir)
        env_db = 20 * np.log10(env / (np.max(env) + 1e-12) + 1e-12)
        below_60 = np.where(env_db < -60)[0]
        rt60 = float(below_60[0] / SR) if len(below_60) > 0 else len(rir) / SR

        out_path = OUT_DIR / f"rir_{preset_name}_{i:02d}.wav"
        sf.write(out_path, rir.astype(np.float32), SR)
        print(
            f"  ✓ {out_path.name:24s} "
            f"({preset_name:6s} {Lx:.1f}×{Ly:.1f}×{Lz:.1f}m, "
            f"abs={e_absorption:.2f}, RT60≈{rt60*1000:.0f}ms)"
        )

    print(f"\n{NUM_RIRS}개 RIR 생성 완료 (small/medium/large 다양화): {OUT_DIR}")


if __name__ == "__main__":
    main()
