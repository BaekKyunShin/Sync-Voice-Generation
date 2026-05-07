# Sync-Voice-Generation

한국어 **AI 합성 음성 탐지 (anti-spoofing)** 모델을 위한 데이터 파이프라인.

> 사람이 직접 말한 것 vs AI가 합성한 것을 raw waveform 입력으로 이진 분류.
> 보이스피싱·딥페이크 음성 사기 방어 목적.

자세한 기획·일정·연구 질문은 [`project_outline.md`](project_outline.md) 참고.

## 데이터

| 종류 | 시간 | 출처 |
|---|---|---|
| 진짜 (Bona-fide) | 15h | KsponSpeech_01 + _02 일부 (한국어 자유발화) |
| 가짜 (Spoofed) | 15h | CLOVA Voice 7h + Google TTS 5h + ElevenLabs 3h |

모두 **16kHz / 16bit / mono PCM WAV** 통일.

## 환경

```bash
cp .env.example .env   # CLOVA 키 등 입력
pip install -r requirements.txt
```

필요한 환경 변수:
- `AIHUB_API_KEY` — KsponSpeech 다운
- `CLOVA_CLIENT_ID` / `CLOVA_CLIENT_SECRET` — Naver Cloud Platform CLOVA Voice Premium

## 디렉터리

```
scripts/
├── lib/                            # 재사용 라이브러리
│   ├── augment_ops.py              # RawBoost-style 4 카테고리 + chain
│   ├── hesitation_inject.py        # 망설임 텍스트 변형
│   └── build_synthetic_rirs.py     # 합성 IR 풀 생성
├── extract_metadata.py             # KsponSpeech 메타 추출
├── pcm_to_wav.py                   # PCM → 16kHz WAV
├── filter_real_clean.py            # § 7.4 텍스트 정제
├── split_real.py                   # speaker-disjoint split
├── build_synth_pool.py             # 합성용 텍스트 풀 구성
├── clean_synth_text.py             # 합성용 텍스트 정제
├── convert_real_wav.py             # 진짜 음성 변환
├── synth_sanity_check.py           # CLOVA 화자 sanity check
├── validate_speakers_v3.py         # 화자 ID validate
├── synth_pilot.py                  # 50발화 pilot 합성
├── synth_main_v3.py                # P1 disfluent × 8 화자 본 합성
├── disfluency_sanity_v2.py         # 망설임 카테고리 비교 sanity
├── apply_augment_pilot.py          # pilot 후처리 (5 변형)
├── apply_augment_v3.py             # v3 후처리 (light/medium/heavy)
└── make_listening_compare.py       # 청취 비교 폴더 정리

project_outline.md                  # 전체 기획·일정·연구 질문
CLAUDE.md                           # AI 협업 컨벤션
```

## 핵심 정책

- **§ 7.4 텍스트 정제** — 망설임("어/아/음")은 보존, 비언어 마커(`b/`·`o/`·`+`·`*`)는 제거
- **Speaker-disjoint split** — 같은 화자가 train·test에 동시 들어가지 않게
- **§ 6 양방향 증강** — 합성음 only(노이즈·코덱·reverb), 진짜 only(약 denoise), 양쪽 동일(SpecAugment)
- **RawBoost-style v3 후처리** — linear conv EQ, soft clipping, impulse, real noise (MUSAN), mp3/μ-law codec

## 라이선스 주의

KsponSpeech 음성·텍스트는 **학내 연구·발표 한정**. 외부 공개 금지 — 따라서 `data_kspon/`, `metadata/`, `archive/` 모두 `.gitignore`로 제외.

## 참고 문헌

- Tak et al. 2022. ["RawBoost: A Raw Data Boosting and Augmentation Method"](https://arxiv.org/abs/2111.04433). Interspeech.
- Kharitonov et al. 2020. "Data Augmenting Contrastive Learning of Speech Representations". [WavAugment](https://github.com/facebookresearch/WavAugment).
- Snyder et al. 2015. ["MUSAN: A Music, Speech, and Noise Corpus"](http://www.openslr.org/17/).
