# 한국어 합성 음성 탐지 (Anti-spoofing)

> 고려대학교 인공지능 대학원 딥러닝 텀프로젝트
> 작성일: 2026-05-05 / 최근 갱신: 2026-05-06 / 발표 마감: 약 3주 후

---

## 1. 한 줄 요약

**한국어 음성을 raw waveform으로 입력 받아 "사람이 직접 말한 것(진짜)" vs "AI가 합성한 것(가짜)"을 이진 분류한다.**

- 입력: **raw waveform** (STT 미사용 — 텍스트 단서가 아니라 음향 단서로 판별)
- 출력: bona-fide / spoof 점수
- 발표 마감: 3주

---

## 2. 왜 이 문제인가

### 배경
- 2025년 국내 보이스피싱 피해 누적 **1조 원 돌파**, AI 합성음 활용 사례 폭증
- 한국어 voice cloning·TTS 품질이 사람이 듣고 못 구분할 수준
- **실제 보이스피싱 음성**은 데이터 수집 자체가 불가(피해자 음성, 법·윤리 문제) → "보이스피싱 탐지" 대신 **"AI 합성음 탐지"** 로 우회

### 학술 기여
- 영어권에는 ASVspoof 시리즈 같은 공개 벤치마크가 있지만 **한국어 합성음 탐지 공개 벤치마크는 사실상 부재**
- → **데이터셋 구축 자체가 기여** (재현 가능한 manifest + speaker-disjoint split + 양방향 증강 파이프라인)

---

## 3. 핵심 가설 (스토리라인)

> "단일 합성기에만 익숙한 탐지기는 새로운 합성기를 만나면 무력해진다.
>  + 음향 환경 차이를 그대로 두면, 모델은 '합성 흔적'이 아니라 '녹음 품질'을 학습한다."

→ 두 함정을 모두 막아야 진짜로 "합성 vs 자연"을 구분하는 탐지기.

### 우리 접근

| 관문 | 대응 |
|---|---|
| 합성기 단일성 | **3종 TTS** (CLOVA·Google·ElevenLabs) 혼합 학습 |
| 도메인 갭 | **양방향 증강**으로 진짜·가짜 음향 환경을 좁힘 (§ 6) |
| 모델 비교 | **GRU → LCNN → SOTA** 점진 ablation으로 "왜 더 정교한 모델이 필요한가" 정량 입증 |

### 비유
위조 화폐 탐지를 만든다고 치자.
- 한 위조범 가짜만 학습 → 그 사람 붓터치만 외움. 다른 위조범엔 무력.
- + 진짜 지폐는 새 지폐, 가짜는 구겨진 종이 → 모델이 "구겨짐"으로만 판단해버림.

→ TTS도 마찬가지. **다양한 합성기 + 음향 환경 균질화** 가 둘 다 필요.

---

## 4. 데이터

### 4.1 총 규모
- **30시간 = 진짜 15h + 가짜 15h** (1:1 균형)
- 결과 보고 부족하면 점진 확장

### 4.2 진짜(Bona-fide) — 15h
- **AI Hub KsponSpeech_01 + KsponSpeech_02 일부** (한국어 자유발화 표준 벤치마크)
  - KsponSpeech_01: 화자 17명, 정제 후 8,787 발화 / 11.80h (이미 확보)
  - KsponSpeech_02 추가: 화자 약 5~6명 추가 다운로드 → +3~5h
- 16kHz / 16bit / mono PCM WAV (표준화 후)
- 텍스트 정제 후 합산 풀 ≈ 16~17h → **학습용 약 15h 추출**
- **Speaker-disjoint split** (같은 사람이 train·test 양쪽 X), 비율 70/15/15:

| 분할 | 화자 (목표) | 시간 |
|---|---|---|
| train | 약 14명 | ≈ 10.5h |
| val | 약 4명 | ≈ 2.3h |
| test | 약 4명 | ≈ 2.3h |

> 정확한 화자 ID·발화 수는 KsponSpeech_02 일부 다운·정제 후 확정. speaker-disjoint 원칙은 유지.

### 4.3 가짜(Spoofed) — 15h
- **방법**: KsponSpeech 미사용 transcript를 3종 TTS API로 합성
- **텍스트 출처**: 합성용 풀 13,180 발화(eval + train_01 잔여) — 진짜로 쓴 텍스트는 절대 미포함

| TTS | 분량 | 역할 |
|---|---|---|
| **Naver CLOVA Voice Premium** | ≈ 7h | 한국어 특화 상용, 자연스러움 |
| **Google Cloud TTS Neural2** | ≈ 5h | 글로벌 상용 대표, 다국어 |
| **ElevenLabs Multilingual v2** | ≈ 3h | 최고급 품질 비교군 |
| **합계** | **15h** | |

→ **3개 회사 = 3개의 다른 합성 backend = 3가지 다른 artifact**.

#### CLOVA 화자 (8명, sanity 청취 후 § 5 가이드 통과만 선정)

| Speaker ID | Korean | 라인 | 화자당 분량 |
|---|---|---|---|
| `vdonghyun` | 동현 | Pro | ~53분 |
| `vyuna` | 유나 | Pro | ~53분 |
| `vhyeri` | 혜리 | Pro | ~53분 |
| `njangj` | 드림 | NES | ~53분 |
| `nreview` | 박리뷰 | NES | ~53분 |
| `nsangdo` | 상도 | NES | ~53분 |
| `nseungpyo` | 승표 | NES | ~53분 |
| `njihwan` | 지환 | NES | ~53분 |

> 9명 후보(+ `nsunhee`)에서 사용자 청취 결과 nsunhee 제외 → 최종 8명. CLOVA 7h ÷ 8명 = 화자당 ≈ 53분. 한 화자 패턴 외울 위험 ↓ (RawBoost·anti-spoofing 권장 화자 다양성).

### 4.4 진짜·가짜 텍스트 분리
- 진짜로 쓴 9,690 발화의 텍스트는 합성에 절대 사용 X (텍스트 누설 방지)
- 합성용 풀과 진짜용 풀의 화자/문장 교집합 자동 검사

---

## 5. 화자 선택 가이드 (필수 — 가장 흔한 함정)

> 스튜디오 아나운서 톤으로 합성하면 모델이 **"녹음 환경"** 으로만 진짜·가짜를 구분 → 일상 환경의 깨끗한 사람 음성마저 fake로 잘못 분류.

### 회피해야 할 화자
- 아나운서·뉴스 앵커 톤
- 아동/캐릭터/만화 더빙 톤
- 영어 학습 톤 (한국어 발음 부자연)

### 선호하는 화자
- **일상 대화 톤** — 친구·동료가 말하는 듯
- 망설임·억양 변화·약한 호흡이 자연스러운 화자
- 감정 옵션이 있는 화자라면 `neutral` + 약한 `emotion`으로 단조로움 회피

### 운영
- 합성 본 작업 전에 각 후보 화자별 **샘플 1~2개 sanity check** → 청취 후 선정
- 화자별 분량 균등 분배 (특정 화자 편향 방지)

---

## 6. 양방향 데이터 증강 (★ 학술 기여 핵심)

### 6.1 도메인 갭 문제

| 차이점 | 실제 음성 | AI 합성음 |
|---|---|---|
| 녹음 환경 | 마이크 다양, 생활 소음 | 스튜디오급, 깨끗 |
| 배경 노이즈 | 있음 (선풍기·키보드·차량 등) | 거의 없음 |
| 음량 | 들쭉날쭉 | 일정 |
| 코덱/압축 | 다양 (전화·녹음기 등) | 깔끔한 PCM |

→ 그대로 두면 모델은 "합성 흔적"이 아니라 "녹음 품질"만 학습.

### 6.2 증강 전략

| 대상 | 적용 |
|---|---|
| **합성음 only** | 노이즈 추가(MUSAN 등), **전화망 코덱 시뮬레이션**(8kHz μ-law round-trip), 볼륨 들쭉날쭉화 |
| **진짜 only** | 약한 denoising, 음량 정규화 |
| **양쪽 동일** | SpecAugment, 시간 마스킹, 약한 reverb — anti-spoofing 표준 증강 |

### 6.3 학술 메시지
> "한국어 합성음 탐지의 핵심 도전 과제는 **도메인 갭**이며, 우리는 **양방향 증강 + 전화망 코덱 시뮬레이션**으로 이를 해결했다."

→ 발표·보고서의 메인 슬라이드.

---

## 7. 함정 (반드시 피할 것)

### 7.1 음질·포맷 차이로 인한 치팅
- 진짜·가짜 sample rate·코덱 다르면 모델이 음질 차이만 학습
- **대책**: 진짜·가짜 모두 **16kHz · 16bit · mono PCM** 통일

### 7.2 발화 텍스트 누출
- 같은 문장이 양쪽에 있으면 텍스트 단서 누출
- **대책**: 진짜로 쓴 9,690 발화 텍스트는 합성에 미사용

### 7.3 데이터 분할 누출
- 같은 화자가 train·test 양쪽이면 평가 무의미
- **대책**: speaker-disjoint split (화자 11/3/3) + split 후 화자 ID 교집합 자동 검사

### 7.4 텍스트 정제로 인한 누설 (KsponSpeech 특이)

> 진짜 발화엔 망설임("어/아/음")·숨소리·잡음이 자주 섞여 있음. 가짜는 깨끗 → 모델이 **합성 흔적이 아니라 망설임 유무**로 판단할 위험.

| 패턴 | 처리 | 이유 |
|---|---|---|
| `어/` `아/` `음/` | `/`만 제거 → `어` `아` `음` | 망설임 보존 (가짜에도 자연스럽게 합성됨) |
| `(0.1프로)/(영 점 일 프로)` | 발음형 사용 | TTS가 한글 발음형을 자연스럽게 읽음 |
| `b/` `n/` `o/` `l/` | 제거 | 숨/잡음/겹침/웃음 — 텍스트 표현 불가 |
| `+` `*` | 제거 | 잘림·부정확 마커 |
| < 3자 / > 200자 | 풀에서 제외 | 합성 부적절 / TTS 입력 한도 |

**보완**: 진짜에 `b/`·`n/`·`l/` 마커가 3개 이상인 발화는 train에서 제외.

**정제 후 실제 수치** (2026-05-06 기준, KsponSpeech_01만):
- 진짜: 9,690 → **8,787 발화 / 11.80h** (KsponSpeech_02 일부 추가 후 학습엔 ≈ 15h 사용 예정)
- 합성 풀: 13,310 → **13,180 발화 / 418K자**

### 7.5 도메인 갭으로 인한 치팅 → § 6 양방향 증강

### 7.6 화자 톤 편향 → § 5 화자 선택 가이드

---

## 8. 모델 (3단계 ablation)

### 8.1 점진 비교 구조

| 단계 | 계열 | 구체 구현 | 학습 방식 | 입력 |
|---|---|---|---|---|
| **Baseline 1** | RNN | **BiGRU + FC** | from scratch | mel-spectrogram (raw 1차 변환) |
| **Baseline 2** | CNN | **LCNN** (Light CNN) | from scratch | LFCC (raw 1차 변환) |
| **SOTA** | Self-Sup. / Transformer | **추후 확정** (후보: wav2vec2+AASIST, RawBoost+AASIST, XLSR-AASIST 등) | 사전학습 + fine-tune | **raw waveform 직접** |

> 입력 표기: 모든 단계가 결국 raw waveform에서 출발. mel/LFCC는 전처리 단계, SOTA는 raw 직접 처리.

### 8.2 Ablation 메시지
- **Baseline 1 (GRU)**: 단순 시계열 모델의 EER은?
- **Baseline 2 (LCNN)**: CNN 표현이 GRU보다 얼마나 나은가?
- **SOTA**: 사전학습 표현이 가져오는 이득은? Cross-TTS 일반화는?

→ "왜 더 발전된 모델이 필요한가"를 정량 입증.

### 8.3 보완 카드
- **SpecAugment**: 데이터 부족 보완
- **Backbone freeze**: 메모리·시간 절감
- **Class-balanced loss**: 진짜·가짜 1:1 유지로 일단 불필요하지만 비상용

---

## 9. 평가 지표

- **EER (Equal Error Rate)** — 음성 anti-spoofing 표준 (낮을수록 ↑)
- **AUROC, F1, Precision, Recall** — 일반 분류 지표
- **TTS별 분리 EER** — 합성기별 일반화 정도
- **Cross-domain 평가 (leave-one-TTS-out)** — 학습에 미사용한 TTS로 test → unseen-system 일반화 검증

---

## 10. 연구 질문 (RQ)

| ID | 질문 | 측정치 | 학습 횟수 |
|---|---|---|---|
| **RQ1** | 영어용 사전학습(wav2vec2+AASIST) zero-shot의 한국어 EER은? | EER | 0 (추론만) |
| **RQ2 (핵심)** | GRU → LCNN → SOTA ablation의 EER 개선폭은? | EER, ROC | 3 |
| **RQ3 (★ 학술 기여)** | 양방향 증강 on/off가 EER에 미치는 영향은? | EER (with/without aug) | +3 (best 모델 ablation) |
| **RQ4 (옵션)** | unseen-TTS leave-one-out 시 EER 저하는? | TTS별 EER | best 모델만 |

---

## 11. 환경 및 도구

| 구분 | 환경 |
|---|---|
| 로컬 | 맥북 (메타데이터·시각화) |
| 합성·학습 | Google Colab Pro/Pro+ (T4·V100·A100) |
| 저장소 | Google Drive |
| 코드 | Python, PyTorch, torchaudio, librosa |
| 증강 | torch-audiomentations, MUSAN noise, μ-law 코덱 round-trip |
| TTS API | NCP, Google Cloud, ElevenLabs |

---

## 12. 3주 일정

### Week 1 — 데이터 + 합성 + 증강 파이프라인
- [x] KsponSpeech_01 다운 + 16kHz 통일
- [x] 화자·발화 메타 + speaker-disjoint split (초안, _01만)
- [x] 합성용 텍스트 풀 + 정제 정책
- [ ] **KsponSpeech_02 일부 다운** (화자 5~6명 추가, +3~5h)
- [ ] 진짜 데이터 split 재구성 (목표 train 14 / val 4 / test 4)
- [ ] 화자 sanity check (CLOVA·Google·ElevenLabs 후보 청취)
- [ ] 합성 본 작업 (총 15h)
- [ ] **양방향 증강 파이프라인 작성 + unit 검증** (§ 6)
- [ ] 통합 manifest CSV (진짜·가짜 + aug flag)

### Week 2 — 학습 (RQ1·RQ2 핵심)
- [ ] **RQ1**: wav2vec2+AASIST zero-shot 평가
- [ ] **RQ2**: BiGRU · LCNN · SOTA 각 1번 학습
- [ ] 동일 test set EER · ROC 비교
- [ ] **RQ3**: best 모델 augmentation off 재학습 (ablation)

### Week 3 — 분석 + 발표
- [ ] TTS별 분리 EER, cross-domain leave-one-out (RQ4)
- [ ] ROC · score 분포 · t-SNE 시각화
- [ ] 발표자료 + 보고서

### 폴백 시나리오
- 합성 시간 부족 → CLOVA 비중 ↑ (가장 저렴) + 다른 TTS 비중 ↓
- 학습 시간 부족 → SOTA 우선 + GRU/LCNN은 hyperparam 고정

---

## 13. 비용·시간 견적

| 단계 | 견적 |
|---|---|
| 합성 비용 (총 15h) | **약 3만원** (CLOVA 무료 한도 100만자 안 + Google $300 크레딧 + ElevenLabs ~3만 / Creator 1달) |
| 합성 GPU 시간 | API 호출 위주 (2~3h) |
| 학습 GPU 시간 | **약 12~18h** (T4 기준, 3 모델 + ablation, 데이터 1.5배) |
| 평가·시각화 | 2~3h |
| **합계 GPU** | **약 16~24h** (Colab Pro 한 달 안에서 충분) |

---

## 14. 산출물

- 학습·평가 코드 (재현 가능한 스크립트)
- **양방향 증강 파이프라인** (재현 가능)
- 모델 3개 × EER · ROC 비교표
- TTS별 / Cross-domain EER 분리 표
- 시각화 (ROC, score 분포, t-SNE 임베딩)
- 발표 슬라이드 + 보고서
- (선택) 음성 입력 → 진짜/가짜 데모

---

## 15. 리스크 및 대응

| 리스크 | 대응 |
|---|---|
| Colab 세션 끊김 | 매 epoch 체크포인트 → Drive 저장 |
| TTS API 키 노출 | `.env` 사용, 채팅·코드 미포함 |
| 합성 품질 낮아 탐지 너무 쉬움 | TTS 다양화 + 양방향 증강 |
| 도메인 갭으로 모델이 "녹음 품질"만 학습 | § 6 양방향 증강으로 직접 대응 |
| 데이터 라이선스 (KsponSpeech) | 학내 연구·발표 한정, 외부 공개 금지 |
| Speaker leakage 실수 | split 후 화자 ID 교집합 자동 검사 |

---

## 16. 변경 이력

| 일자 | 변경 |
|---|---|
| 2026-05-07 | **CLOVA 화자 8명 확정** (§ 4.3) — 9명 후보 sanity 청취 후 nsunhee 제외. Pro 3 + NES 5 = 화자당 ≈ 53분 분배. P1 disfluent 텍스트(filler·lengthening·repetition·mid-pause) + RawBoost-style v3 후처리(linear conv·soft clip·impulse·real noise·codec) 청취 검증 통과. |
| 2026-05-06 (저녁) | **데이터 규모 상향**: 진짜 10h+가짜 10h → **진짜 15h+가짜 15h** (총 30h). KsponSpeech_02 일부 추가 다운 필요. TTS 분량 7h/5h/3h로 재배분. split 비율 70/15/15 유지(목표 화자 14/4/4). 비용 ~2.7만원, 합계 GPU ~16~24h. |
| 2026-05-06 (오후) | **방향 재정리**: 가짜 분량 20h → 10h (진짜 10h와 1:1). **§ 5 화자 선택 가이드 신설**(아나운서/스튜디오 톤 회피). **§ 6 양방향 증강 신설**(학술 기여 핵심). **§ 8 모델을 GRU → LCNN → SOTA 점진 ablation 구조로 재구성**(SOTA 구체 모델은 추후 확정). RQ3에 양방향 증강 ablation, RQ4에 cross-domain leave-one-out 추가. 입력을 raw waveform으로 명시. |
| 2026-05-06 | KsponSpeech_01 다운·압축 해제(zip 일부 손상이지만 화자 17명 정상 추출). Speaker-disjoint split 11/3/3 화자 = 9,690 발화 14.87h. 합성용 텍스트 풀 13,310 발화. **§ 7.4 텍스트 정제 정책 신설** — 망설임 보존·이중표기 발음형·비언어 마커 제거. |
| 2026-05-05 | 어린이 데이터(M3까지 진행분) 폐기 → KsponSpeech 성인 한국어로 도메인 변경. 합성기 XTTS·OpenVoice·F5 → CLOVA·Google·ElevenLabs로 교체. voice clone 빼고 generic TTS 분류로 단순화. |
| 2026-05-01 | 최초 어린이 anti-spoofing 기획 |

> 폐기된 어린이 v1 자료는 `archive/children_v1/` 보관.
