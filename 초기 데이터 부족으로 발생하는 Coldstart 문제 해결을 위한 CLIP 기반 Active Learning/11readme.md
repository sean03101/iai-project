# 초기 데이터 부족으로 발생하는 Coldstart 문제 해결을 위한 CLIP 기반 Active Learning

> **Active Learning Initialization Using CLIP for Visual Defect Detection**  
> 2023 한국데이터마이닝학회 추계학술대회

배소희†, 이성호†, 심재웅* | 서울과학기술대학교 데이터사이언스학과

---

## 📌 연구 개요

딥러닝 기반 불량 검출 모델은 대량의 라벨링 데이터를 필요로 하지만, 라벨링 비용 문제로 인해 Active Learning 도입이 필요합니다. 그러나 기존 Supervised 기반 Active Learning은 **초기 학습 데이터 부족(Cold-start)** 으로 인해 성능이 제한됩니다.

본 연구는 **CLIP의 Zero-shot 성능을 활용한 Knowledge Distillation**으로 Cold-start 문제를 해결하고, 초기 Labeled data 없이도 정보력 있는 샘플을 선택할 수 있는 프레임워크를 제안합니다.

---

## 🔍 문제 정의

- 기존 Supervised 기반 Active Learning은 초기 학습 데이터 부족 시 성능 제한
- 불량 데이터가 부족한 경우 샘플 선택 효율성 저하
- One-Class 기반 불량 검출은 정상 데이터만으로 구성된 정제 데이터 구축에 별도 라벨링 비용 발생
- **Labeled data가 0인 상태에서도 불량 클래스의 정보력 높은 샘플을 선택할 수 있는 방법론 필요**

---

## 🧠 선행 연구

### CLIP (Contrastive Language–Image Pre-training)
이미지와 텍스트 임베딩 간의 코사인 유사도를 최적화하여 Contrastive Learning을 통해 다양한 Vision Language Task에서 효과적인 표현을 학습하는 모델. 추가 학습 없이 새로운 카테고리 분류가 가능한 **Zero-shot** 능력 보유.

- **ZegCLIP**: CLIP의 Zero-shot 예측 능력을 픽셀 수준으로 확장, Unseen Class에 대한 Semantic Mask 생성
- **WinCLIP**: 다양한 불량 정보의 Prompt Ensemble과 다중 윈도우 슬라이싱을 결합한 Zero-shot Anomaly Detection

### Knowledge Distillation
복잡한 Teacher Model이 단순한 Student Model을 지도하는 기법. Student Model이 Teacher Model을 모방하여 경량화 및 성능 향상을 동시에 달성.

### Active Learning
최소한의 라벨링으로 모델 성능을 최대화하는 방법론.
- **Cold-start 문제**: 초기 모델이 정보력 있는 데이터를 선택하기 어려워 성능 향상 제한
- PT4AL, ALCS 등 기존 Cold-start 해결 방법론 존재하나 초기 Labeled data 필요

---

## 🏗️ 제안 방법

### 프레임워크 구조

```
Unlabeled Pool ──► CLIP (Teacher)
                        │
                  Knowledge Distillation
                        │
Labeled Pool ──► Classification Model ──► k samples ──► Oracle
      ▲                                                     │
      └─────────────────────────────────────────────────────┘
```

### CLIP with Prompt Ensemble

WinCLIP의 Prompt 구조를 활용한 Zero-shot Anomaly Detection:

- **Normal prompts**: "flawless", "perfect", "unblemished", "without flaw", "without defect", "without damage", "good", "normal"
- **Anomaly prompts**: "damaged", "with flaw", "with defect", "with damage" + {defect type}
- **최종 구조**: `"a" + {state level} + "photo of" + {categories}`

Anomaly Score 계산:

$$p(\mathbf{s} = s_i | \mathbf{x}; \mathbf{s} \in S) := \frac{\exp(\langle f(\mathbf{x}), g(s_i) \rangle / \tau)}{\sum_{s \in S} \exp(\langle f(\mathbf{x}), g(s) \rangle / \tau)}$$

### Knowledge Distillation

| Loss | 수식 |
|------|------|
| Labeled Dataset Loss | $\mathcal{L}_{label} = \mathcal{L}_{CE}(y, \text{Classification}(x))$ |
| Unlabeled Dataset Loss | $\mathcal{L}_{unlabel} = \mathcal{L}_{KD}(\text{CLIP}(x/T), \text{Classification}(x/T))$ |
| Total Loss | $\mathcal{L} = \sum \alpha T^2 \mathcal{L}_{unlabel} + (1-\alpha)\mathcal{L}_{label}$ |

### Active Learning 절차

1. KD된 Student Model을 Active Learning의 초기 Classification Model로 사용
2. Labeled data 수 = 0인 상태에서 Unlabeled Pool로 cycle 시작
3. **Least Confidence** 방식으로 K개 샘플 선택 후 라벨링
4. 종료 조건 만족 시까지 cycle 반복

> 매 cycle마다 Labeled data가 추가되며 Labeled Data Loss의 가중치가 점진적으로 증가

---

## 🧪 실험

### Dataset: MVTec AD

- 산업 이미지 이상 탐지용 벤치마크 데이터셋
- 15개 카테고리 (Texture + Object)
- Train: 3,629장 (정상) / Test: 1,725장 (정상 + 이상)
- 본 실험에서는 Train/Test 병합 후 0.8 : 0.2로 재분할

### Experiment Setting

| 항목 | 설정 |
|------|------|
| Teacher Network | CLIP (ViT-B/32) |
| Student Network | Pretrained ResNet18 |
| Temperature (T) | 0.2 |
| Alpha (α) | 0.01 |
| KD Epoch | 25 |
| Query Strategy | Least Confidence |
| Query 수 (K) | 1 |
| Cycle 수 | 30 |
| 평가지표 | AUROC, F1 Score |
| 반복 횟수 | 3회 |

---

## 📊 실험 결과

### Prompt Ensemble 효과

단일 Prompt 대비 Prompt Ensemble 사용 시 전체 카테고리 평균 AUROC 향상:

| 방법 | 평균 AUROC |
|------|-----------|
| 단일 Prompt | 0.5807 |
| Prompt Ensemble | **0.7754** |

### Active Learning 결과

- 제안 방법(CLIP 기반)이 전체 cycle에 걸쳐 Random 초기화 대비 일관되게 높은 AUROC 및 F1 score 달성
- Labeled data 10개 수준에서 Cold-start 문제 해결 확인
- Labeled data 1~2개 구간의 일시적 성능 하강은 CLIP의 Zero-shot 성능이 few-shot보다 우수하기 때문

---

## ✅ 결론

- CLIP의 지식 증류를 활용한 Active Learning으로 **Cold-start 문제 해결**
- Labeled data가 없는 상황에서도 자연어 기반 불량 정보만으로 정보력 있는 샘플 선택 가능
- 초반 성능뿐 아니라 전체 학습 과정의 성능 향상에도 긍정적 영향 확인

### 추후 연구
- 다양한 산업 제조 데이터셋으로 일반화 검증
- Few-shot Prompt Learning을 통한 자동화된 Prompt 엔지니어링 적용

---

## 📚 References

주요 참고문헌은 다음과 같습니다.

- Radford et al. (2021). Learning transferable visual models from natural language supervision. *ICML*.
- Jeong et al. (2023). WinCLIP: Zero-/few-shot anomaly classification and segmentation. *CVPR*.
- Settles, B. (2009). Active learning literature survey.
- Bergmann et al. (2019). MVTec AD. *CVPR*.

---

## 💰 연구 지원

본 연구는 산업통상자원부 (P0017123, 2023년 산업혁신인재성장지원사업) 및 과학기술정보통신부 (No. RS-2022-00165783)의 지원을 받아 수행되었습니다.
