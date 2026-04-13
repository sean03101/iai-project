# 변화하는 제조 환경에서의 미세 결함 분류를 위한 테스트 시점 도메인 적응

> **Test-Time Domain Adaptation for Fine-Grained Defect Classification in Changing Manufacturing Environments**  
> 대한산업공학회/한국경영과학회/한국시뮬레이션학회 2025년 춘계공동학술대회

김나연¹, 이성호¹, 심재웅²* | 서울과학기술대학교 데이터사이언스학과¹ · 산업공학과²

---

## 📌 연구 개요

딥러닝 기반 결함 분류 모델은 생산 라인마다 달라지는 조명·각도·배경 등의 환경 변화(Domain Shift)로 인해 성능이 저하됩니다. 재학습 없이 실시간으로 환경 변화에 적응하는 **Fully-Test Time Adaptation(TTA)** 이 가장 실용적인 해결책이지만, 기존 미세 결함 탐지 모델은 복잡한 attention 구조로 인해 TTA 적용이 어렵습니다.

본 연구는 **결함 마스크(GT mask)와 Grad-CAM 정렬 기반의 단일 네트워크** 구조로 두 가지 목표를 동시에 달성합니다.

1. 보조 branch 없이 결함 부위에 정확히 집중하는 단순한 구조 설계
2. 다양한 Fully-TTA 기법과 호환 가능한 구조로 도메인 변화에 안정적으로 대응

---

## 🔍 문제 정의

- 미세 결함은 크기가 작고 희미하여 기존 분류 기반 방법으로 정확한 탐지가 어려움
- 결함과 배경의 경계가 모호하여 모델이 잘못된 영역에 attention을 집중할 가능성 존재
- 단순 ResNet 계열 모델은 지역적 특징 구분에 한계 존재
- 생산 라인마다 별도 모델을 구축하고 재훈련하는 것은 **비용적으로 비현실적**
- **새로운 라벨링 데이터나 재학습 없이 실시간으로 환경 변화에 적응하는 방법론 필요**

---

## 🧠 선행 연구

### Fully Test-Time Adaptation (Fully-TTA)

훈련 시에는 source data만으로 학습하고, 테스트 단계에서 라벨 없는 target data에 기반하여 실시간으로 파라미터를 업데이트하는 방법론.

| Setting | Source Data | Target Data | Train Loss | Test Loss |
|---------|-------------|-------------|------------|-----------|
| Fine-tuning | - | $x^t, y^t$ | $L(x^t, y^t)$ | - |
| Domain Adaptation | $x^s, y^s$ | $x^t$ | $L(x^s, y^s) + L(x^s, x^t)$ | - |
| Test-time Training | $x^s, y^s$ | $x^t$ | $L(x^s, y^s) + L(x^s)$ | $L(x^t)$ |
| **Fully-TTA** | - | $x^t$ | - | $L(x^t)$ |

주요 Fully-TTA 방법론:
- **TENT**: 테스트 배치의 예측 엔트로피를 최소화하는 방향으로 BatchNorm 파라미터를 실시간 업데이트
- **EATA**: 엔트로피가 낮고 중복 정보가 많은 샘플은 학습에서 제외하고, Fisher Information으로 중요 파라미터를 파악하여 안정적으로 적응
- **DeYO**: 객체의 shape을 변형했을 때 예측이 크게 달라지는 샘플을 정보량이 높은 샘플로 간주하여 선택적으로 학습

### 미세 결함 탐지

- **Attention 기반**: CBAM, Grad-CAM 기반 attention 정확성 향상 방법 등
- **형태 정보 활용**: GT segmentation mask로 결함 위치를 학습하는 two-stage 구조(Tabernik et al.), mask 기반 attention map을 활용한 fine-grained classification(MGANet)
- **한계**: 기존 연구 대부분이 복잡한 attention module 구조로 TTA와의 연계가 고려되지 않아 domain shift 환경 적용이 어려움

---

## 🏗️ 제안 방법

### 프레임워크 구조

```
[image]     ──┐
[mask image]──┤── Shared Network ──► FC layer ──► CE loss   (Warm UP)
[Grad-CAM]  ──┘                  └──► KL div loss           (Train)
     ▲                                    │
     └────────────────────────────────────┘
```

### 학습 절차

**1단계: Warm UP (50 epochs)**
- 일반 이미지만 입력하여 CE Loss로 기본 분류 학습
- Grad-CAM이 안정적으로 형성되도록 사전 학습
- ✔ 기본적인 분류 능력 확보 및 안정적인 attention 형성

**2단계: Train**
- 이전 epoch에서 생성된 Grad-CAM 이미지와 defect mask를 각각 모델에 입력하여 feature 추출
- 추출한 feature에 GAP + softmax를 적용하여 확률 분포로 정규화
- KL Divergence Loss + CE Loss로 학습
- ✔ 모델이 주목하는 영역(Grad-CAM)이 실제 결함(mask)과 일치하도록 정렬

### 최종 학습 Loss

$$L_{\text{total}} = \alpha \cdot L_{\text{CE}}(f(x_{\text{img}}), y) + (1-\alpha) \cdot L_{\text{KLD}}\bigl(\text{softmax}(\text{GAP}(g(f(x_{\text{mask}})))),\ \text{softmax}(\text{GAP}(g(f(x_{\text{cam}}))))\bigr)$$

| 기호 | 의미 |
|------|------|
| $f(\cdot)$ | Feature extractor |
| $g(\cdot)$ | Classifier |
| $x_{\text{img}}$ | 원본 제품 이미지 |
| $x_{\text{mask}}$ | 이미지에 대응되는 결함 마스크 이미지 |
| $x_{\text{cam}}$ | 이전 epoch에서 추출된 CAM 히트맵을 정규화하여 만든 3채널 이미지 |
| $\alpha$ | 가중치 조절 파라미터 |

---

## 🧪 실험

### Dataset: D-SUB 커넥터 결함 데이터

국내 제조 AI 기업의 모니터 D-SUB 커넥터 제품 이미지 데이터로, 1개의 정상 클래스와 5종의 결함 클래스로 구성됩니다.

| 클래스 | 설명 |
|--------|------|
| ok | 정상 |
| Dent | 검은 표면의 움푹 팬 흔적 |
| Scratch | 예리한 것에 의한 긁힘 |
| Pin | 핀이 휘거나 변형된 상태 |
| F.M | 핀 홀 내 이물질 |
| Glue | 검은 표면에 부착된 끈적한 물질 |

환경 변화는 총 **13개 도메인**으로 구성됩니다 (Default + Color/Brightness/Focus × 4단계):

| 환경 | 설명 |
|------|------|
| Color | 조명 색상 변화 |
| Brightness | 조명 밝기 변화 |
| Focus | 카메라 포커스 흔들림 |

### Experiment Setting

| 항목 | 설정 |
|------|------|
| Backbone | ResNet-101 |
| Optimizer | SGD (lr=0.01, momentum=0.9, weight decay=0.0005) |
| Scheduler | Cosine Annealing |
| Batch size | 32 |
| Epochs | 100 (Warm UP 50 + Train 50) |
| 평가지표 | AUROC |
| 반복 횟수 | 3회 |

### Baseline 비교 모델

| 모델 | Fine-grained | TTA | TTA 방식 |
|------|:---:|:---:|------|
| ResNet101 | ✗ | ✗ | - |
| ResNet101 + TENT | ✗ | ✓ | TENT |
| MGANet | ✓ | ✗ | - |
| MGANet + TENT | ✓ | ✓ | TENT |
| **Ours** | ✓ | ✗ | - |
| **Ours + TENT** | ✓ | ✓ | TENT |
| **Ours + EATA** | ✓ | ✓ | EATA |
| **Ours + DeYO** | ✓ | ✓ | DeYO |

---

## 📊 실험 결과 (AUROC)

| 도메인 | ResNet101 | ResNet101+TENT | MGANet | MGANet+TENT | Ours | Ours+TENT | Ours+EATA | Ours+DeYO |
|--------|:---------:|:--------------:|:------:|:-----------:|:----:|:---------:|:---------:|:---------:|
| No drift | 0.9236 | - | 0.9385 | - | 0.9442 | - | - | - |
| Color 0 | 0.8869 | 0.9161 | 0.9494 | 0.9159 | 0.9388 | 0.9583 | **0.9583** | 0.9556 |
| Color 1 | 0.9008 | 0.9271 | 0.9294 | 0.9238 | 0.9422 | 0.9506 | **0.9528** | 0.9520 |
| Color 2 | 0.8397 | 0.9305 | 0.8429 | 0.9031 | 0.9317 | 0.9509 | **0.9522** | 0.9516 |
| Color 3 | 0.8132 | 0.9183 | 0.8600 | 0.9245 | 0.9425 | 0.9484 | **0.9509** | 0.9494 |
| Brightness 0 | 0.8084 | 0.9017 | 0.8168 | 0.8727 | 0.9140 | 0.9405 | 0.9463 | **0.9465** |
| Brightness 1 | 0.8694 | 0.9124 | 0.8866 | 0.8928 | 0.9282 | 0.9456 | **0.9497** | 0.9484 |
| Brightness 2 | 0.7253 | 0.9319 | 0.8531 | 0.8994 | 0.9402 | 0.9463 | **0.9486** | 0.9479 |
| Brightness 3 | 0.5032 | 0.9127 | 0.5048 | 0.8344 | 0.8620 | 0.9354 | **0.9404** | 0.9376 |
| Focus 0 | 0.8774 | 0.9084 | 0.8178 | 0.9101 | 0.9404 | 0.9433 | **0.9486** | 0.9464 |
| Focus 1 | 0.9136 | 0.9177 | 0.9227 | 0.9147 | 0.9526 | 0.9500 | **0.9522** | 0.9512 |
| Focus 2 | 0.8822 | 0.9010 | 0.9164 | 0.9221 | 0.9204 | 0.9343 | **0.9393** | 0.9376 |
| Focus 3 | 0.7922 | 0.8499 | 0.8509 | 0.8643 | 0.9092 | 0.9169 | **0.9194** | 0.9171 |

TTA 적용 시, 모델이 실제 결함 부위에 더 정확히 집중하며 도메인 변화에도 안정적으로 적응함을 CAM 시각화를 통해 확인.

---

## ✅ 결론

- 보조 branch나 추가 module 없이 **단일 네트워크**로 attention 영역을 결함 부위에 정확히 정렬
- CAM 기반 attention map과 결함 마스크 간의 logit 정렬로 모델이 실제 결함 영역에 집중하도록 유도
- ResNet 기반 BatchNorm을 활용하는 다양한 **Fully-TTA 기법(TENT, EATA, DeYO)과 효과적으로 결합 가능**
- 실험을 통해 정확한 결함 탐지와 안정적인 도메인 적응을 동시에 달성함을 입증

---

## 📚 References

- Wang et al. (2021). TENT: Fully test-time adaptation by entropy minimization. *ICLR*.
- Niu et al. (2022). Efficient test-time model adaptation without forgetting. *ICML*.
- Lee et al. (2024). DeYO: Entropy is not enough for test-time adaptation. *ICLR*.
- Woo et al. (2018). CBAM: Convolutional block attention module. *ECCV*.
- Tabernik et al. (2020). Segmentation-based deep-learning approach for surface-defect detection. *Journal of Intelligent Manufacturing*.
- Wang et al. (2021). MGANet: Mask guided attention for fine-grained patchy image classification. *ICIP*.

---

## 💰 연구 지원

본 연구는 2025년도 정부(산업통상자원부)의 재원으로 한국산업기술진흥회의 지원을 받아 수행되었음 (P0017123, 2025년 산업혁신인재성장지원사업).
