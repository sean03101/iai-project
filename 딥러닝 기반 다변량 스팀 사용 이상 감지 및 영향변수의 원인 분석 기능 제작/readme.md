# 딥러닝 기반 다변량 스팀 사용 이상 감지 및 영향변수 원인 분석

> **Deep Learning-Based Multivariate Steam Usage Anomaly Detection and Root Cause Analysis**  
> 용역 결과 보고서 | 2024. 02. 22.

서울과학기술대학교 데이터사이언스학과 | 연구책임자 심재웅

---

## 📌 프로젝트 개요

제지 공정 건조 설비에서 수집되는 다변량 시계열 센서 데이터를 활용하여, **제품 생산을 반영한 스팀 사용량 이상을 사전에 감지**하고 예측에 영향을 미치는 주요 원인 변수를 파악하기 위한 딥러닝 기반 모델을 개발합니다.

세 가지 딥러닝 모델(1D CNN, LSTM, IMV-LSTM)을 구축하고, Feature Importance / SHAP / Attention Score 기반 XAI 방법론을 적용하여 예측 원인 인자를 분석합니다.

---

## 📂 파일 구조

```
├── main.ipynb                  # 전체 작업 흐름 실행 파일
├── utils.py                    # 데이터 전처리 및 변수 생성
├── dataloader.py               # PyTorch DataLoader 변환
├── models/
│   ├── CNN1D.py                # 1D CNN 모델 정의 및 학습
│   ├── LSTM.py                 # LSTM 모델 정의 및 학습
│   └── IMVLSTM.py              # IMV-LSTM 모델 정의 및 학습
└── shap_visualization.py       # SHAP 기반 변수 중요도 시각화
```

---

## 🗂️ 데이터

### 데이터 소개

- **제조 공정**: 제지 생산 공정 중 스팀을 이용한 열처리 공정
- **수집 기간**: 2023년 3월 2일 ~ 2023년 8월 27일 (약 6개월)
- **수집 주기**: 1분
- **규모**: 40개 컬럼, 89,840행

### 주요 변수

| 변수 | 설명 |
|------|------|
| tg04 | 스팀 순간값 |
| tg05 | 추가 스팀값 |
| tg17 | 스팀 누적값 |
| tg20 | 스팀 압력 |
| tg41 | 스팀 온도 |
| tg02 | 종이별 측정 무게 |
| tg03 | 끝단 설비 속도 |
| tg06 | 공기 압력 |
| tg33 | 수분값 |
| ei | 원단위 계산값 (tg04 ÷ (tg03 × tg02 × 0.0004)) |

### 이상(Anomaly) 정의

적응형 관리한계선(UCL/LCL)을 사용하여 이상을 정의합니다.

- **모니터링 기준 컬럼**: `ei` (원단위 에너지사용량)
- **업데이트 주기**: 매 단위 공정
- **과거 기준**: 이전 5개 단위 공정 내 관리한계선 내부 ei
- **중심선**: 과거 ei 평균
- **UCL / LCL**: 중심선 ± 0.15

| 구분 | 개수 |
|------|------|
| 정상 (UCL~LCL 내부) | 82,668개 |
| LCL 미만 (이상) | 2,794개 |
| UCL 초과 (이상) | 3,247개 |

---

## ⚙️ 데이터 전처리

### 측정 오류 제거 (`utils.py – calculate_ei`)

| 변수 | 제거 조건 |
|------|----------|
| tg02 | 10 이하 |
| tg03 | 1000 이하 |
| tg04 | 1 이하 |

총 1,131개 이상치 제거 후 88,709건 사용.

### 파생 변수 생성

- **`jr_progress`**: 각 제품의 생산 시작 시점을 0으로 하여 1분마다 1씩 증가하는 공정 경과 시간
- **`jr_window_patch`**: 시계열 윈도우 내 단위 공정 변화를 식별하는 공정변화도 변수
- **`is_abnormal`**: 관리한계선 기준 라벨링 (0: 정상, 1: LCL 이하, 2: UCL 초과)

### 데이터셋 구축

- **입력 변수**: 과거 30 time step의 38개 센서 데이터 + jr_progress + jr_window_patch + ei + is_abnormal (윈도우 크기 30)
- **목표 변수**: 5분 이후 ei (Regression) / 5분 이후 이상 유무 (Classification)
- **분할 비율**: Train 80% / Test 20%
- **스케일링**: MinMaxScaler

---

## 🧠 모델 구조

### 1D CNN

시계열 데이터를 슬라이딩 윈도우 방식의 1D Convolution으로 처리하여 국소적인 시간 패턴을 추출합니다. SHAP을 통한 변수 중요도 해석이 용이합니다.

### LSTM

게이트 메커니즘(입력/망각/출력 게이트)을 통해 장기 의존성을 학습합니다. Attention 메커니즘을 추가하여 모델이 주목하는 시점을 파악할 수 있습니다.

### IMV-LSTM (Interpretable Multi-Variable LSTM)

변수별 독립적인 hidden state를 학습하고 probabilistic mixture attention을 통해 **변수별/시점별 예측 기여도를 동시에 확인**할 수 있는 해석 가능한 모델입니다.

- **Variable-wise temporal attention (α)**: 각 변수별 시점별 기여도
- **Variable level attention (β)**: 전체 변수 중요도

---

## 🔬 이상 탐지 접근법

두 가지 방식을 비교합니다.

**Regression 접근**: 딥러닝으로 ei 값을 예측 → 예측값이 관리한계선을 벗어나면 이상으로 판단

**Classification 접근**: 이상 라벨(is_abnormal)을 직접 분류 (학습: CrossEntropy Loss)

---

## 🛠️ 실험 설정

### 1D CNN

| 항목 | Regression | Classification |
|------|-----------|----------------|
| Optimizer | Adam | Adam |
| Learning rate | 0.0001 | 0.0001 |
| Loss | L1 Loss | CrossEntropy |
| Epoch | 100 | 100 |
| Input channel | 42 | 42 |
| Output channel | 1 | 3 |

### LSTM

| 항목 | Regression | Classification |
|------|-----------|----------------|
| Hidden dim | 256 | 64 |
| Num layers | 4 | 4 |
| Dropout | 0.4 | 0.4 |

### IMV-LSTM

| 항목 | Regression | Classification |
|------|-----------|----------------|
| Hidden dim | 32 | 32 |
| Learning rate | 0.001 | 0.001 |

---

## 📊 실험 결과

### 전체 모델 성능 비교

| 모델 | Regression Acc | Regression F1 | Classification Acc | Classification F1 |
|------|:--------------:|:-------------:|:-----------------:|:-----------------:|
| 1D CNN | 0.9748 | 0.5681 | 0.9010 | 0.4538 |
| LSTM | 0.9832 | 0.5877 | 0.9619 | 0.4945 |
| **IMV-LSTM** | **0.9843** | **0.6047** | 0.9546 | **0.5102** |

### 주요 분석 결과

- Accuracy/F1에서는 Regression이 높으나, **이상 발생 이후 뒤늦게 탐지하는 경향** 존재
- **Classification 접근법이 이상을 사전에 감지하는 데 더 효과적**
- IMV-LSTM이 두 task 모두에서 가장 균형 있는 성능 달성

### 세부 결과 (IMV-LSTM Classification)

| 클래스 | Precision | Recall | F1 |
|--------|:---------:|:------:|:--:|
| Ok (0) | 0.9601 | 0.9949 | 0.9772 |
| Under LCL (1) | 0.5652 | 0.2241 | 0.3210 |
| Over UCL (2) | 0.6174 | 0.1431 | 0.2324 |
| **Accuracy** | | | **0.9546** |

---

## 🔍 모델 해석 (XAI)

### 주요 영향 변수 요약

모델 전반에 걸쳐 공통적으로 중요하게 나타난 변수:

| 변수 | 설명 | 비고 |
|------|------|------|
| tg17 | 스팀 누적값 | 거의 모든 모델에서 최상위 |
| tg02 | 종이별 측정 무게 | 거의 모든 모델에서 최상위 |
| tg38 | 설비 AE 속도 | 1D CNN, LSTM에서 중요 |
| ei | 원단위 에너지사용량 | Regression에서 높은 중요도 |
| is_abnormal | ei 라벨값 | Classification에서 높은 중요도 |

### 모델별 XAI 방법

| 모델 | XAI 방법 |
|------|---------|
| 1D CNN | SHAP (Feature Importance) |
| LSTM | SHAP + Attention Map (시점별 주의 분포) |
| IMV-LSTM | Variable level attention (β) + Temporal attention map (α) |

### LSTM Attention 패턴

- **Regression**: 가장 최근 시점에 집중
- **Classification**: 초기 시점과 가장 최근 시점 모두에 집중

---

## 🖥️ 환경 설정

### 컴퓨팅 사양

- OS: Ubuntu 20.04
- CPU: AMD Ryzen Threadripper PRO 5955WX (16코어 × 32)
- GPU: NVIDIA GeForce RTX 3090

### 주요 라이브러리

| 라이브러리 | 버전 | 용도 |
|-----------|------|------|
| torch | 1.12.0 | 딥러닝 |
| torchvision | 0.13.0 | 이미지 처리 |
| numpy | 1.24.4 | 수치 연산 |
| pandas | 2.0.3 | 데이터 처리 |
| shap | 0.44.1 | 모델 해석 |
| scikit-learn | 1.3.2 | 머신러닝 도구 |
| matplotlib | 3.7.3 | 시각화 |
| seaborn | 0.13.0 | 통계 시각화 |

### 설치 방법

**Ubuntu**
```bash
# CUDA 11.6 + cuDNN 8.3 설치 후
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.6 -c pytorch -c conda-forge
pip install -r requirements_linux.txt
```

**Windows**
```bash
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.6 -c pytorch -c conda-forge
pip install -r requirements_window.txt
```

### 실행

`main.ipynb` 참고

---

## ✅ 결론 및 고찰

- 적응형 관리한계선 기반으로 이상을 정의하고 Regression/Classification 두 접근법을 비교했으나, **이상 발생 이전 사전 탐지 능력이 부족**한 한계 존재
- ei 계산에 직접 연관된 변수(tg17, tg02, tg38)가 주요 변수로 도출되었으며, 그 외 변수들의 미래 이상 시그널 포함 여부 검토 필요
- **추후 개선 방향**:
  - 제조현장의 '에너지 사용 이상' 상황을 구체적으로 재정의
  - 실시간 현재 시점 이상 탐지 방식 시도
  - 미래 이상의 사전 시그널을 담는 추가 변수 발굴
