# Environmentally Robust Defect Classification with Domain Augmentation Framework

## Abstract
Visual defect classification is a critical process in manufacturing systems, aiming to achieve high-quality production and reduce costs. Recently, deep learning-based defect classification models have achieved significant success. However, the performance of these models can be significantly reduced due to variations in manufacturing environments across multiple production lines. These variations, not present in the training data, result in a domain gap between training and test data. To address this challenge, we propose a domain augmentation framework for constructing a robust defect classification model. This model can deliver high performance across various manufacturing environments using a training dataset from only a single production line. The proposed framework first creates multiple augmented domains using image transformation functions. Then, a defect classification model is trained using a multi-source domain generalization (DG) method with these augmented domains. This approach mitigates the single-source DG problem to a multi-source DG problem, enabling the adoption of multi-source DG methods, which leads to performance improvements. The effectiveness of the proposed framework is demonstrated  through experiments using a dataset provided by a Korean manufacturing company.


제조 시스템에서 결함 분류는 고품질 생산을 달성하고 비용을 절감하기 위한 중요한 과정입니다. 최근에는 딥러닝 기반의 결함 분류 모델이 중요한 성공을 거두고 있습니다. 그러나 다양한 제조 환경의 변화로 인해 이러한 모델의 성능이 크게 떨어질 수 있습니다. 여러 생산 라인에서 발생하는 이러한 변화는 훈련 데이터에는 없는 도메인 간격을 야기하여 테스트 데이터에서 훈련 데이터로의 간격을 만듭니다. 이러한 도전을 해결하기 위해, 우리는 단일 생산 라인의 훈련 데이터셋만을 사용하여 다양한 제조 환경에서도 높은 성능을 낼 수 있는 강건한 결함 분류 모델을 구축하는 도메인 증강 프레임워크를 제안합니다. 제안된 프레임워크는 먼저 여러 증강 도메인을 이미지 변환 기능을 사용하여 생성합니다. 그런 다음 이러한 증강된 도메인을 사용하여 다중 소스 도메인 일반화(DG) 방법으로 결함 분류 모델을 훈련합니다. 이 접근법은 단일 소스 DG 문제를 다중 소스 DG 문제로 완화시켜 다중 소스 DG 방법의 채택을 가능하게 하여 성능 향상을 이끌어냅니다. 제안된 프레임워크의 효과는 한국의 제조 회사가 제공한 데이터셋을 사용한 실험을 통해 입증되었습니다.


## How to install & run

1. **Install requirements.txt and PyTorch 1.7.1**
   - Open your terminal or command prompt.
   - Navigate to your project directory.
   - Run the following command to install the required packages from `requirements.txt`:
     ```sh
     pip install -r requirements.txt
     ```
   - To install PyTorch 1.7.1, run the following command:
     ```sh
     pip install torch==1.7.1
     ```

2. **Perform offline augmentation using augmentation functions in the `aug` folder**
   - Ensure that you have the augmentation functions available in the `aug` folder.
   - Use these functions to perform offline data augmentation. This can typically be done by writing a script that imports the augmentation functions and applies them to your dataset.
   - An example of how you might write such a script:
     ```python
     import os
     from aug import some_augmentation_function  # replace with actual function names

     data_dir = 'path/to/your/data'
     augmented_data_dir = 'path/to/save/augmented/data'

     if not os.path.exists(augmented_data_dir):
         os.makedirs(augmented_data_dir)

     for img_name in os.listdir(data_dir):
         img_path = os.path.join(data_dir, img_name)
         augmented_img = some_augmentation_function(img_path)
         augmented_img.save(os.path.join(augmented_data_dir, img_name))
     ```

3. **Run the method in the `baseline` code and the methods in `/dg_lib/Transfer-Learning-Library/examples/domain_generalization`**
   - First, navigate to the `baseline` directory and run the code present there. This might involve executing a script or running a Jupyter notebook.
     ```sh
     cd path/to/baseline
     python baseline_script.py  # replace with the actual script name
     ```
   - Then, navigate to the `domain_generalization` directory in the Transfer Learning Library examples and run the relevant method scripts.
     ```sh
     cd /dg_lib/Transfer-Learning-Library/examples/domain_generalization
     python method_script.py  # replace with the actual script name
     ```



## 서론

### 연구 배경
- 머신 비전(Machine Vision)
  - 기계에 인간이 가지고 있는 시각과 판단 기능을 부여하는 기술
  - 사람의 시각 및 지각 기능을 하드웨어와 소프트웨어가 탑재된 시스템이 대신 처리
  - 시스템의 구성 요소로 검사물, 조명, 렌즈, 카메라, 프로세서, 소프트웨어가 존재
  - 제조 공정에서 머신 비전은 딥러닝 도입을 통해 제품 품질과 전반적인 시스템 효율성을 크게 개선

### 연구 동기

![image](https://github.com/sean03101/iai-project/assets/59594037/8de72ea4-6d3d-455c-b7e1-2c0e43a528b1)


- 제조 현장에서 각 생산 라인에서 불량 검출을 위해 촬영하는 환경은 조명의 세기와 종류, 카메라의 위치와 각도, 장애물의 존재 등으로 인해 미세하게 다름
    - 각 생산 라인에서 촬영된 이미지들이 **생산 라인마다 각자 다른 도메인을 가지는 문제 발생**
- 딥러닝 기반의 머신 비전은 훈련 단계에서 학습한 인스턴스와 다른 환경에서 수집한 인스턴스가 들어올 경우, **도메인 변화(domain shift)**로 인해 성능 저하가 발생
- 다른 라인에서 수집한 인스턴스 추가적인 라벨링, 촬영 환경 하드웨어 표준화 등 다양한 해결책을 모색했지만 상당한 비용 부담이라는 큰 단점이 존재


### 연구 목적
**일부 제조 환경에서 관측된 한정적인 인스턴스, 특히 단일 제조 라인에서 수집한 인스턴스만 활용해 다양한 환경에서 강건한 성능을 지닌 머신 비전 개발**

## 제안하는 연구 방법

![image](https://github.com/sean03101/iai-project/assets/59594037/92b4e974-8a27-45ef-913a-5f625959e9e1)

- 단일 환경에서 수집한 인스턴스를 활용해 다양한 환경 변화에서 적용할 수 있는 강건한 모델을 만들기 위해 일반적으로 단일 소스 도메인 일반화(Single source domain generalization)을 사용
- 본 연구는 단일 소스 데이터셋에 오프라인 데이터 증강을 사용하고, **각 증강 방법을 하나의 도메인**으로 설정하는 'domain augmentation' 프레임워크 제안
- 제안하는 프레임워크에서의 학습 데이터는 기존의 도메인 뿐만 아니라 ‘증강된 도메인’(augmented domain)을 가지기에 **단일 소스 도메인 일반화가 아닌 다중 소스 도메인 일반화 문제로 치환** 가능



![image](https://github.com/sean03101/iai-project/assets/59594037/7fb3f7cb-ee19-4ba7-b019-91f7f06875c1)




![image](https://github.com/sean03101/iai-project/assets/59594037/949ffda5-1d1c-405d-8a57-05d199aea84a)


## 실험 환경 및 데이터셋
### D-SUB connect dataset 설명
카테고리 정보
  - 실험 데이터 셋은 코그넥스 회사에서 제작한, D-SUB 커넥터(전기 및 전자 장치에 사용되는 D 형태의 다양한 핀 수를 가진 표준 커넥터)에 관련된 제조 데이터 셋
  - 데이터 셋의 카테고리는 6개 존재 (정상(ok) + 5개의 결함 종류)
    
![image](https://github.com/sean03101/iai-project/assets/59594037/a81a67b7-d22b-4130-8cac-f083de9af3dc)


도메인 정보
  - D-sub 커넥터 데이터 셋은 4종류의 환경에 따라 관측된 이미지로 구성
  - 각 환경은 환경이 변화하는 정도에 따라 5단계의 서브 조건을 가짐
  - 총 13개의 도메인(default + Lighting, Brightness, Cameraz 마다 4개) 존재
    
![image](https://github.com/sean03101/iai-project/assets/59594037/75c85f7d-1a87-44b2-818d-b45b6073d0e4)



학습/ 검증/ 테스트 데이터셋

![image](https://github.com/sean03101/iai-project/assets/59594037/b0886363-c266-4e0b-ab42-45f4aef87968)


이미지 증강 방법
  - 제조 현장에서 자주 발생하는 8가지의 상황을 가정, 즉 8개의 가상 도메인을 추가로 생성

    이미지 증강 방법 적용 예시
![image](https://github.com/sean03101/iai-project/assets/59594037/198d1fbd-5c94-4ce7-90c6-458689ffe42e)


## 실험 결과
### 실험1
**증강 도메인 별 도메인 일반화 기여도**

![image](https://github.com/sean03101/iai-project/assets/59594037/7ffdbbf3-bf5e-4a4a-839a-8ab73704ea96)


- 표는 단일 증강 도메인(Augmented domain) 사용했을 때, 방법에 대한 성능 비교 결과값
- 표의 값들은 5회 반복 실험에서의 평균 정확도 × 100 값이며 이에 대한 표준 편차는 ± (해당 표준 편차 값)로 표현
- 모든 방법론의 실험 결과를 종합한 결과 기여도 순서는 Brightness > Camera > Flip > Lightness > Contrast > Blur > Dropout > Noise 순서
- 대부분의 방법론의 기여도 순서는 거의 비슷하지만 baseline과 VReX 방법은 예외로 순서가 다르며 좋은 성능을 보이는 도메인 증강 방법과 좋지 않는 성능을 보이는 증강 간의 차이가 확연히 큰 것을 알 수 있음
- 또한, 방법론 전반적으로 기여도 순서가 낮은 증강 도메인일수록 표준 편차가 커지는 사실을 알 수 있음


### 실험2
**증강 도메인 개수에 따른 성능 변화**

![image](https://github.com/sean03101/iai-project/assets/59594037/6ac97d2d-aeca-4d94-b198-2089a34b2625)


- 증강 도메인 별 도메인 일반화 기여도가 높은 순서대로 증강 도메인을 추가했을 때, 성능 변화에 대한 실험
- 실험 결과 증강 도메인의 개수가 3-4개 정도일 때 가장 좋은 성능을 보이다가, 도메인 일반화 기여도가 낮은 도메인이 추가될수록 성능이 점차 하락하는 모습을 보임
- 앞선 실험 결과와 유사하게 VReX를 제외한 다른 멀티 DG 모델과 달리, baseline과 VReX 모델은 기여도가 낮은 증강 도메인이 추가될수록 성능이 급격하게 하락하는 모습을 보임
- 데이터셋의 종류 및 성질에 따라 어떤 증강 방법을 도입하는지에 따라 도메인 일반화 성능이 바뀌며, 제안하는 프레임워크는 기존의 일반적인 방법과 다르게 증강 방법에 대해 강건하게 학습 가능하다는 사실을 알 수 있음

  

### 실험3
**타겟 도메인에 따른 정확도(Accuracy)**

![image](https://github.com/sean03101/iai-project/assets/59594037/bb3b3c1a-a539-40d6-87df-9121eef869fe)


- 5번의 반복 실험 결과 모든 방법 중 제안하는 프레임워크의 IBN-net 모델이 가장 좋은 성능을 보임
- 전반적으로 가상 도메인 증강을 활용한 일반화 방법이 다른 방법에 비해 좋은 성능을 보임(평균 정확도 0.86)
- 단일 소스 도메인 일반화 방법은 ‘Brightness’ 도메인에 대해 약점을 가지고 있지만 다중 소스 도메인 일반화는 ‘Cameraz’ 도메인에 대한 성능이 안 좋은 차이점을 가지고 있음
- 단일 소스 도메인 일반화 방법보다 온라인 단순 증강 기법을 활용한 부분이 더 좋은 성능을 보이고 있음(다양한 가상의 분포에 대한 인스턴스를 학습하는 것이 보다 효과적이었다고 추측)


![image](https://github.com/sean03101/iai-project/assets/59594037/49ab9db7-9641-48fc-811a-4c2a12b6e952)


- 5번의 반복 실험 중에서 모델의 정확도가 가장 높았던 모델에 대한 혼동 행렬(confustion matrix)
- ‘Lcondition’ 도메인만 예측을 진행했을 때, 앞선 ‘repeat’ 도메인과 비교해서 혼동 행렬들은 성능 차이가 더욱 심해짐 (정확도 : 0.9025 / 0.8215 / 0.942)
- 도메인 일반화는 제안하는 방법(도메인 증강 프레임워크)에서 가장 잘 일어난다는 것을 알 수 있음



## 결론

- 제조 현장의 머신 비전은 제한된 환경의 데이터만 학습하여 다양한 환경에서 강건한 예측을 진행하는 모델이어야 함

- 제한된 제조 환경에서 보다 좋은 모델 개발을 위해 제한된 환경 데이터 셋을 증강하고 증강한 방법을 하나의 도메인으로 취급하는 “도메인 증강(Domain Augmentation)” 프레임워크 제안 

- 실험을 통해 다음과 같은 사실을 알 수 있었음
  - 오프라인 이미지 증강을 도메인으로 취급하는 ‘가상 도메인 증강’ 방식을 사용하여 단일 소스 도메인 문제 상황을 다중 소스 도메인 문제 상황으로 변환 할 수 있다
  - 오프라인 이미지 증강 기반 다중 소스 도메인 생성 및 일반화 방법은 기존 도메인 뿐 아니라 처음 접하는 다양한 도메인 또한 좋은 성능을 보이는 강건한 모델을 학습할 수 있게 만들어 준다는 것을 알 수 있다

- 제안하는 프레임워크는 Domainbed 혹은 도메인 일반화에 분야에서의 SOTA 등 최신의 다중 소스 도메인 일반화 방법(method)를 사용할 수 있으며 장점을 가지고 있음


## 설치 방법

- 개발결과물 사용설명서.docx 및 requirement.txt 참고

![image](https://github.com/sean03101/iai-project/assets/59594037/679e02e1-23e4-44b6-a1ce-c4c9564d43d4)


![image](https://github.com/sean03101/iai-project/assets/59594037/b1f50116-de5b-4b42-ac15-c0be210c1fec)



## 참고 출처

- [Transfer Learning Library](https://github.com/thuml/Transfer-Learning-Library) : 제안하는 프레임워크를 검증하기 위해 **여러 DG method가 존재하는 open library에서 D-sub connector dataset 전처리, dataloader 구성 및 시각화 부분을 추가**
