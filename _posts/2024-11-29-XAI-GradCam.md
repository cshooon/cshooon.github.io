---
title: XAI Grad-CAM
date: 2024-11-29 19:10:20 +09:00
categories: ['AI']
tags: ['XAI', 'Image', 'Grad-CAM']
use_math: true
---

# Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization


XAI(eXplainable AI)는 **인공지능(AI) 모델의 결정을 사람이 이해할 수 있게 만드는 방법론**을 의미함.

> ✨ What makes a good visual explanation?


**good** visual explanation in image classification

1. class-discriminative (이미지에서 해당 카테고리가 존재하는 위치를 식별.)
2. high-resolution (pixel 단위로 detail 포착.)

![그림 1](/assets/posts_images/ai-gradcam/feature_image.png)

그림 1

그림 1-(b)와 1-(h)의 경우 fine-grained detail을 잘 포착하지만 class-discriminative하지 않음. (두 이미지의 클래스가 Cat과 Dog로 다르지만 비슷함.) 이러한 문제를 해결하고자  Class Activation Mapping(CAM)이라는 방법이 고안됨. 1-(c), 1-(i)를 보면 Cat과 Dog에 해당하는 위치를 잘 식별함.

그렇지만 CAM 방법론은 모델의 구조 변경이 필요하고 구조를 변경함으로써 성능이 미미하게 하락한다는 단점이 존재함.

본 논문에서는 모델의 구조를 변경하지 않으면서 CAM을 사용할 수 있는 방법론을 제안함. 

1. 모델의 구조를 변경하지 않으면서 객체의 위치를 추출하는 **Grad-CAM** 을 제시함.
2. 높은 해상도의 이미지에서 pixel을 추출하는 방법인 **Guided Grad-CAM** 을 제시함.

### CAM

CAM의 목표는 이미지를 분류할 때 이미지의 어느 부분을 보는지 Class Activation MAP을 추출하는 것임. 

![그림 2](/assets/posts_images/ai-gradcam/cam.png)

그림 2

1. $F^k$: feature map의 글로벌 평균 풀링 값. (GAP)
    
    /[
    \begin{align} F^k = \frac{1}{Z} \sum_{i,j} A^k_{i,j} \end{align}
    ]/
    
    $k$: k번째 feature map, 인덱스
    
    $Z$: feature map의 pixel 수 (i x j)
    
    $A^k_{i, j}$: A는 feature map, 위치 (i, j)
    
2. $Y_c$: 클래스 c에 대한 score
    
    $$
    \begin{align} Y_c = \sum_k w^c_k F^k \end{align}
    $$
    
    식을 전개하면: 
    
    $$
    \begin{align} Y_c = \sum_k w^c_k \left(\frac{1}{Z} \sum_{i,j} A^k_{i,j}\right) =  \sum_{i,j} \frac{1}{Z} \sum_k w^c_k  A^k_{i,j} \end{align}
    $$
    
    $w^c_k$: k번째 feature map에 대한 c번째 클래스 가중치
    
3. $M^c_{i,j}$: 클래스 c에 대한 (i, j) 위치의 피처 맵에서의 활성화 정도. 모델이 이미지의 어느 부분을 중요하게 판단하는지를 보여줌.

/[
\begin {align} M^c_{i,j} = \frac{1}{Z}\sum_k w^c_k \cdot A^k_{i,j} \end{align}
]/

/[
Y_c = \sum_{i,j} M^c_{i,j}
]/

### Grad-CAM

CAM을 구하려면 $w^k_c$가 추출되어야 함. GAP 이후 FC layer 한개만 연결되어 있는 형태여야만 위 수식을 이용하여 CAM을 추출할 수 있음. FC 레이어가 여러 층이거나 활성화 함수(ReLU)가 중간에 있는 경우 하나의 가중치 행렬 $w_c^k$ 로 나타낼 수 없음. (비선형)

![cam2.png](/assets/posts_images/ai-gradcam/cam2.png)

식 (4)에서 $w^c_k$ 대신에 gradient를 이용해 계산한 $\alpha^c_k$ 사용!

/[
\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{ij}^k}
]/

**ReLU**를 적용하고, Localization Map L을 생성함.

/[
L_{\text{Grad-CAM}}^c = \text{ReLU} \left( \sum_k \alpha_c^k A^k \right)
]/

### Guided Grad-CAM

**Guided Backpropagation** 

입력 이미지의 각 픽셀이 모델 출력에 기여하는 정도를 기울기(gradient)로 계산함. 이 과정에서 ReLU 활성화 함수의 순방향과 역방향을 수정하여, 음수 값을 제거함. **+ 값만 강조!**

**원래 이미지와 동일한 해상도**의 시각화 맵을 생성함. 하지만 **class-discriminative x**

**Grad-CAM**
**class-discriminative**, feature map과 gradients로 히트맵을 생성하므로 **해상도가 떨어짐.**


> 💡 두 가지 방법의 장점만 가져오기 위해 element-wise product 수행!

![final.png](/assets/posts_images/ai-gradcam/final.png)
