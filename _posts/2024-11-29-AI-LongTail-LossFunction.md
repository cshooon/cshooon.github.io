---
title: Long-Tail Loss Function Approach
date: 2024-11-29 19:10:20 +09:00
categories: ['AI']
tags: ['ICD code', 'Long-Tail', 'Loss Function']
---

# A Unified Generalization Analysis of Re-Weighting and Logit-Adjustment for Imbalanced Learning

## 1. Introduction

현실 데이터셋은 일반적으로 불균형함. (Long-tailed)
단순한 ERM (Empirical Risk Minimization) process는 majority classes에 편향됨.
→ minority classes을 일반화하기 어려워짐.

이 문제를 해결하기 위해, minority classes 학습에 초점을 맞추도록 **loss function**을 수정함.

1. re-weighting the losses
assign larger weights to the losses of the miniority classes
→ instability in optimization
Deferred Re-Weighting (DRW), appled only during the terminal phase of training
2. adjusting the logits by class-dependent terms
Label Distribution Aware Margin (LDAM) loss enforces larger margins 
Logit-Adjustment (LA) loss, Class-Dependent Temperatures (CDT) loss 
utilize additive and multiplicative terms
Vector-Scaling (VS) combine two type of terms

기존 방법으로 성능이 개선되었지만, 이를 뒷받침하는 이론이 부족하거나 불완전함. 

- Prior Research
    
    Cao et al. [12]과 Ren et al. [18]은 LDAM loss, classic margin theory
    → DRW scheme의 성능 향상을 설명하지 못함.
    
    Menon et al. [13]은 addictive terms in LA loss을 Fisher consistency로 분석함.
    → loss의 일반화 분석은 제공되지 않음.
    
    Kini et al. [16]은 VS loss의 일반화 분석을 제공하지만, 이 결과는 선형 모델이 선형적으로 분리 가능한 데이터에 대해 학습되는 경우에만 곱셈 항(multiplicative terms)의 역할을 설명할 수 있음.
    
    VS loss가 DRW scheme과 상당히 호환되지 않음을 발견함. 
    **There is a gap between the theory and the practice of the loss-modificiation approaches**
    

Main contribution

1. Lipschitz continuity를 확장함. (data-dependent contraction)
→ imbalanced learning의 fine-grained 일반화 경계를 얻음.
2. 이러한 fine-grained 경계를 바탕으로 re-weighting, logit-adjustment를 통합된 방식으로 설명함.
3. 이러한 이론적 기반을 바탕으로 principled learning algorithm을 개발함.
→ DRW phase와 multiplicative logit-adjustment 충돌을 방지함.
4. 벤츠마크 데이터셋 실험: 이론적 결과 검증하고 제안된 방법의 효과도 입증함.

## 2. Preliminary

### 2.1 Notations and Problem Definition

The samples are drawn i.i.d. (independent and identically distributed)
from product space $Z = X \times Y$

$x$는 input, $y = \{1, ..., C \}$

$S = \{(x^{(n)}, y^{(n)}\}^N_{n=1}$ imbalnaced training set

sampled from the imbalnace distribution $D$ defined on $Z$

$S_y = \{x \mid (x, y) \in S\}$

클래스 y의 샘플 수: $N_y = |S_y|$
클래스 비율:  $\pi_y = N_y / N_1$

- 클래스들은 $N_1 \geq N_2 \geq \dotsb \geq N_C$의 순서를 가진다고 가정함.

**$D_{\text{bal}}$:** balanced distribution defined on $Z$

클래스 y를 레이블 공간 $Y$에서 균등하게 샘플링함.
해당 클래스의 조건부 분포 $D_y = P[x \mid y]$에서 x를 샘플링함.

**목표**: **스코어 함수** $f: X \rightarrow \mathbb{R}^C$를 학습하여 **균형 분포**에서의 **리스크**  $R_{\text{bal}}(f)$를 최소화함.

**리스크 정의**:

$$
 ⁍
$$

$R_y(f)$는 클래스 y에 대한 risk, M은 모델의 성능을 평가하는 측도임.

ex) $M(f(x), y) = \mathbf{1}[y \notin \arg\max_{y' \in Y} f(x)_{y'}]$

$\mathbf{1}[\cdot]$: indicator function

M은 일반적으로 미분 가능하지 않아 최적화하기 어려움.
미분 가능한 loss function $L: \mathbb{R}^C \times Y \rightarrow \mathbb{R}^+$를 사용하여 **리스크**를 정의함.

$$
⁍
$$

hypothesis set $\mathcal{G} = \{L \circ f \mid f \in F\}$로 정의함.

**Vector-Scaling(VS) loss function**

$$
⁍
$$

재가중치(re-weighting)와 로짓 조정(logit adjustment)를 일반화함.
Cross Entropy Loss: $\alpha_y = 1, \beta_y = 1, \Delta_y = 0$ 

Class Balanaced Loss: 
$\beta_y = 1, \Delta_y = 0$ 이고, 
re-weighting terms $\alpha_y = \pi_y^{-1}$ 또는 $\alpha_y = \frac{1 - p}{1 - p^{N_y}}$, $p \in (0, 1)$이면 

LA Loss: 
$\alpha_y = 1, \beta_y = 1, \Delta_y = \tau \log \pi_y, \tau > 0$

CDT(Class-Dependent Tempering) Loss: 
$\alpha_1 = 1, \beta_y = (N_y / N_1)^\gamma, \Delta_y = 0, \gamma > 0$

**Fisher Consistency:** 
손실 함수를 최소화했을 때 모델이 실제 데이터 분포의 참 값을 정확하게 예측할 수 있는지를 나타냄.

 **a subset of the VS loss family** satisfies such a property:

$$
⁍
$$

**$f(x)_y$**: 입력 x에 대해 모델이 예측한 클래스 y의 점수

**비율 조정 $\frac{\delta_y}{\pi_y}$**
클래스의 사전 확률 $\pi_y$로 나누어주어 **소수 클래스에** 집중해 학습함.

**합산 부분($\sum_{y' \ne y} \cdots$)**

실제 클래스 y가 아닌 다른 모든 클래스 y′에 대해 모델의 예측 점수 차이를 지수 함수로 계산함.
이 값이 클수록(즉, 잘못된 클래스의 점수가 높을수록) loss가 커짐.

$\delta_y$는 임의의 양의 상수임.

minimizing $R^L_{bal}(f)$ not only can put more emphasis on minority classes,
but also helps bound $R_{bal}(f)$

### 2.2 Existing Generalization Analysis for Imbalanced Learning

In balanced learning, we can directly minimize the empirical balanced risk defined on the balanced datasets $S_{bal}$ sampled from $D_{bal}$:

$$
\hat{R}{_\text{bal}}^L(f) := \frac{1}{N} \sum_{(x, y) \in S_{\text{bal}}} L(f(x), y)
$$

The generalization guarantee is available by **traditional concentration techniques**

However, in imbalanced learning, we can only minimize the empirical risk on the imbalanced dataset S:

$$
\hat{R}^L(f) := \frac{1}{N} \sum_{(x, y) \in S} L(f(x), y)
$$

**can’t minimize balanced risk.**

**명제 1 (불균형 학습을 위한 합집합 경계)**

함수 집합  $F$와 손실 함수  $L : \mathbb{R}^C \times Y \rightarrow [0, M]$이 주어졌을 때, 임의의  $\delta \in (0, 1)$에 대해,  training set $S$에 대한 확률이 최소 $1 - \delta$로, 모든 $g \in G$에 대해 다음의 일반화 경계가 성립함.

$$
R_{\text{bal}}^L(f) = \frac{1}{C} \sum_{y=1}^C R_y^L(f) \leq \frac{1}{C} \sum_{y=1}^C \hat{R}_y^L(f) + \hat{\mathcal{C}}_{S_y}(\mathcal{G}) + 3M \sqrt{\frac{\log 2C/\delta}{2N_y}}  \,\,\,\ (7)
$$

- **$R_{\text{bal}}^L(f)$**: 모델 $f$의 균형된(클래스별로 동일하게 가중치를 준) risk
- **$R_y^L(f)$**: 클래스 $y$에 대한 실제 리스크.
- **$\hat{R}_y^L(f)$**: 클래스 $y$에 대한 경험적(학습 데이터 기반) 리스크.
- **$\hat{\mathcal{C}}_{S_y}(G)$**: 클래스 $y$에 대한 함수 집합 $G$의 경험적 복잡도.
- **$N_y$** : 클래스 $y$의 샘플 수.
- **$M$**: 손실 함수의 최대값으로, 손실 함수의 값이 $[0, M]$ 범위에 있음을 의미함.

모델의 balanced risk <= 클래스별 경험적 리스크와 복잡도, 그리고 샘플 수에 따른 확률론적 항의 합

이 합집합 경계에는 다음과 같은 한계가 있음.

1. 이론적으로, 이 일반화 경계는 거칠고(sharp하지 않고) 충분히 세밀하지 않음. 
2. 구체적으로, 서로 다른 손실 함수들 간의 차이는 $\alpha_y, \beta_y, \Delta_y$의 선택에 있음. 그러나 증명에서 사용된 손실 함수 $L$의 유일한 속성인 Lipschitz 연속성은 전역적(global) 특성을 가지므로 이러한 차이점을 흐리게 만듦.  

$$
⁍
$$

$R_{\text{bal}}^L(f)$를 직접 상한화할 수 있다면 더 날카로운(sharper) 경계를 얻을 수 있을 것임.

LDAM 손실이 CE 손실보다 성능이 우수하지만, 그 향상은 그렇게 크지 않음. 
학습의 마지막 단계에서  $\alpha_y = \frac{1 - p}{1 - p^{N_y}}$, $p \in (0, 1)$로 설정하는 DRW을 결합하면, 성능이 향상됨. 
그러나  proposition 1은 이 현상을 설명하지 못함.

## 3. Fine-Grained Generalization Analysis for Imbalanced Learning

### 3.2 Application to the VS Loss

**Proposition 3 (Data-Dependent Bound for the VS Loss).**

함수 집합 $F$와 VS 손실 함수 $L_{\text{VS}}$ 가 주어졌을 때, 임의의 $\delta \in (0, 1)$에 대해, training set $S$에 대한 확률이 최소 $1 - \delta$로, 모든 $f \in F$에 대해 다음의 일반화 경계가 성립함.

$$
R_{\text{bal}}^{L}(f) \leq \Phi(L_{\text{VS}}, \delta) + \frac{\hat{\mathcal{C}}_S(F)}{C\pi_C} \sum_{y=1}^C \alpha_y \tilde{\beta}_y \sqrt{\pi_y} [1 - \text{softmax}(\beta_y B_y(f) + \Delta_y)] 
$$

여기서 $\Phi(L, \delta) := \frac{1}{C\pi_C} \left[ \hat{R}^L(f) + 3M \sqrt{\frac{\log (2/\delta)}{2N}} \right]$는 $S$에서의 경험적 리스크와 $\delta$ 항을 포함함.

M: loss function L의 최댓값

$\tilde{\beta}y := \sqrt{ \beta_y^2 + \left( \sum{y' \neq y} \beta_{y'} \right)^2 } ;$ $\text{softmax}(\cdot)$는 소프트맥스 함수를 나타냄. 

$B_y(f)$는 실제 클래스 $y$에 대한 최소 예측값으로, $B_y(f) := \min_{x \in S_y} f(x)_y$임.

 $\pi_C := \frac{N_C}{N}, N_1 \geq N_2 \geq \dotsb \geq N_C$임. 따라서, 이 보조정리는 모델 성능이 데이터의 불균형 정도에 어떻게 의존하는지를 보여줌. C는 전체 클래스 수.

**local Lipschitz 상수 $\mu_y = \alpha_y \tilde{\beta}_y [1 - \text{softmax}(\beta_y B_y(f) + \Delta_y)],$** 

**(In1) Why re-weighting and logit-adjustment are necessary?**

imbalanced 데이터에서는 클래스별 샘플 수가 크게 다르기 때문에, 모델의 **일반화 성능**이 클래스마다 **불균형함**.
 $\sqrt{\pi_y}$와 $B_y(f)$ 항 때문임.

$*α_y$는 $π_y$*의 영향을 상쇄하여 **클래스 간 일반화 성능의 불균형을 완화함**.
소수 클래스에 더 큰 $*α_y*$를 부여함.

$*β_y*$, $Δ_y$는 $*B_y(f)*$의 클래스 간 불균형을 조정함.
모델의 출력 값을 조정하여 소수 클래스의 예측 성능을 향상시킵니다.

**(In2) Why the deferred scheme is necessary?**

소수 클래스의 가중치를 높이는 것은 데이터의 분포가 불균형할 때 최적화 과정에서 어려움이 있음. 
→ 학습 초기 단계에서는 $\alpha_y = 1$, 마지막 단계에서는 $\alpha_y = \frac{1 - p}{1 - p^{N_y}}$, $p \in (0, 1)$로 설정함.

re-weighting loss는 소수 클래스에서의 최적화를 도아주지만, 다수 클래스의 성능 향상에는 해로움. (그림3). 
다수/소수 클래스는 각각 상대적으로 작은/큰 $B_y(f)$를 가지며, 일반화 경계는 더욱 느슨해짐. 

DRW 스킴에서는 학습 초기 단계에서 $\alpha_y = 1$임. 이러한 워밍업 단계는 모델이 다수 클래스에 집중하도록 도와주고, 소수 클래스의 가중치를 높인 후에도 다수와 소수 클래스 모두에 대해 작은 $B_y(f)$를 가짐. 

**(In3) How does our result explain the design of existing losses?**

re-weighting loss의 경우, $\alpha_y$는 $\pi_y$가 증가함에 따라 감소해야 함.
$\alpha_y = \pi_y^{-1}$ $\alpha_y = \frac{1 - p}{1 - p^{N_y}}$, $p \in (0, 1)$로 설정된 balanced loss와 일치함. 
In2에서 알 수 있듯이, $\alpha_y = 1$일 때 $B_y(f)$ 는 $\pi_y$에 따라 증가함. 
→ logit adjusment loss의 경우, $\beta_y$와 $\Delta_y$ 모두 $\pi_y$가 증가함에 따라 증가해야 함. 
LDAM loss $\Delta_y \propto -N_y^{-1/4}$ , logit adjusment loss $\Delta_y = \tau \log \pi_y$, CDT loss $\beta_y = (N_y / N_1)^\gamma$

**(In4) Are re-weighting and logit-adjustment fully compatible?**

(a) No. 
re-weighting terms $\alpha_y$는 $\pi_y$에 따라 감소하는 반면, multiplicative logit adjustment terms $\beta_y$는 $\pi_y$에 따라 증가함. → $\tilde{\beta}_y$는 $\alpha_y$의 효과를 약화시킴.
(b) $\alpha_y$는 **additive** logit adjsutment terms $\Delta_y$와 호환됨.

### 3.3 Principled Learning Algorithm induced by the Theoretical Insights

1. (In1)-(In3)에 따르면, 일반화 경계를 개선하기 위해 재가중치(re-weighting), 로짓 조정(logit-adjustment), 그리고 DRW(Deferred Re-Weighting) 기법을 종합적으로 활용하는 것이 중요함. 
2. (In4)에 따라, $α_y$와 $β_y$ 사이의 충돌을 피하기 위해 **Truncated Logit-Adjustment(TLA)** 기법을 제안함. 이 스킴에서는 학습 초기 단계에서 $β_y$가 $π_y$에 따라 증가하지만, 학습 마지막 단계에서는 $β_y$를 1로 잘라냄(truncated). 
3. $α_y$를 $α_y ∝ π^{(-ν)}_y, ν > 0$로 설정하여 $α_y$와 $\sqrt{π_y}$를 정렬(aligned)시키는데, 이를 Aligned DRW(ADRW)라고 명명함. 이러한 re-weighting 스킴은 **Fisher consistency를** 따름.

| **손실 함수/기법** | **$α_y​$** | **$β_y$​** | **$Δ_y$​** | **설명** |
| --- | --- | --- | --- | --- |
| **Cross Entropy (CE)** | 1 | 1 | 0 | 기본 손실 함수로, 클래스 간 재가중치나 로짓 조정을 하지 않음. |
| **Class Balanced (CB)** | $\pi_y^{-1}$​ 또는 $\frac{1 - p}{1 - p^{N_y}}$$p \in (0, 1)$ | 1 | 0 | 재가중치를 통해 소수 클래스의 중요도를 높임. |
| **Logit Adjustment (LA)** | 1 | 1 | $\tau \log \pi_y$ $\tau > 0$ | 덧셈적 로짓 조정을 통해 클래스 불균형을 보정. |
| **Class-Dependent Temperatures (CDT)**  | 1 | $(\frac{N_y}{N_1})^\gamma$
$\gamma > 0$ | 0 | 곱셈적 로짓 조정을 통해 클래스 불균형을 보정. |
| **Vector-Scaling (VS)** | 필요에 따라 설정 | 필요에 따라 설정 | 필요에 따라 설정 | 재가중치와 로짓 조정을 모두 포함하는 일반화된 손실 함수. |
| **Deferred Re-Weighting (DRW)** | 초기: 1, 후반: $\frac{1 - p}{1 - p^{N_y}}$$p \in (0, 1)$ | 1 | 0 | 훈련 후반부에 재가중치를 적용하여 소수 클래스의 중요도를 높임. |
| **Aligned DRW (ADRW)** | $\pi_y^{-\nu}$, $\nu > 0$ | 1 | 0 | 클래스 비율에 따라 재가중치를 세밀하게 조정하여 일반화 경계를 개선. |
| **Truncated Logit Adjustment (TLA)** | DRW or ADRW | 초기: $\pi_y$에 비례, 후반: 1 | 필요에 따라 설정 | 훈련 후반부에 $\beta_y$​를 고정하여 재가중치와의 충돌을 방지. |


## 5. Conclusion

imbalanced learning에서 loss function 에 대한 **통합된 일반화 분석**을 제시함.

- **세밀한 분석**을 통해 **re-weighting**와 **logit adjustment**의 역할을 밝히고, 기존 이론으로 설명되지 않는 실험적 현상도 설명함.
- **이론적 통찰**에 기반한 **원칙적인 학습 알고리즘**을 제안함. (ADRW, TLA)
