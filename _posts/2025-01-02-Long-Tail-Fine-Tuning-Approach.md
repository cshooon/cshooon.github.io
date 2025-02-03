---
title: Long-Tail Loss Function Approach
date: 2025-01-02 15:13:35 +09:00
categories: ['AI']
tags: ['ICD code', 'Long-Tail', 'Fine-Tuning']
use_math: true
---

# Long-Tail Learning with Foundation Model: Heavy Fine-Tuning Hurts

## 0. Abstract

Long-Tail Learning에서 Fine-Tuning이 성능에 어떤 영향을 주는지는 구체적으로 밝혀지지 않음. 
Heavy Fine-Tuning은 tail class의 성능을 떨어뜨림. 

Heavy Fine-Tuning: 전체 중에서 대부분의 parameters를 학습시킴.
Lightweight Fine-Tuning: 전체 중에서 작은 일부분의 parameters을 학습시킴.

본 논문에서는 LIFT를 제안함.

## 1. Introduction

Long-Tail Learning은 sample 수가 많은 head class보다 sample 수가 적은 tail class의 성능이 떨어지는 문제를 다룸. tail class의 성능을 높이고 Head class의 성능을 유지하는 것이 목표임.

선행 연구에서 Scratch에서부터 학습을 하는 것보다 Fine-Tuning을 한 경우가 성능이 더 좋다고 알려짐. 본 논문에서는 CLIP 모델을 foundation model로 사용함. **아래는 CLIP을 Fine-Tuning한 모델임.**

1. BALLAD: first fully fine-tunes the foundation model, then freezes the backbone and optimizes a linear adapter on the re-sampled data.
2. VL-LTR: incorporates additional image-text web data during the fine-tuning process.
3. RAC: jointly fine-tunes an encoder and trains a retrieval module to augment the input
image with external datasets such as ImageNet-21K.
4. LPT: fine-tunes the foundation model utilizing prompt tuning via two-phrase training.

그렇지만, **Long-Tail Learning에서** Fine-Tuning의 영향은 알려지지 않음. 

본 논문의 Contributions은 다음과 같음.

1. Limitation in heavy fine-tuning distorts the tail-class performance.
2. Optimizing a small proportion of parameters helps.
3. LIFT: **Low complexity** (fewer Learnable Params) and **accurate** method.
4. LIFT outperforms the state-of-the-art methods with a **lower computational cost**.

![Figure 1.](/assets/posts_images/ai-ft/accuracy.png)
Figure 1. 
원 크기는 epoch 수를 의미. 

## 2. Long-Tail Learning with Foundation Model

Transformer Architecture는 computer vision and natural language processing tasks에서 검증되었음.

CLIP compares the **cosine similarity** between the image and each of the class prompts

$$
\begin{equation}
    y_{\text{pred}} = \arg\max_{k \in [K]} \frac{\mathbf{f}_I^\top \mathbf{f}_T^k}{\|\mathbf{f}_I\|_2 \|\mathbf{f}_T^k\|_2}.
\end{equation}
$$

${f}_I$: Image Feature

${f}_T^k$: Texture Feature, k candidate classes

Zero-shot CLIP은 학습을 진행하지 않고, Inference를 수행함. Zero-shot CLIP은 conventional method 보다 대체로 성능이 좋음. backbone을 freeze하고 추가로 classifier를 학습해 성능이 향상됨. 그렇지만 세 가지 데이터셋 모두에서 head class에 비해 tail class의 성능이 많이 떨어짐.

![Figure 2.](/assets/posts_images/ai-ft/foundation.png)

## 3. Heavy Fine-Tuning Hurts

CLIP을 사용하여 Head class의 성능은 개선되었지만, tail class의 성능은 Head class만큼 개선되지 않음.

> 💡 Is the Fine-Tuning not sufficient?

Zero-shot: 학습 X
Classifier Fine-Tuning: classifier만 학습.
Full Fine-Tuning: classfier + foundation model까지 전부 학습.

![Figure 3.](/assets/posts_images/ai-ft/fine-tuning.png)

Figure 3. 
Left: inter-class feature similarities Right: intra-class distributions from tail class

**Cosine similarity**:  -1, 반대 방향 / 0, orthogonal  / 1, 같은 방향

Inter-class feature similarities: 클래스 간에 존재하는 feature의 유사도를 나타냄.

Intra-class distributions: 같은 클래스 내의 샘플들이 얼마나 분산되어 있는지를 나타냄.

(b) Classifier Fine-Tuning: ~~Inter-class feature similarities~~**,** Intra-class distributions

(c) Full Fine-Tuning: Inter-class feature similarities, ~~Intra-class distributions~~

(b)의 경우 Train과 Test 분포가 유사함 (brown), unoptimized (red). **y축은 샘플 수**를 의미함.

(c)의 경우 optimized (green) 되었지만 Train과 Test의 분포가 다름. 이는 다음 2가지 문제를 일으킴.

1. Overfitting (blue)
    1. Train의 분포가 0에 쏠림.
2. Underestimated (orange)
    1. Test 분포가 1 방향으로 넓게 퍼짐.

Underestimated

$P_s(y)$: source distribution, long tail

$P_t(y)$: target distribution. uniform

$P_s(x \mid y=j)$, $P_t(x \mid y=j)$: Label Shift

$P_s(x \mid y=j)$ 에서 tail class의 경우 빈도가 적어 작게 추정됨.

$P_t(x \mid y=j)$ 에서 source distribution보다 tail class 빈도가 높음. 크게 추정됨.

$$
\zeta_{s-t}(j) = \frac{P_s(y = k \mid x)}{P_t(y = k \mid x)}
$$

$$
\begin{align}
    L(x, y = j) &= -\log P_s(y = j \mid x) \\
    &= -\log \frac{\exp(z_j + \log P_s(y = j) + \log \zeta_{s-t}(j))}{\sum_{k \in [K]} \exp(z_k + \log P_s(y = k) + \log \zeta_{s-t}(k))},
\end{align} 
$$

Loss underestimated. tail class를 예측할 때 penalty를 받아, head class라고 추정하기 쉬움.

tail class에서의 성능 문제를 해결하기 위해, 선행 연구에서는 two-stage training을 제안함.

그러나 위 방법은 다음과 같은 문제가 있음.

1. Significant Training Overhead (훈련 시간/비용 증가)
2. External Data Requirement (외부 데이터 필요성)

따라서 본 논문에서는 LIFT를 제안함.

## 4. Efficient and Accurate Long-Tail Learning

### 4.1 Lightweight Fine-Tuning Helps

Intra-class distributions로 인한 왜곡을 완화하기 위해, 학습 가능한 parameter 수를 제한함.

$W \in \mathbb{R}^{d_1 \times d_2}$, a specified proportion $\alpha$, a sparse 0-1 mask $M \in \{0, 1\}^{d_1 \times d_2}$

used to **control the optimized parameters**

$$
XW \rightarrow X(W \odot M) + X(W \odot (1 − M))
$$

element-wise product를 수행함. mask M이 1이면 학습을 진행하고 0이면 학습을 진행하지 않음.

$X(W \odot (1 − M))$ 항은 detached gradient(freeze)로 학습을 진행하지 않음. 모델의 출력값에 영향을 주지 않게 학습 전의 값 그대로 사용함. 

$\alpha$가 작을 때(e.g. 0.1%) 성능 향상이 컸음. $\alpha$가 커지면 성능이 하락함.

![Figure 5.](/assets/posts_images/ai-ft/lightweight.png)

the optimized parameters are selected **arbitrary** → remarkable improvements.

full Fine-Tuning과 비교했을 때 head class 성능 유사, tail class 성능 좋음.

### 4.2 The Proposed Fine-Tuning (LIFT) Method

LIFT (well structured lightweight fine-tuning) optimizes inter-class feature similarities 

Linear classifier $z_j = w_j^\top f + b \cdot \|w_j\|$ , j가 tail class인 경우 값이 작아 편향됨.

→ cosine classifier을 사용함. $z_j = \sigma \cdot \frac{\mathbf{w}_j^\top \mathbf{f}}{\|\mathbf{w}_j\|_2 \|\mathbf{f}\|_2}$, $\sigma$: scaling factor

**LA loss**

$$
L_{\text{LA}}(x, y = j) = -\log \frac{\exp(z_j + \log P(y = j))}{\sum_{k \in [K]} \exp(z_k + \log P(y = k))}
$$

$y = j$ represents the ground-truth label of $x$ and $z_j$ is the predicted logit.

### 4.3 Semantic-Aware Initialization

선행 연구에 따르면, uninitialized classifier는 Fine-Tuning에 부정적 영향을 미침. classifier의 initial state를 잘 설정해야 함. 

~~re-weighted or LA loss / class mean feature~~

**Because** requires extracting features, scarce tail class에 적용 안 됨.

→ semantic-aware initialization (SAI)

prompt about class, hand-crafted format

strength: single forward pass, low complexity

텍스트 인코더를 학습시키지 않음. 텍스트 인코더는 classifier weights를 초기화할 때만 사용됨.

Inference에는 image encoder, classifier만 사용됨.

### 4.4 Test-Time Ensembling

Test, Inference generalization 성능 향상을 위해 아래 방법을 적용함.

$$
z = \log P(y | x) = \frac{1}{M} \sum_{i=1}^{M} \log P(y | \alpha_i(x))
$$

- x는 테스트 데이터 포인트 (e.g. image).
- $\alpha_i(x)$는 x의 여러 가지 perturbed versions.
- M은 생성된 perturbed versions의 개수.

## 6. Conclusion

heavy fine-tuning in distorting tail-class performance

1. arbitary lightweight fine-tuning
2. LIFT (well structured lightweight fine-tuning)
    1. convergence in fewer than 20 epochs, **no need for any external data**
    2. outperforms baseline methods
