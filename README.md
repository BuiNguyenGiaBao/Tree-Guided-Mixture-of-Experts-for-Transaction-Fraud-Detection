# Tree-Guided Mixture-of-Experts for Transaction Fraud Detection

## Overview

This project proposes a hybrid **Tree-Guided Mixture-of-Experts (MoE)** framework for transaction fraud detection on highly imbalanced tabular data.

The framework combines the strengths of tree-based machine learning and deep neural networks. A LightGBM model is used as a teacher, while two neural experts learn complementary representations:

- **CNN Expert** for nonlinear and local feature interactions
- **DeepFM Expert** for first-order, second-order, and higher-order feature interactions

A tree-guided gating network adaptively combines the two expert representations. Knowledge from the tree-based teacher is transferred to the neural model through **Knowledge Distillation**.

The final proposed model further integrates **Focal Loss** and **Supervised Contrastive Learning** to improve fraud-class discrimination and representation quality.

---

## Dataset

The experiments are conducted on the **IEEE-CIS Fraud Detection** dataset.

The processed data are divided chronologically into:

- Training set
- Validation set
- Internal test set

The preprocessing pipeline includes:

- Missing-value handling
- Numerical feature scaling
- Categorical encoding
- Frequency-based features
- Group-level statistical features
- Feature alignment
- Memory optimization
- Train-fitted preprocessing objects
- Cost-aware evaluation metadata

The preprocessing procedure is designed so that transformation parameters are learned from the training data and subsequently applied to validation and test data.

---

## Proposed Architecture

The proposed framework contains two complementary neural experts.

### CNN Expert

The CNN branch is designed to learn nonlinear interaction patterns from transaction features.

Main components include:

- Feature projection
- Pseudo-sequence construction
- 1D convolutional layers
- Attention pooling
- Low-rank bilinear interaction
- Latent feature representation

The CNN representation is denoted as:

$$
z_{\mathrm{CNN}}
$$

---

### DeepFM Expert

The DeepFM branch captures different levels of feature interaction through:

- Linear feature effects
- Factorization Machine interactions
- Deep nonlinear interactions
- Categorical embeddings
- Multilayer perceptron

The DeepFM representation is denoted as:

$$
z_{\mathrm{DeepFM}}
$$

---

## Tree-Guided Mixture-of-Experts

LightGBM is used as the tree-based teacher model.

For a transaction $x$, the teacher produces a fraud probability:

$$
p_T = P(y=1 \mid x)
$$

which is represented as a teacher logit:

$$
l_T = \log \frac{p_T}{1-p_T}
$$

The gating network receives:

$$
z_{\mathrm{CNN}},
\quad
z_{\mathrm{DeepFM}},
\quad
l_T
$$

and estimates adaptive expert weights:

$$
g =
\operatorname{Softmax}
\left(
\operatorname{MLP}
\left[
z_{\mathrm{CNN}}
\Vert
z_{\mathrm{DeepFM}}
\Vert
l_T
\right]
\right)
$$

The expert weights are:

$$
g =
[g_{\mathrm{CNN}}, g_{\mathrm{DeepFM}}]
$$

with:

$$
g_{\mathrm{CNN}} + g_{\mathrm{DeepFM}} = 1
$$

The fused representation is:

$$
z_{\mathrm{fused}}
=
g_{\mathrm{CNN}} z_{\mathrm{CNN}}
+
g_{\mathrm{DeepFM}} z_{\mathrm{DeepFM}}
$$

The final classifier predicts the fraud probability from the fused representation.

---

## Proposed Models

### Model 1

**Tree-Guided MoE + Knowledge Distillation + Focal Loss**

The objective function is:

$$
\mathcal{L}_{M1}
=
\mathcal{L}_{\mathrm{Focal}}
+
\lambda_{\mathrm{KD}}
\mathcal{L}_{\mathrm{KD}}
$$

Model 1 is used as an intermediate architecture in the ablation analysis.

---

### Model 2

**Tree-Guided MoE + Knowledge Distillation + Focal Loss + Supervised Contrastive Learning**

The final objective function is:

$$
\mathcal{L}_{M2}
=
\mathcal{L}_{\mathrm{Focal}}
+
\lambda_{\mathrm{KD}}
\mathcal{L}_{\mathrm{KD}}
+
\lambda_{\mathrm{SC}}
\mathcal{L}_{\mathrm{SupCon}}
$$

Model 2 is the final proposed model.

---

## Loss Functions

### Focal Loss

Focal Loss is used to reduce the influence of easy majority-class samples and place greater emphasis on difficult fraud examples.

$$
\mathcal{L}_{\mathrm{Focal}}
=
-\alpha_t (1-p_t)^\gamma \log(p_t)
$$

---

### Knowledge Distillation

Knowledge Distillation transfers information from the LightGBM teacher to the neural student.

The student is encouraged to produce predictions consistent with the teacher probability distribution while still optimizing the fraud classification objective.

A general form of the distillation loss is:

$$
\mathcal{L}_{\mathrm{KD}}
=
T^2
\cdot
\mathrm{BCE}
\left(
\frac{z_S}{T},
\sigma\left(\frac{z_T}{T}\right)
\right)
$$

where:

- $z_S$ is the student logit
- $z_T$ is the teacher logit
- $T$ is the distillation temperature

---

### Supervised Contrastive Learning

Supervised Contrastive Learning is applied to the learned representation space.

Its objective is to:

- Pull samples from the same class closer together
- Push samples from different classes farther apart
- Improve fraud and non-fraud representation separation

The contrastive objective operates on the fused representation:

$$
z_{\mathrm{fused}}
$$

---

## Machine Learning Baselines

The following traditional machine learning models are evaluated:

- Logistic Regression
- HistGradientBoosting
- XGBoost
- CatBoost
- LightGBM

LightGBM is also used as the tree-based teacher in the proposed framework.

---

## Deep Learning Baselines

The following deep learning models and architectural variants are evaluated:

- DeepFM
- CNN
- CNN + DeepFM
- Mixture-of-Experts
- MoE + Focal Loss
- MoE + Focal Loss + Supervised Contrastive Learning
- FT-Transformer
- TabM-like
- TabR-like
- SAINT-like

The TabM-like, TabR-like, and SAINT-like models are experimental reproduced implementations used for comparative evaluation.

---

## Evaluation Metrics

### Standard Classification Metrics

The project reports:

- PR-AUC
- ROC-AUC
- Precision
- Recall
- F1-score
- Matthews Correlation Coefficient
- Recall at Precision ≥ 0.80

PR-AUC is treated as the primary ranking metric because fraud detection is a highly imbalanced classification problem.

---

## Cost-Aware Evaluation

The project additionally evaluates fraud detection under limited transaction-review budgets.

The following review budgets are considered:

- 1%
- 3%
- 5%
- 10%
- 15%
- 20%

For each review budget, the following metrics are reported:

- Precision@K
- Recall@K
- F1@K
- Lift@K
- Captured Fraud Amount Rate
- Expected Utility

---

## Main Results

| Model | PR-AUC | F1 | MCC | Recall@P≥0.80 |
|---|---:|---:|---:|---:|
| LightGBM | 0.5431 | 0.5212 | 0.5042 | 0.3480 |
| Model 1 | 0.5366 | 0.5260 | 0.5101 | 0.3500 |
| Model 2 | 0.5354 ± 0.0024 | 0.5313 ± 0.0048 | 0.5185 ± 0.0052 | 0.3582 ± 0.0067 |

The proposed Tree-Guided MoE substantially improves over the evaluated pure deep learning baselines and approaches the PR-AUC performance of the strongest tree-based baseline.

The final model also achieves higher observed F1, MCC, and high-precision recall than LightGBM on the internal test set.

---

## Ablation Study

The model development is evaluated progressively through the following sequence:

$$
\mathrm{CNN}
\rightarrow
\mathrm{DeepFM}
\rightarrow
\mathrm{CNN+DeepFM}
\rightarrow
\mathrm{MoE}
\rightarrow
\mathrm{MoE+Focal}
$$

$$
\rightarrow
\mathrm{Tree\text{-}Guided\ MoE+KD+Focal}
\rightarrow
\mathrm{Tree\text{-}Guided\ MoE+KD+Focal+SupCon}
$$

This ablation design is used to analyze the contribution of each major architectural and optimization component.

---

## Project Structure

```text
TreeGuidedFraudDetection/
│
├── preprocessing/
│   └── data preprocessing pipeline
│
├── models/
│   ├── cnn_branch.py
│   ├── deepfm_branch.py
│   ├── gated_moe.py
│   └── tree_guided_moe.py
│
├── losses/
│   └── fraud_losses.py
│
├── training/
│   ├── ml_baselines
│   ├── dl_baselines
│   ├── ablation
│   └── proposed_model
│
├── evaluation/
│   ├── standard_metrics
│   ├── cost_aware_metrics
│   └── stability_analysis
│
├── results/
│   ├── ml
│   ├── dl
│   ├── ablation
│   ├── tuning
│   └── final
│
├── checkpoints/
│
└── archive/
```

---

## Final Proposed Model

The final architecture is:

**Tree-Guided CNN-DeepFM Mixture-of-Experts with Knowledge Distillation, Focal Loss, and Supervised Contrastive Learning**

The framework integrates:

$$
\text{Tree Knowledge}
+
\text{CNN Representation Learning}
+
\text{DeepFM Feature Interaction}
$$

$$
+
\text{Adaptive Expert Routing}
+
\text{Class-Imbalance Learning}
+
\text{Contrastive Representation Learning}
$$

---

## Research Scope

The project focuses on:

- Transaction fraud detection
- Highly imbalanced tabular classification
- Hybrid tree-deep learning
- Mixture-of-Experts
- Knowledge Distillation
- Focal Loss
- Supervised Contrastive Learning
- Cost-aware fraud detection
- Multi-seed stability evaluation

The main empirical finding can be summarized as:

$$
\text{Pure Deep Learning}
<
\text{Tree-Guided Deep Learning}
\approx
\text{Strong Tree-Based Machine Learning}
$$

---

## License

This project is intended for academic and research purposes.
