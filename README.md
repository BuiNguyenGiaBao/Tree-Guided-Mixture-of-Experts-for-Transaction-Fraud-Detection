# Tree-Guided Mixture-of-Experts for Transaction Fraud Detection

## Overview

This project presents a hybrid Tree-Guided Mixture-of-Experts framework for transaction fraud detection on highly imbalanced tabular data.

The proposed approach combines the strengths of tree-based machine learning and deep neural networks.

LightGBM is used as a teacher model, while two neural experts learn complementary feature representations:

- CNN Expert
- DeepFM Expert

A tree-guided gating mechanism adaptively determines the contribution of each expert.

The final proposed model combines:

- CNN representation learning
- DeepFM feature interaction learning
- Mixture-of-Experts
- LightGBM-guided routing
- Knowledge Distillation
- Focal Loss
- Supervised Contrastive Learning
- Cost-aware fraud evaluation

---

## Dataset

The experiments are conducted on the IEEE-CIS Fraud Detection dataset.

The dataset contains transaction and identity information associated with fraudulent and legitimate transactions.

The processed dataset is divided chronologically into three subsets:

- Training set
- Validation set
- Internal test set

The preprocessing pipeline includes:

- Missing-value processing
- Numerical feature scaling
- Categorical feature encoding
- Frequency-based feature construction
- Group-level statistical features
- Feature alignment
- Memory optimization
- Train-fitted preprocessing objects
- Cost-aware transaction metadata

The preprocessing parameters are estimated from the training data and subsequently applied to the validation and test sets.

---

## System Architecture

The proposed framework consists of four main components:

1. CNN Expert
2. DeepFM Expert
3. Tree-Guided Mixture-of-Experts
4. Fraud Classification Head

The overall processing flow is:

```text
Transaction Features
        |
        +-----------------------+
        |                       |
        v                       v
   CNN Expert              DeepFM Expert
        |                       |
        v                       v
 CNN Representation       DeepFM Representation
        |                       |
        +-----------+-----------+
                    |
                    v
             Tree-Guided Gate
                    ^
                    |
             LightGBM Teacher
                    |
                    v
             Expert Weighting
                    |
                    v
            Fused Representation
                    |
                    v
             Fraud Classifier
                    |
                    v
             Fraud Probability
```

---

## CNN Expert

The CNN branch is designed to capture nonlinear and local interaction patterns among transaction features.

Its main components include:

- Feature projection
- Pseudo-sequence construction
- 1D convolutional layers
- Attention pooling
- Low-rank bilinear interaction
- Latent feature representation

The CNN branch produces a feature vector called:

```text
z_CNN
```

This representation summarizes interaction patterns learned by the convolutional expert.

---

## DeepFM Expert

The DeepFM branch is designed to model multiple levels of feature interaction.

It combines:

- First-order linear effects
- Second-order Factorization Machine interactions
- Higher-order nonlinear interactions
- Categorical embeddings
- Multilayer perceptron

The DeepFM branch produces:

```text
z_DeepFM
```

The DeepFM expert complements the CNN branch by explicitly modeling both shallow and deep feature interactions.

---

## LightGBM Teacher

LightGBM is used as the tree-based teacher model.

For each transaction, the teacher produces a fraud probability:

```text
p_teacher = P(fraud | transaction)
```

The probability is converted into a teacher logit:

```text
teacher_logit = log(
    p_teacher / (1 - p_teacher)
)
```

The teacher signal is used in two ways:

1. It guides the Mixture-of-Experts gating network.
2. It provides soft supervision through Knowledge Distillation.

---

## Tree-Guided Mixture-of-Experts

The gating network receives three main inputs:

```text
z_CNN
z_DeepFM
teacher_logit
```

The input to the gate can be represented as:

```text
gate_input =
    concatenate(
        z_CNN,
        z_DeepFM,
        teacher_logit
    )
```

The gating network generates two adaptive weights:

```text
g_CNN
g_DeepFM
```

where:

```text
g_CNN + g_DeepFM = 1
```

The final representation is computed as:

```text
z_fused =
    g_CNN * z_CNN
    +
    g_DeepFM * z_DeepFM
```

Therefore, the model does not assign fixed importance to CNN and DeepFM.

Instead, the contribution of each expert can change according to the characteristics of each transaction and the information provided by the tree-based teacher.

---

## Proposed Models

### Model 1

Model 1 is:

**Tree-Guided MoE + Knowledge Distillation + Focal Loss**

Its training objective is:

```text
Loss_Model1 =
    Focal_Loss
    +
    lambda_KD * KD_Loss
```

Model 1 is used as an intermediate model in the ablation study.

It is designed to evaluate the contribution of:

- Tree-guided expert routing
- Knowledge Distillation
- Focal Loss

---

### Model 2

Model 2 is the final proposed model:

**Tree-Guided MoE + Knowledge Distillation + Focal Loss + Supervised Contrastive Learning**

Its training objective is:

```text
Loss_Model2 =
    Focal_Loss
    +
    lambda_KD * KD_Loss
    +
    lambda_SC * SupCon_Loss
```

Model 2 extends Model 1 by introducing Supervised Contrastive Learning.

The additional contrastive objective is used to improve the structure of the learned representation space.

---

## Focal Loss

Fraud detection is a highly imbalanced classification problem.

The majority of transactions are legitimate, while fraudulent transactions represent only a small proportion of the dataset.

Focal Loss reduces the contribution of easy majority-class samples and assigns greater importance to difficult examples.

Conceptually:

```text
Focal Loss =
    classification loss
    *
    difficulty weighting
```

This encourages the model to focus more strongly on difficult fraud samples.

---

## Knowledge Distillation

Knowledge Distillation transfers predictive information from the LightGBM teacher to the neural student.

The student is trained using both:

- Ground-truth labels
- Teacher predictions

The general training relationship is:

```text
LightGBM Teacher
        |
        v
Teacher Probability / Teacher Logit
        |
        v
Knowledge Distillation Loss
        |
        v
Tree-Guided Neural Student
```

The goal is not necessarily for the student to outperform the teacher on every metric.

Instead, the teacher provides structural information that helps the neural model learn tabular fraud patterns more effectively.

---

## Supervised Contrastive Learning

Supervised Contrastive Learning operates on the learned fused representation.

Its main objective is to:

- Pull representations of samples from the same class closer together
- Push representations of different classes farther apart
- Improve separation between fraud and non-fraud transactions

Conceptually:

```text
Fraud Samples
     |
     +----> closer representations

Non-Fraud Samples
     |
     +----> closer representations

Fraud vs Non-Fraud
     |
     +----> greater separation
```

The contrastive objective is applied to the fused representation produced by the Mixture-of-Experts module.

---

## Machine Learning Baselines

The following traditional machine learning models are evaluated:

- Logistic Regression
- HistGradientBoosting
- XGBoost
- CatBoost
- LightGBM

Among these models, LightGBM is also used as the teacher model for the proposed architecture.

---

## Deep Learning Baselines

The following deep learning models and architectural variants are evaluated:

- DeepFM
- CNN
- CNN + DeepFM
- Mixture-of-Experts with BCE
- MoE + Focal Loss
- MoE + Focal Loss + Supervised Contrastive Learning
- FT-Transformer
- TabM-like
- TabR-like
- SAINT-like

TabM-like, TabR-like, and SAINT-like are experimental reproduced implementations used for comparative evaluation.

---

## Ablation Study

The architecture is evaluated progressively.

```text
CNN
 |
 v
DeepFM
 |
 v
CNN + DeepFM
 |
 v
Mixture-of-Experts
 |
 v
MoE + Focal Loss
 |
 v
Tree-Guided MoE
+ Knowledge Distillation
+ Focal Loss
 |
 v
Tree-Guided MoE
+ Knowledge Distillation
+ Focal Loss
+ Supervised Contrastive Learning
```

The final stage corresponds to Model 2.

The ablation study is designed to examine the contribution of each major component of the proposed framework.

---

## Evaluation Metrics

The project evaluates the models using both standard classification metrics and cost-aware operational metrics.

### Standard Metrics

The following metrics are reported:

- PR-AUC
- ROC-AUC
- Precision
- Recall
- F1-score
- Matthews Correlation Coefficient
- Recall at Precision >= 0.80

PR-AUC is treated as the primary ranking metric because of the severe class imbalance in fraud detection.

---

## Cost-Aware Evaluation

In real fraud detection systems, only a limited proportion of transactions can be manually reviewed.

Therefore, the project evaluates model performance under several review budgets:

- 1%
- 3%
- 5%
- 10%
- 15%
- 20%

For each budget, the following metrics are calculated:

- Precision@K
- Recall@K
- F1@K
- Lift@K
- Captured Fraud Amount Rate
- Expected Utility

The cost-aware evaluation measures how effectively each model prioritizes high-risk transactions under limited investigation capacity.

---

## Main Results

| Model | PR-AUC | F1 | MCC | Recall@P>=0.80 |
|---|---:|---:|---:|---:|
| LightGBM | 0.5431 | 0.5212 | 0.5042 | 0.3480 |
| Model 1 | 0.5366 | 0.5260 | 0.5101 | 0.3500 |
| Model 2 | 0.5354 +/- 0.0024 | 0.5313 +/- 0.0048 | 0.5185 +/- 0.0052 | 0.3582 +/- 0.0067 |

Model 1 represents the intermediate Tree-Guided MoE architecture with Knowledge Distillation and Focal Loss.

Model 2 is the final proposed model and includes Supervised Contrastive Learning.

The proposed models achieve PR-AUC performance close to LightGBM while substantially improving over the evaluated pure deep learning baselines.

Model 2 also achieves higher observed F1, MCC, and Recall at Precision >= 0.80 than LightGBM on the internal test set.

---

## Final Model Stability

The final proposed model is evaluated using three random seeds.

The main results are reported as mean and standard deviation.

```text
PR-AUC:
0.5354 +/- 0.0024

F1:
0.5313 +/- 0.0048

MCC:
0.5185 +/- 0.0052

Recall at Precision >= 0.80:
0.3582 +/- 0.0067
```

The relatively small standard deviations indicate stable performance across the evaluated random seeds.

---

## Performance Summary

The main empirical relationship observed in the experiments is:

```text
Pure Deep Learning
        <
Tree-Guided Deep Learning
        ~
Strong Tree-Based Machine Learning
```

The proposed framework significantly reduces the performance gap between conventional deep learning models and strong gradient-boosted tree models for imbalanced tabular fraud detection.

At the same time, the final model provides competitive or improved performance on several threshold-dependent metrics.

---

## Final Proposed Architecture

The final system can be summarized as:

```text
Transaction Data
        |
        +-------------------------------+
        |                               |
        v                               v
   CNN Expert                     DeepFM Expert
        |                               |
        v                               v
     z_CNN                         z_DeepFM
        |                               |
        +---------------+---------------+
                        |
                        v
              Tree-Guided MoE Gate
                        ^
                        |
                 LightGBM Teacher
                        |
                        v
                Adaptive Routing
                        |
                        v
                  z_fused
                        |
            +-----------+-----------+
            |           |           |
            v           v           v
        Focal Loss    KD Loss    SupCon Loss
            |           |           |
            +-----------+-----------+
                        |
                        v
                Final Optimization
                        |
                        v
                Fraud Prediction
```

---

## Project Structure

```text
TreeGuidedFraudDetection/
|
|-- preprocessing/
|   |-- data preprocessing pipeline
|
|-- models/
|   |-- cnn_branch.py
|   |-- deepfm_branch.py
|   |-- gated_moe.py
|   |-- tree_guided_moe.py
|
|-- losses/
|   |-- fraud_losses.py
|
|-- training/
|   |-- ml_baselines
|   |-- dl_baselines
|   |-- ablation
|   |-- proposed_model
|
|-- evaluation/
|   |-- standard_metrics
|   |-- cost_aware_metrics
|   |-- stability_analysis
|
|-- results/
|   |-- ml
|   |-- dl
|   |-- ablation
|   |-- tuning
|   |-- final
|
|-- checkpoints/
|
|-- archive/
|
|-- requirements.txt
|
`-- README.md
```

---

## Final Proposed Model

The final model is:

**Tree-Guided CNN-DeepFM Mixture-of-Experts with Knowledge Distillation, Focal Loss, and Supervised Contrastive Learning**

Its major components can be summarized as:

```text
Tree-Based Knowledge
        +
CNN Representation Learning
        +
DeepFM Feature Interaction
        +
Adaptive Expert Routing
        +
Knowledge Distillation
        +
Class-Imbalance Learning
        +
Contrastive Representation Learning
```

---

## Research Scope

This project focuses on:

- Transaction fraud detection
- Highly imbalanced tabular classification
- Hybrid tree-deep learning
- Mixture-of-Experts
- CNN-based tabular representation learning
- DeepFM feature interactions
- Knowledge Distillation
- Focal Loss
- Supervised Contrastive Learning
- Cost-aware fraud detection
- Multi-seed stability evaluation

---

## Keywords

`Fraud Detection`  
`IEEE-CIS`  
`Mixture-of-Experts`  
`LightGBM`  
`DeepFM`  
`CNN`  
`Knowledge Distillation`  
`Focal Loss`  
`Supervised Contrastive Learning`  
`Imbalanced Learning`  
`Tabular Deep Learning`  
`Cost-Aware Learning`

---

## License

This project is intended for academic and research purposes.
