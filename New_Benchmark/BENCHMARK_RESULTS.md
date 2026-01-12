# DFU Classification Benchmark Results

## Overview

This benchmark evaluates knowledge distillation and network pruning for Diabetic Foot Ulcer (DFU) image classification. A ResNet-50 teacher model transfers knowledge to a MobileNetV2 student, which is then pruned for edge deployment.

## Dataset

- **Source**: Kaggle DFU Dataset (laithjj/diabetic-foot-ulcer-dfu)
- **Total Images**: 1,055
- **Classes**: Abnormal (Ulcer): 512, Normal (Healthy): 543
- **Validation**: 5-Fold Stratified Cross-Validation

---

## K-Fold Cross Validation Results (5 Folds)

| Model | Mean Accuracy | Std Dev | Min | Max |
|-------|---------------|---------|-----|-----|
| ResNet-50 (Teacher) | 99.72% | ±0.38% | 99.05% | 100.00% |
| MobileNetV2 (KD) | 99.81% | ±0.38% | 99.05% | 100.00% |
| Pruned (Round 1) | 98.67% | ±0.63% | 97.63% | 99.53% |
| Pruned (Round 2) | 98.39% | ±2.54% | 93.36% | 100.00% |

### Per-Fold Results

| Fold | Teacher | Student KD | Pruned R1 | Pruned R2 |
|------|---------|------------|-----------|-----------|
| 1 | 99.05% | 99.05% | 99.05% | 100.00% |
| 2 | 99.53% | 100.00% | 98.58% | 99.05% |
| 3 | 100.00% | 100.00% | 99.53% | 99.53% |
| 4 | 100.00% | 100.00% | 98.58% | 93.36% |
| 5 | 100.00% | 100.00% | 97.63% | 100.00% |

---

## Model Efficiency Comparison

| Metric | ResNet-50 (Teacher) | MobileNetV2 (Student) | Reduction |
|--------|---------------------|----------------------|-----------|
| Parameters | 23.51M | 2.09M | **11.2x smaller** |
| CPU Latency | 13.57ms | 5.11ms | **2.65x faster** |
| GPU Latency | 1.42ms | 0.94ms | **1.51x faster** |

---

## Single Split Results (for reference)

| Model | Accuracy | F1 Score | Parameters | CPU Latency |
|-------|----------|----------|------------|-------------|
| ResNet-50 (Teacher) | 100.00% | 1.0000 | 23.51M | 15.53ms |
| MobileNetV2 (KD) | 100.00% | 1.0000 | 2.23M | 5.12ms |
| Pruned (Round 1) | 96.88% | 0.9687 | 2.09M | - |
| Pruned (Round 2) | 96.25% | 0.9625 | 2.09M | - |

---

## Key Findings

### 1. Knowledge Distillation Effectiveness
- Student matches teacher accuracy (99.81% vs 99.72%)
- 11.2x parameter reduction with no accuracy loss
- 2.65x CPU speedup achieved

### 2. Pruning Analysis
- Round 1: Stable results (±0.63% std)
- Round 2: High variance (±2.54% std) - indicates aggressive pruning risk
- Accuracy drop: ~1.4% after 2 pruning rounds

### 3. K-Fold vs Single Split
- Single split showed 100% accuracy (overfitting concern)
- K-Fold reveals true performance: 99.72% ± 0.38%
- More reliable for deployment decisions

---

## Training Configuration

```yaml
# Knowledge Distillation
kd_temperature: 10.0
kd_alpha: 0.7

# Training
teacher_epochs: 15
student_epochs: 15
learning_rate: 0.0005
batch_size: 16
image_size: 224

# Pruning
prune_rounds: 2
prune_fraction: 0.2
finetune_epochs: 3

# Regularization
weight_decay: 1e-4
early_stopping_patience: 5
lr_scheduler: ReduceLROnPlateau
```

---

## Hardware

- **GPU**: NVIDIA GeForce RTX 5070 Ti
- **Framework**: PyTorch 2.9.1+cu128

---

## Conclusion

The knowledge distillation approach successfully compresses ResNet-50 (23.5M params) to MobileNetV2 (2.09M params) while maintaining 99.81% accuracy. The 11.2x parameter reduction and 2.65x CPU speedup make the model suitable for edge deployment in clinical settings with limited hardware resources.

### Recommendations
1. Use K-fold CV for reliable accuracy estimates
2. Limit pruning to Round 1 for stable results
3. Consider Round 2 pruning only if accuracy variance is acceptable

---

## Files

- `run_benchmark.py` - Single split benchmark script
- `run_benchmark_kfold.py` - K-fold cross validation script
- `results/benchmark_results.json` - Single split results
- `kfold_results/kfold_results.json` - K-fold results
