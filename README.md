# Reducing Processing Cost in Diabetic Foot Ulcer Image Classification

Using Knowledge Distillation and Network Pruning

## Authors
- Lakshya (lakshya.2502010001@muj.manipal.edu)
- Jaydeep Kishore (jaydeep.kishore@jaipur.manipal.edu)

## Abstract

Deep learning models have become central to automated diabetic foot ulcer (DFU) classification, yet the computational burden of deploying such architectures remains a persistent problem in clinical and mobile health environments. This paper explores a paired reduction strategy built around knowledge distillation and history-based filter pruning. A ResNet-50 teacher teaches a MobileNetV2 student via softened targets, then the student is pruned using a training-history-informed filter selection method.

## Key Results

| Model | Accuracy | Parameters | GFLOPs | CPU Latency |
|-------|----------|------------|--------|-------------|
| ResNet-50 (Teacher) | 99.07% | 23.5M | 4.13 | 18.26 ms |
| MobileNetV2 (KD Student) | 100.00% | 2.23M | 0.32 | 5.26 ms |
| Pruned Student (Round 2) | 98.13% | 1.82M | 0.32 | - |

**Speed Gains:** 3.47x faster on CPU, 1.71x faster on GPU

## Repository Structure

```
├── Benchmark_Code/          # Training & benchmarking scripts
│   ├── dfu_kd_prune_benchmark_notebook.py
│   ├── banch.py
│   └── proning_knowlidge.py
├── Benchmarks/              # Results & metrics
│   ├── benchmarks_summary.json
│   ├── complete_results.json
│   ├── prune_rounds.csv
│   └── test_predictions.csv
├── Dataset/                 # DFU image dataset
│   ├── data/               # Train/Val/Test splits
│   └── TestSet/            # Test images with labels
├── Data_Split_Code/         # Dataset preparation scripts
│   ├── split_dataset.py
│   ├── data_splits.json
│   └── DFU_data_import_from_kagal.ipynb
└── Reference_Papers/        # Related research papers (PDFs)
    ├── DFU-Main-PaperNew.QUT/
    ├── DFUNet_.../
    ├── 2102.00160v2/
    ├── 2204.11618v2/
    └── s10791-025-09637-8/
```

## Methodology

1. **Teacher Training**: ResNet-50 pretrained on ImageNet, fine-tuned on DFU dataset
2. **Knowledge Distillation**: Hybrid loss with temperature τ=10, α=0.7
3. **History-Based Filter Pruning (HBFP)**: Track filter L1 norms across epochs, prune redundant filters
4. **Fine-tuning**: Retrain after each pruning round

## Dataset

- **Classes**: Healthy skin, Infected ulcer, Non-infected ulcer
- **Split**: 5,000 train / 1,200 val / 800 test images
- **Image size**: 224×224, normalized

## Requirements

```
torch
torchvision
ptflops
numpy
pandas
scikit-learn
```

## Usage

```bash
# Train teacher model
python Benchmark_Code/proning_knowlidge.py --mode teacher

# Train student with KD + pruning
python Benchmark_Code/proning_knowlidge.py --mode student_prune

# Run full benchmark
python Benchmark_Code/dfu_kd_prune_benchmark_notebook.py
```

## References

1. Basha et al., "Deep model compression based on the training history," 2022
2. Kishore et al., "Enhancing medical diagnosis on chest X-rays: knowledge distillation," Discover Computing, 2025
3. Hinton et al., "Distilling the knowledge in a neural network," arXiv:1503.02531, 2015
4. Howard et al., "MobileNetV2: Inverted residuals and linear bottlenecks," CVPR, 2018

## License

For academic and research use only.

## Citation

```bibtex
@article{lakshya2025dfu,
  title={Reducing Processing Cost in Diabetic Foot Ulcer Image Classification Using Knowledge Distillation and Network Pruning},
  author={Lakshya and Kishore, Jaydeep},
  year={2025}
}
```
