#!/usr/bin/env python3
"""
DFU Classification Benchmark - Easy to Use Version
===================================================
Automatically downloads dataset, splits data, trains teacher/student, and runs benchmarks.

Usage:
    python run_benchmark.py --data-dir /path/to/data

Or with Kaggle auto-download:
    python run_benchmark.py --kaggle-download

Author: Lakshya
"""

import argparse
import os
import sys
import shutil
import random
import copy
import time
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Optional imports
try:
    from ptflops import get_model_complexity_info
    HAS_PTFLOPS = True
except ImportError:
    HAS_PTFLOPS = False
    print("Warning: ptflops not installed. GFLOPs calculation will be skipped.")

try:
    import kagglehub
    HAS_KAGGLE = True
except ImportError:
    HAS_KAGGLE = False


def print_banner(text):
    """Print a nice banner"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DATASET HANDLING
# ============================================================

def download_kaggle_dataset():
    """Download DFU dataset from Kaggle"""
    if not HAS_KAGGLE:
        print("Error: kagglehub not installed. Install with: pip install kagglehub")
        sys.exit(1)

    print("Downloading DFU dataset from Kaggle...")
    path = kagglehub.dataset_download("laithjj/diabetic-foot-ulcer-dfu")
    print(f"Dataset downloaded to: {path}")
    return path


def auto_detect_dataset_structure(data_dir):
    """
    Auto-detect dataset structure and return info about it.
    Handles various folder structures.
    """
    data_dir = Path(data_dir)

    # Check if already has train/val/test structure
    if all((data_dir / split).exists() for split in ['train', 'val', 'test']):
        return {'type': 'split', 'path': str(data_dir)}

    # Check for DFU/Patches structure (Kaggle download)
    patches_path = data_dir / 'DFU' / 'Patches'
    if patches_path.exists():
        return {'type': 'kaggle_patches', 'path': str(patches_path)}

    # Check for direct Abnormal/Normal structure
    abnormal_names = ['Abnormal(Ulcer)', 'Abnormal', 'abnormal', 'ulcer', 'Ulcer', 'positive', 'infected']
    normal_names = ['Normal(Healthy skin)', 'Normal', 'normal', 'healthy', 'Healthy', 'negative']

    for abn in abnormal_names:
        for norm in normal_names:
            if (data_dir / abn).exists() and (data_dir / norm).exists():
                return {
                    'type': 'two_class',
                    'path': str(data_dir),
                    'abnormal': abn,
                    'normal': norm
                }

    # Check subdirectories
    for subdir in data_dir.iterdir():
        if subdir.is_dir():
            result = auto_detect_dataset_structure(subdir)
            if result:
                return result

    return None


def split_dataset(src_dir, dst_dir, abnormal_folder, normal_folder,
                  train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Split dataset into train/val/test"""
    print(f"\nSplitting dataset...")
    print(f"  Source: {src_dir}")
    print(f"  Destination: {dst_dir}")
    print(f"  Ratios - Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio}")

    random.seed(seed)
    dst_dir = Path(dst_dir)
    src_dir = Path(src_dir)

    # Create directories
    for split in ['train', 'val', 'test']:
        for cls in ['Abnormal', 'Normal']:
            (dst_dir / split / cls).mkdir(parents=True, exist_ok=True)

    # Process each class
    class_mapping = [
        (abnormal_folder, 'Abnormal'),
        (normal_folder, 'Normal')
    ]

    total_images = 0
    for src_cls, dst_cls in class_mapping:
        src_path = src_dir / src_cls
        if not src_path.exists():
            print(f"  Warning: {src_path} not found")
            continue

        files = [f for f in os.listdir(src_path)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
        random.shuffle(files)

        n = len(files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits = {
            'train': files[:n_train],
            'val': files[n_train:n_train+n_val],
            'test': files[n_train+n_val:]
        }

        for split_name, split_files in splits.items():
            for f in split_files:
                shutil.copy(src_path / f, dst_dir / split_name / dst_cls / f)

        print(f"  {dst_cls}: {len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test")
        total_images += n

    print(f"  Total: {total_images} images")
    return str(dst_dir)


def prepare_dataset(args):
    """Prepare dataset - download if needed, split if needed"""

    # Download from Kaggle if requested
    if args.kaggle_download:
        kaggle_path = download_kaggle_dataset()
        args.data_dir = kaggle_path

    if not args.data_dir:
        print("Error: Please provide --data-dir or use --kaggle-download")
        sys.exit(1)

    # Auto-detect structure
    print(f"\nAnalyzing dataset at: {args.data_dir}")
    structure = auto_detect_dataset_structure(args.data_dir)

    if not structure:
        print("Error: Could not detect dataset structure.")
        print("Expected: folder with Abnormal/Normal subfolders or train/val/test splits")
        sys.exit(1)

    print(f"  Detected structure: {structure['type']}")

    # If already split, use directly
    if structure['type'] == 'split':
        return structure['path']

    # Otherwise, split the data
    output_dir = Path(args.output_dir) / 'data'

    if structure['type'] == 'kaggle_patches':
        return split_dataset(
            structure['path'], output_dir,
            'Abnormal(Ulcer)', 'Normal(Healthy skin)',
            args.train_ratio, args.val_ratio, args.test_ratio, args.seed
        )
    elif structure['type'] == 'two_class':
        return split_dataset(
            structure['path'], output_dir,
            structure['abnormal'], structure['normal'],
            args.train_ratio, args.val_ratio, args.test_ratio, args.seed
        )


# ============================================================
# DATA LOADERS
# ============================================================

def make_dataloaders(data_dir, image_size=224, batch_size=32, num_workers=4):
    """Create train/val/test dataloaders"""
    transform_train = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_eval = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform_train)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=transform_eval)
    test_ds = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=transform_eval)

    print(f"\nDataset loaded:")
    print(f"  Train: {len(train_ds)} images")
    print(f"  Val: {len(val_ds)} images")
    print(f"  Test: {len(test_ds)} images")
    print(f"  Classes: {train_ds.classes}")

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, len(train_ds.classes)


# ============================================================
# MODEL UTILITIES
# ============================================================

def count_parameters(model):
    """Count total and non-zero parameters"""
    total = 0
    nonzero = 0
    for p in model.parameters():
        total += p.numel()
        nonzero += (p.abs() > 1e-8).sum().item()
    return total, nonzero


def compute_gflops(model, input_size=(3, 224, 224)):
    """Compute GFLOPs using ptflops"""
    if not HAS_PTFLOPS:
        return None
    try:
        macs, _ = get_model_complexity_info(model, input_size, as_strings=False, print_per_layer_stat=False)
        return macs / 1e9
    except:
        return None


def measure_latency(model, device, input_size=(3, 224, 224), n_runs=50):
    """Measure inference latency"""
    model.eval()
    dummy = torch.randn((1,) + input_size).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy)

    # Measure
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.time()
            _ = model(dummy)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.time() - t0)

    return {
        'mean_ms': 1000 * np.mean(times),
        'std_ms': 1000 * np.std(times),
        'min_ms': 1000 * np.min(times),
        'max_ms': 1000 * np.max(times)
    }


# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    preds, trues = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        preds.extend(out.argmax(dim=1).cpu().tolist())
        trues.extend(y.cpu().tolist())

    return {
        'loss': running_loss / len(loader.dataset),
        'acc': accuracy_score(trues, preds),
        'f1': f1_score(trues, preds, average='macro')
    }


def evaluate(model, loader, device):
    """Evaluate model"""
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds.extend(out.argmax(dim=1).cpu().tolist())
            trues.extend(y.cpu().tolist())

    return {
        'accuracy': accuracy_score(trues, preds),
        'f1': f1_score(trues, preds, average='macro'),
        'precision': precision_score(trues, preds, average='macro'),
        'recall': recall_score(trues, preds, average='macro')
    }


# ============================================================
# KNOWLEDGE DISTILLATION
# ============================================================

class KDLoss(nn.Module):
    """Knowledge Distillation Loss"""
    def __init__(self, temperature=10.0, alpha=0.7):
        super().__init__()
        self.T = temperature
        self.alpha = alpha
        self.kl = nn.KLDivLoss(reduction='batchmean')
        self.ce = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, targets):
        soft_loss = self.kl(
            nn.functional.log_softmax(student_logits / self.T, dim=1),
            nn.functional.softmax(teacher_logits / self.T, dim=1)
        ) * (self.T * self.T)
        hard_loss = self.ce(student_logits, targets)
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss


def train_kd_epoch(student, teacher, loader, kd_loss, optimizer, device):
    """Train student with knowledge distillation"""
    student.train()
    teacher.eval()
    running_loss = 0.0
    preds, trues = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        s_logits = student(x)
        with torch.no_grad():
            t_logits = teacher(x)

        loss = kd_loss(s_logits, t_logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        preds.extend(s_logits.argmax(dim=1).cpu().tolist())
        trues.extend(y.cpu().tolist())

    return {
        'loss': running_loss / len(loader.dataset),
        'acc': accuracy_score(trues, preds),
        'f1': f1_score(trues, preds, average='macro')
    }


# ============================================================
# HISTORY-BASED FILTER PRUNING (HBFP)
# ============================================================

def get_conv_layers(model):
    """Get all Conv2d layers"""
    convs = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            convs.append((m, name))
    return convs


def record_filter_history(model, history):
    """Record L1 norm of each filter"""
    for module, name in get_conv_layers(model):
        w = module.weight.detach().cpu().numpy()
        l1_norms = np.sum(np.abs(w), axis=(1, 2, 3))
        if name not in history:
            history[name] = []
        history[name].append(l1_norms.copy())


def prune_filters(model, history, prune_fraction=0.2, device='cpu'):
    """Prune filters based on history"""
    pruned_count = 0

    for module, name in get_conv_layers(model):
        if name not in history or len(history[name]) == 0:
            continue

        # Get latest L1 norms
        l1_norms = history[name][-1]
        num_filters = len(l1_norms)
        num_to_prune = int(num_filters * prune_fraction)

        if num_to_prune == 0:
            continue

        # Find filters with lowest L1 norm
        indices_to_prune = np.argsort(l1_norms)[:num_to_prune]

        # Zero out the filters
        with torch.no_grad():
            for idx in indices_to_prune:
                if idx < module.weight.size(0):
                    module.weight.data[idx].zero_()
                    if module.bias is not None and idx < module.bias.size(0):
                        module.bias.data[idx] = 0.0

        pruned_count += len(indices_to_prune)

    return pruned_count


# ============================================================
# MAIN BENCHMARK PIPELINE
# ============================================================

def run_benchmark(args):
    """Run the complete benchmark pipeline"""

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Prepare dataset
    print_banner("PREPARING DATASET")
    data_dir = prepare_dataset(args)

    # Create dataloaders
    train_loader, val_loader, test_loader, n_classes = make_dataloaders(
        data_dir, args.image_size, args.batch_size, args.num_workers
    )

    results = {'config': vars(args)}

    # ========== TEACHER TRAINING ==========
    print_banner("TRAINING TEACHER (ResNet-50)")

    teacher = models.resnet50(weights='IMAGENET1K_V1')
    teacher.fc = nn.Linear(teacher.fc.in_features, n_classes)
    teacher = teacher.to(device)

    optimizer = optim.Adam(teacher.parameters(), lr=args.teacher_lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(args.teacher_epochs):
        metrics = train_epoch(teacher, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(teacher, val_loader, device)

        print(f"Epoch {epoch+1}/{args.teacher_epochs} - "
              f"Loss: {metrics['loss']:.4f}, Train Acc: {metrics['acc']:.4f}, "
              f"Val Acc: {val_metrics['accuracy']:.4f}")

        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save(teacher.state_dict(), os.path.join(args.output_dir, 'teacher_best.pth'))

    teacher.load_state_dict(torch.load(os.path.join(args.output_dir, 'teacher_best.pth')))

    # Benchmark teacher
    print("\nBenchmarking teacher...")
    teacher_test = evaluate(teacher, test_loader, device)
    teacher_params, teacher_nonzero = count_parameters(teacher)
    teacher_gflops = compute_gflops(teacher)
    teacher_latency_gpu = measure_latency(teacher, device) if device.type == 'cuda' else None
    teacher_latency_cpu = measure_latency(teacher.cpu(), torch.device('cpu'))
    teacher = teacher.to(device)

    results['teacher'] = {
        'metrics': teacher_test,
        'params': teacher_params,
        'gflops': teacher_gflops,
        'latency_gpu': teacher_latency_gpu,
        'latency_cpu': teacher_latency_cpu
    }

    print(f"\nTeacher Results:")
    print(f"  Accuracy: {teacher_test['accuracy']*100:.2f}%")
    print(f"  Parameters: {teacher_params/1e6:.2f}M")
    print(f"  GFLOPs: {teacher_gflops:.3f}" if teacher_gflops else "  GFLOPs: N/A")
    print(f"  CPU Latency: {teacher_latency_cpu['mean_ms']:.2f}ms")

    # ========== STUDENT TRAINING WITH KD ==========
    print_banner("TRAINING STUDENT (MobileNetV2) WITH KNOWLEDGE DISTILLATION")

    student = models.mobilenet_v2(weights='IMAGENET1K_V1')
    student.classifier[1] = nn.Linear(student.classifier[1].in_features, n_classes)
    student = student.to(device)

    kd_loss = KDLoss(temperature=args.kd_temp, alpha=args.kd_alpha)
    optimizer = optim.Adam(student.parameters(), lr=args.student_lr)

    filter_history = defaultdict(list)
    best_val_acc = 0.0

    for epoch in range(args.student_epochs):
        metrics = train_kd_epoch(student, teacher, train_loader, kd_loss, optimizer, device)
        record_filter_history(student, filter_history)
        val_metrics = evaluate(student, val_loader, device)

        print(f"Epoch {epoch+1}/{args.student_epochs} - "
              f"Loss: {metrics['loss']:.4f}, Train Acc: {metrics['acc']:.4f}, "
              f"Val Acc: {val_metrics['accuracy']:.4f}")

        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save(student.state_dict(), os.path.join(args.output_dir, 'student_kd_best.pth'))

    student.load_state_dict(torch.load(os.path.join(args.output_dir, 'student_kd_best.pth')))

    # Benchmark student (pre-pruning)
    print("\nBenchmarking student (KD only)...")
    student_test = evaluate(student, test_loader, device)
    student_params, student_nonzero = count_parameters(student)
    student_gflops = compute_gflops(student)
    student_latency_gpu = measure_latency(student, device) if device.type == 'cuda' else None
    student_latency_cpu = measure_latency(student.cpu(), torch.device('cpu'))
    student = student.to(device)

    results['student_kd'] = {
        'metrics': student_test,
        'params': student_params,
        'gflops': student_gflops,
        'latency_gpu': student_latency_gpu,
        'latency_cpu': student_latency_cpu
    }

    print(f"\nStudent (KD) Results:")
    print(f"  Accuracy: {student_test['accuracy']*100:.2f}%")
    print(f"  Parameters: {student_params/1e6:.2f}M")
    print(f"  GFLOPs: {student_gflops:.3f}" if student_gflops else "  GFLOPs: N/A")
    print(f"  CPU Latency: {student_latency_cpu['mean_ms']:.2f}ms")

    # ========== PRUNING ROUNDS ==========
    print_banner("PRUNING STUDENT MODEL")

    results['pruning_rounds'] = []

    for round_idx in range(args.prune_rounds):
        print(f"\n--- Pruning Round {round_idx + 1}/{args.prune_rounds} ---")

        # Prune
        pruned = prune_filters(student, filter_history, args.prune_fraction, device)
        print(f"Pruned {pruned} filters")

        # Fine-tune
        optimizer = optim.Adam(student.parameters(), lr=args.finetune_lr)
        for epoch in range(args.finetune_epochs):
            metrics = train_epoch(student, train_loader, criterion, optimizer, device)
            record_filter_history(student, filter_history)

        # Evaluate
        test_metrics = evaluate(student, test_loader, device)
        total_params, nonzero_params = count_parameters(student)
        gflops = compute_gflops(student)

        round_result = {
            'round': round_idx + 1,
            'pruned_filters': pruned,
            'metrics': test_metrics,
            'total_params': total_params,
            'nonzero_params': nonzero_params,
            'gflops': gflops
        }
        results['pruning_rounds'].append(round_result)

        print(f"  Accuracy: {test_metrics['accuracy']*100:.2f}%")
        print(f"  Non-zero params: {nonzero_params/1e6:.2f}M")

    # Save final model
    torch.save(student.state_dict(), os.path.join(args.output_dir, 'student_pruned_final.pth'))

    # ========== FINAL SUMMARY ==========
    print_banner("BENCHMARK RESULTS SUMMARY")

    print(f"{'Model':<25} {'Accuracy':<12} {'Params':<12} {'GFLOPs':<10} {'CPU (ms)':<10}")
    print("-" * 70)

    # Teacher row
    t_gflops = f"{results['teacher']['gflops']:.3f}" if results['teacher']['gflops'] else "N/A"
    t_cpu = f"{results['teacher']['latency_cpu']['mean_ms']:.2f}"
    print(f"{'ResNet-50 (Teacher)':<25} "
          f"{results['teacher']['metrics']['accuracy']*100:.2f}%{'':<5} "
          f"{results['teacher']['params']/1e6:.2f}M{'':<6} "
          f"{t_gflops:<10} "
          f"{t_cpu}")

    # Student KD row
    s_gflops = f"{results['student_kd']['gflops']:.3f}" if results['student_kd']['gflops'] else "N/A"
    s_cpu = f"{results['student_kd']['latency_cpu']['mean_ms']:.2f}"
    print(f"{'MobileNetV2 (KD)':<25} "
          f"{results['student_kd']['metrics']['accuracy']*100:.2f}%{'':<5} "
          f"{results['student_kd']['params']/1e6:.2f}M{'':<6} "
          f"{s_gflops:<10} "
          f"{s_cpu}")

    # Pruning rounds
    for r in results['pruning_rounds']:
        r_gflops = f"{r['gflops']:.3f}" if r['gflops'] else "N/A"
        print(f"{'Pruned (Round '+str(r['round'])+')':<25} "
              f"{r['metrics']['accuracy']*100:.2f}%{'':<5} "
              f"{r['nonzero_params']/1e6:.2f}M{'':<6} "
              f"{r_gflops:<10} "
              f"N/A")

    # Speed gains
    if results['teacher']['latency_cpu'] and results['student_kd']['latency_cpu']:
        cpu_speedup = results['teacher']['latency_cpu']['mean_ms'] / results['student_kd']['latency_cpu']['mean_ms']
        print(f"\nCPU Speed Gain: {cpu_speedup:.2f}x faster")

    if results['teacher']['latency_gpu'] and results['student_kd']['latency_gpu']:
        gpu_speedup = results['teacher']['latency_gpu']['mean_ms'] / results['student_kd']['latency_gpu']['mean_ms']
        print(f"GPU Speed Gain: {gpu_speedup:.2f}x faster")

    # Save results
    results_file = os.path.join(args.output_dir, 'benchmark_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {results_file}")
    print(f"Models saved to: {args.output_dir}")

    return results


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="DFU Classification Benchmark - Easy to Use",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use with local data directory
  python run_benchmark.py --data-dir /path/to/dfu/data

  # Auto-download from Kaggle
  python run_benchmark.py --kaggle-download

  # Quick test run
  python run_benchmark.py --kaggle-download --teacher-epochs 2 --student-epochs 2 --prune-rounds 1
        """
    )

    # Data options
    data_group = parser.add_argument_group('Data Options')
    data_group.add_argument('--data-dir', type=str, help='Path to dataset directory')
    data_group.add_argument('--kaggle-download', action='store_true', help='Download dataset from Kaggle')
    data_group.add_argument('--train-ratio', type=float, default=0.7, help='Train split ratio')
    data_group.add_argument('--val-ratio', type=float, default=0.15, help='Validation split ratio')
    data_group.add_argument('--test-ratio', type=float, default=0.15, help='Test split ratio')

    # Training options
    train_group = parser.add_argument_group('Training Options')
    train_group.add_argument('--teacher-epochs', type=int, default=10, help='Teacher training epochs')
    train_group.add_argument('--student-epochs', type=int, default=10, help='Student KD training epochs')
    train_group.add_argument('--teacher-lr', type=float, default=0.001, help='Teacher learning rate')
    train_group.add_argument('--student-lr', type=float, default=0.001, help='Student learning rate')
    train_group.add_argument('--batch-size', type=int, default=32, help='Batch size')
    train_group.add_argument('--image-size', type=int, default=224, help='Image size')

    # KD options
    kd_group = parser.add_argument_group('Knowledge Distillation Options')
    kd_group.add_argument('--kd-temp', type=float, default=10.0, help='KD temperature')
    kd_group.add_argument('--kd-alpha', type=float, default=0.7, help='KD alpha (soft loss weight)')

    # Pruning options
    prune_group = parser.add_argument_group('Pruning Options')
    prune_group.add_argument('--prune-rounds', type=int, default=2, help='Number of pruning rounds')
    prune_group.add_argument('--prune-fraction', type=float, default=0.2, help='Fraction of filters to prune per round')
    prune_group.add_argument('--finetune-epochs', type=int, default=3, help='Fine-tune epochs after pruning')
    prune_group.add_argument('--finetune-lr', type=float, default=0.0001, help='Fine-tune learning rate')

    # Other options
    other_group = parser.add_argument_group('Other Options')
    other_group.add_argument('--output-dir', type=str, default='./benchmark_output', help='Output directory')
    other_group.add_argument('--num-workers', type=int, default=4, help='DataLoader workers')
    other_group.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Validate
    if not args.data_dir and not args.kaggle_download:
        parser.error("Please provide --data-dir or use --kaggle-download")

    run_benchmark(args)


if __name__ == '__main__':
    main()
