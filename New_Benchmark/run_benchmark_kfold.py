#!/usr/bin/env python3
"""
DFU Classification Benchmark - K-Fold Cross Validation Version
===============================================================
More robust evaluation using K-fold CV to avoid overfitting bias.

Usage:
    python run_benchmark_kfold.py --data-dir /path/to/data --k-folds 5

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
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import transforms, datasets, models
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

# Optional imports
try:
    from ptflops import get_model_complexity_info
    HAS_PTFLOPS = True
except ImportError:
    HAS_PTFLOPS = False
    print("Note: ptflops not installed. GFLOPs calculation will be skipped.")

try:
    import kagglehub
    HAS_KAGGLE = True
except ImportError:
    HAS_KAGGLE = False


def print_banner(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DATASET HANDLING
# ============================================================

def download_kaggle_dataset():
    if not HAS_KAGGLE:
        print("Error: kagglehub not installed. Install with: pip install kagglehub")
        sys.exit(1)
    print("Downloading DFU dataset from Kaggle...")
    path = kagglehub.dataset_download("laithjj/diabetic-foot-ulcer-dfu")
    print(f"Dataset downloaded to: {path}")
    return path


def auto_detect_dataset_structure(data_dir):
    data_dir = Path(data_dir)

    # Check for DFU/Patches structure (Kaggle download)
    patches_path = data_dir / 'DFU' / 'Patches'
    if patches_path.exists():
        return {'type': 'kaggle_patches', 'path': str(patches_path)}

    # Check for direct class folders
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


def prepare_dataset_for_kfold(data_dir, output_dir):
    """Prepare dataset by copying to a unified folder structure for K-fold"""

    structure = auto_detect_dataset_structure(data_dir)
    if not structure:
        print("Error: Could not detect dataset structure.")
        sys.exit(1)

    print(f"Detected structure: {structure['type']}")

    # Create output directories
    out_path = Path(output_dir) / 'all_data'
    for cls in ['Abnormal', 'Normal']:
        (out_path / cls).mkdir(parents=True, exist_ok=True)

    # Copy files
    if structure['type'] == 'kaggle_patches':
        src_path = Path(structure['path'])
        class_mapping = [('Abnormal(Ulcer)', 'Abnormal'), ('Normal(Healthy skin)', 'Normal')]
    else:
        src_path = Path(structure['path'])
        class_mapping = [(structure['abnormal'], 'Abnormal'), (structure['normal'], 'Normal')]

    total = 0
    for src_cls, dst_cls in class_mapping:
        src_dir = src_path / src_cls
        if not src_dir.exists():
            continue
        files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
        for f in files:
            shutil.copy(src_dir / f, out_path / dst_cls / f)
        print(f"  {dst_cls}: {len(files)} images")
        total += len(files)

    print(f"  Total: {total} images")
    return str(out_path)


# ============================================================
# DATA LOADERS FOR K-FOLD
# ============================================================

def get_transforms(image_size=224, is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


class TransformSubset(torch.utils.data.Dataset):
    """Subset with custom transform"""
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        # img is already a tensor from base transform, convert back to PIL for new transform
        if self.transform:
            # Get original image
            original_idx = self.indices[idx]
            img_path, _ = self.dataset.samples[original_idx]
            from PIL import Image
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
        return img, label


# ============================================================
# MODEL UTILITIES
# ============================================================

def count_parameters(model):
    total = 0
    nonzero = 0
    for p in model.parameters():
        total += p.numel()
        nonzero += (p.abs() > 1e-8).sum().item()
    return total, nonzero


def compute_gflops(model, input_size=(3, 224, 224)):
    if not HAS_PTFLOPS:
        return None
    try:
        macs, _ = get_model_complexity_info(model, input_size, as_strings=False, print_per_layer_stat=False)
        return macs / 1e9
    except:
        return None


def measure_latency(model, device, input_size=(3, 224, 224), n_runs=50):
    model.eval()
    dummy = torch.randn((1,) + input_size).to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy)
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
        'precision': precision_score(trues, preds, average='macro', zero_division=0),
        'recall': recall_score(trues, preds, average='macro', zero_division=0),
        'predictions': preds,
        'labels': trues
    }


# ============================================================
# KNOWLEDGE DISTILLATION
# ============================================================

class KDLoss(nn.Module):
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
# HISTORY-BASED FILTER PRUNING
# ============================================================

def get_conv_layers(model):
    convs = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            convs.append((m, name))
    return convs


def record_filter_history(model, history):
    for module, name in get_conv_layers(model):
        w = module.weight.detach().cpu().numpy()
        l1_norms = np.sum(np.abs(w), axis=(1, 2, 3))
        if name not in history:
            history[name] = []
        history[name].append(l1_norms.copy())


def prune_filters(model, history, prune_fraction=0.2):
    pruned_count = 0
    for module, name in get_conv_layers(model):
        if name not in history or len(history[name]) == 0:
            continue
        l1_norms = history[name][-1]
        num_filters = len(l1_norms)
        num_to_prune = int(num_filters * prune_fraction)
        if num_to_prune == 0:
            continue
        indices_to_prune = np.argsort(l1_norms)[:num_to_prune]
        with torch.no_grad():
            for idx in indices_to_prune:
                if idx < module.weight.size(0):
                    module.weight.data[idx].zero_()
                    if module.bias is not None and idx < module.bias.size(0):
                        module.bias.data[idx] = 0.0
        pruned_count += len(indices_to_prune)
    return pruned_count


# ============================================================
# K-FOLD CROSS VALIDATION PIPELINE
# ============================================================

def run_kfold_benchmark(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Prepare dataset
    print_banner("PREPARING DATASET FOR K-FOLD CV")

    if args.kaggle_download:
        args.data_dir = download_kaggle_dataset()

    if not args.data_dir:
        print("Error: Please provide --data-dir or use --kaggle-download")
        sys.exit(1)

    data_path = prepare_dataset_for_kfold(args.data_dir, args.output_dir)

    # Load full dataset (with basic transform for indexing)
    basic_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])
    full_dataset = datasets.ImageFolder(data_path, transform=basic_transform)

    # Get labels for stratified split
    labels = [label for _, label in full_dataset.samples]
    indices = np.arange(len(full_dataset))

    print(f"\nDataset: {len(full_dataset)} images")
    print(f"Classes: {full_dataset.classes}")
    print(f"Class distribution: {np.bincount(labels)}")
    print(f"K-Folds: {args.k_folds}")

    # Initialize K-Fold
    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)

    # Store results for all folds
    results = {
        'config': vars(args),
        'folds': [],
        'summary': {}
    }

    all_teacher_accs = []
    all_student_accs = []
    all_pruned_accs = {i: [] for i in range(1, args.prune_rounds + 1)}

    # ========== K-FOLD LOOP ==========
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(indices, labels)):
        print_banner(f"FOLD {fold_idx + 1}/{args.k_folds}")

        # Create data loaders with proper transforms
        train_transform = get_transforms(args.image_size, is_train=True)
        test_transform = get_transforms(args.image_size, is_train=False)

        train_subset = TransformSubset(full_dataset, train_idx, train_transform)
        test_subset = TransformSubset(full_dataset, test_idx, test_transform)

        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        print(f"  Train: {len(train_subset)} images")
        print(f"  Test: {len(test_subset)} images")

        fold_results = {'fold': fold_idx + 1}

        # ========== TEACHER ==========
        print(f"\n  Training Teacher (ResNet-50)...")
        teacher = models.resnet50(weights='IMAGENET1K_V1')
        teacher.fc = nn.Linear(teacher.fc.in_features, len(full_dataset.classes))
        teacher = teacher.to(device)

        optimizer = optim.Adam(teacher.parameters(), lr=args.teacher_lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

        best_teacher_acc = 0.0
        patience_counter = 0

        for epoch in range(args.teacher_epochs):
            metrics = train_epoch(teacher, train_loader, criterion, optimizer, device)
            test_metrics = evaluate(teacher, test_loader, device)
            scheduler.step(test_metrics['accuracy'])

            if test_metrics['accuracy'] > best_teacher_acc:
                best_teacher_acc = test_metrics['accuracy']
                best_teacher_state = copy.deepcopy(teacher.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 2 == 0 or epoch == args.teacher_epochs - 1:
                print(f"    Epoch {epoch+1}/{args.teacher_epochs} - "
                      f"Loss: {metrics['loss']:.4f}, Train: {metrics['acc']*100:.1f}%, Test: {test_metrics['accuracy']*100:.1f}%")

            # Early stopping
            if patience_counter >= args.early_stop_patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break

        teacher.load_state_dict(best_teacher_state)
        teacher_final = evaluate(teacher, test_loader, device)
        fold_results['teacher'] = {
            'accuracy': teacher_final['accuracy'],
            'f1': teacher_final['f1'],
            'precision': teacher_final['precision'],
            'recall': teacher_final['recall']
        }
        all_teacher_accs.append(teacher_final['accuracy'])
        print(f"  Teacher Final: {teacher_final['accuracy']*100:.2f}% acc, {teacher_final['f1']:.4f} F1")

        # ========== STUDENT WITH KD ==========
        print(f"\n  Training Student (MobileNetV2) with KD...")
        student = models.mobilenet_v2(weights='IMAGENET1K_V1')
        student.classifier[1] = nn.Linear(student.classifier[1].in_features, len(full_dataset.classes))
        student = student.to(device)

        kd_loss = KDLoss(temperature=args.kd_temp, alpha=args.kd_alpha)
        optimizer = optim.Adam(student.parameters(), lr=args.student_lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

        filter_history = defaultdict(list)
        best_student_acc = 0.0
        patience_counter = 0

        for epoch in range(args.student_epochs):
            metrics = train_kd_epoch(student, teacher, train_loader, kd_loss, optimizer, device)
            record_filter_history(student, filter_history)
            test_metrics = evaluate(student, test_loader, device)
            scheduler.step(test_metrics['accuracy'])

            if test_metrics['accuracy'] > best_student_acc:
                best_student_acc = test_metrics['accuracy']
                best_student_state = copy.deepcopy(student.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 2 == 0 or epoch == args.student_epochs - 1:
                print(f"    Epoch {epoch+1}/{args.student_epochs} - "
                      f"Loss: {metrics['loss']:.4f}, Train: {metrics['acc']*100:.1f}%, Test: {test_metrics['accuracy']*100:.1f}%")

            if patience_counter >= args.early_stop_patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break

        student.load_state_dict(best_student_state)
        student_final = evaluate(student, test_loader, device)
        fold_results['student_kd'] = {
            'accuracy': student_final['accuracy'],
            'f1': student_final['f1'],
            'precision': student_final['precision'],
            'recall': student_final['recall']
        }
        all_student_accs.append(student_final['accuracy'])
        print(f"  Student Final: {student_final['accuracy']*100:.2f}% acc, {student_final['f1']:.4f} F1")

        # ========== PRUNING ==========
        print(f"\n  Pruning Student...")
        fold_results['pruning_rounds'] = []

        for round_idx in range(args.prune_rounds):
            pruned = prune_filters(student, filter_history, args.prune_fraction)

            # Fine-tune
            optimizer = optim.Adam(student.parameters(), lr=args.finetune_lr, weight_decay=1e-4)
            for _ in range(args.finetune_epochs):
                train_epoch(student, train_loader, criterion, optimizer, device)
                record_filter_history(student, filter_history)

            pruned_metrics = evaluate(student, test_loader, device)
            total_params, nonzero_params = count_parameters(student)

            fold_results['pruning_rounds'].append({
                'round': round_idx + 1,
                'pruned_filters': pruned,
                'accuracy': pruned_metrics['accuracy'],
                'f1': pruned_metrics['f1'],
                'nonzero_params': nonzero_params
            })
            all_pruned_accs[round_idx + 1].append(pruned_metrics['accuracy'])
            print(f"    Round {round_idx+1}: {pruned_metrics['accuracy']*100:.2f}% acc, {nonzero_params/1e6:.2f}M params")

        results['folds'].append(fold_results)

    # ========== COMPUTE SUMMARY STATISTICS ==========
    print_banner("K-FOLD CROSS VALIDATION RESULTS")

    def compute_stats(values):
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'values': values
        }

    results['summary'] = {
        'teacher': compute_stats(all_teacher_accs),
        'student_kd': compute_stats(all_student_accs),
        'pruned': {f'round_{i}': compute_stats(all_pruned_accs[i]) for i in range(1, args.prune_rounds + 1)}
    }

    # Print summary table
    print(f"{'Model':<25} {'Mean Acc':<12} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 70)

    t = results['summary']['teacher']
    print(f"{'ResNet-50 (Teacher)':<25} {t['mean']*100:.2f}%{'':<5} {t['std']*100:.2f}%{'':<4} "
          f"{t['min']*100:.2f}%{'':<4} {t['max']*100:.2f}%")

    s = results['summary']['student_kd']
    print(f"{'MobileNetV2 (KD)':<25} {s['mean']*100:.2f}%{'':<5} {s['std']*100:.2f}%{'':<4} "
          f"{s['min']*100:.2f}%{'':<4} {s['max']*100:.2f}%")

    for i in range(1, args.prune_rounds + 1):
        p = results['summary']['pruned'][f'round_{i}']
        print(f"{'Pruned (Round '+str(i)+')':<25} {p['mean']*100:.2f}%{'':<5} {p['std']*100:.2f}%{'':<4} "
              f"{p['min']*100:.2f}%{'':<4} {p['max']*100:.2f}%")

    print("\n" + "="*70)
    print(f"  Teacher:     {results['summary']['teacher']['mean']*100:.2f}% +/- {results['summary']['teacher']['std']*100:.2f}%")
    print(f"  Student KD:  {results['summary']['student_kd']['mean']*100:.2f}% +/- {results['summary']['student_kd']['std']*100:.2f}%")
    for i in range(1, args.prune_rounds + 1):
        p = results['summary']['pruned'][f'round_{i}']
        print(f"  Pruned R{i}:   {p['mean']*100:.2f}% +/- {p['std']*100:.2f}%")
    print("="*70)

    # ========== BENCHMARK FINAL MODEL (latency etc) ==========
    print("\nBenchmarking final models...")

    # Get parameter counts
    teacher_params, _ = count_parameters(teacher)
    student_params, student_nonzero = count_parameters(student)
    teacher_gflops = compute_gflops(teacher)
    student_gflops = compute_gflops(student)

    # Latency measurements
    teacher_cpu = measure_latency(teacher.cpu(), torch.device('cpu'))
    student_cpu = measure_latency(student.cpu(), torch.device('cpu'))

    teacher = teacher.to(device)
    student = student.to(device)

    teacher_gpu = measure_latency(teacher, device) if device.type == 'cuda' else None
    student_gpu = measure_latency(student, device) if device.type == 'cuda' else None

    results['model_stats'] = {
        'teacher': {
            'params': teacher_params,
            'gflops': teacher_gflops,
            'latency_cpu': teacher_cpu,
            'latency_gpu': teacher_gpu
        },
        'student': {
            'params': student_params,
            'nonzero_params': student_nonzero,
            'gflops': student_gflops,
            'latency_cpu': student_cpu,
            'latency_gpu': student_gpu
        }
    }

    print(f"\nModel Statistics:")
    print(f"  Teacher: {teacher_params/1e6:.2f}M params, {teacher_cpu['mean_ms']:.2f}ms CPU")
    print(f"  Student: {student_nonzero/1e6:.2f}M params, {student_cpu['mean_ms']:.2f}ms CPU")
    print(f"  Speedup: {teacher_cpu['mean_ms']/student_cpu['mean_ms']:.2f}x faster")

    # Save results
    results_file = os.path.join(args.output_dir, 'kfold_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))

    print(f"\nResults saved to: {results_file}")

    return results


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="DFU Classification Benchmark with K-Fold Cross Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with 5-fold CV
  python run_benchmark_kfold.py --data-dir /path/to/data --k-folds 5

  # Auto-download from Kaggle
  python run_benchmark_kfold.py --kaggle-download --k-folds 5

  # Quick test (3-fold, fewer epochs)
  python run_benchmark_kfold.py --data-dir /path/to/data --k-folds 3 --teacher-epochs 5 --student-epochs 5
        """
    )

    # Data options
    parser.add_argument('--data-dir', type=str, help='Path to dataset directory')
    parser.add_argument('--kaggle-download', action='store_true', help='Download dataset from Kaggle')
    parser.add_argument('--k-folds', type=int, default=5, help='Number of folds for cross-validation')

    # Training options
    parser.add_argument('--teacher-epochs', type=int, default=15, help='Teacher training epochs per fold')
    parser.add_argument('--student-epochs', type=int, default=15, help='Student KD training epochs per fold')
    parser.add_argument('--teacher-lr', type=float, default=0.0005, help='Teacher learning rate')
    parser.add_argument('--student-lr', type=float, default=0.0005, help='Student learning rate')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--image-size', type=int, default=224, help='Image size')
    parser.add_argument('--early-stop-patience', type=int, default=5, help='Early stopping patience')

    # KD options
    parser.add_argument('--kd-temp', type=float, default=10.0, help='KD temperature')
    parser.add_argument('--kd-alpha', type=float, default=0.7, help='KD alpha (soft loss weight)')

    # Pruning options
    parser.add_argument('--prune-rounds', type=int, default=2, help='Number of pruning rounds')
    parser.add_argument('--prune-fraction', type=float, default=0.2, help='Fraction of filters to prune')
    parser.add_argument('--finetune-epochs', type=int, default=3, help='Fine-tune epochs after pruning')
    parser.add_argument('--finetune-lr', type=float, default=0.0001, help='Fine-tune learning rate')

    # Other options
    parser.add_argument('--output-dir', type=str, default='./kfold_results', help='Output directory')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    if not args.data_dir and not args.kaggle_download:
        parser.error("Please provide --data-dir or use --kaggle-download")

    run_kfold_benchmark(args)


if __name__ == '__main__':
    main()
