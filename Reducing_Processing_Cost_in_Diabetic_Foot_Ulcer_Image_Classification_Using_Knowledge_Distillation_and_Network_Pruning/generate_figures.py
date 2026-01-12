"""
Generate figures for DFU Classification paper
Style inspired by HBFP paper (Basha et al.)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os
from PIL import Image

# Create figures directory if it doesn't exist
os.makedirs('figures', exist_ok=True)

# Set style for academic papers - clean style like HBFP paper
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['axes.spines.top'] = True
plt.rcParams['axes.spines.right'] = True
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.edgecolor'] = 'black'
plt.rcParams['legend.fancybox'] = False

# Color palette inspired by HBFP paper
COLORS = {
    'before': '#D32F2F',    # Red for before/baseline
    'after': '#388E3C',     # Green for after/compressed
    'teacher': '#D32F2F',   # Red
    'student': '#388E3C',   # Green
    'pruned_r1': '#1976D2', # Blue
    'pruned_r2': '#7B1FA2', # Purple
    'gray': '#616161',      # Gray for other methods
}


def create_dfu_samples():
    """Create DFU sample images figure from actual dataset"""
    normal_path = "../Dataset/data/Normal(Healthy skin)/"
    abnormal_path = "../Dataset/data/Abnormal(Ulcer)/"

    normal_images = ['1.jpg', '2.jpg', '3.jpg']
    abnormal_images = ['1.jpg', '2.jpg', '3.jpg']

    fig, axes = plt.subplots(2, 3, figsize=(10, 7))

    fig.text(0.02, 0.75, 'Normal\n(Healthy)', fontsize=12, fontweight='bold',
             va='center', ha='center', rotation=90)
    fig.text(0.02, 0.3, 'Abnormal\n(Ulcer)', fontsize=12, fontweight='bold',
             va='center', ha='center', rotation=90)

    for i, img_name in enumerate(normal_images):
        try:
            img = Image.open(os.path.join(normal_path, img_name))
            axes[0, i].imshow(img)
            axes[0, i].axis('off')
            axes[0, i].set_title(f'Sample {i+1}', fontsize=10)
        except:
            axes[0, i].text(0.5, 0.5, 'Image\nNot Found', ha='center', va='center')
            axes[0, i].axis('off')

    for i, img_name in enumerate(abnormal_images):
        try:
            img = Image.open(os.path.join(abnormal_path, img_name))
            axes[1, i].imshow(img)
            axes[1, i].axis('off')
            axes[1, i].set_title(f'Sample {i+1}', fontsize=10)
        except:
            axes[1, i].text(0.5, 0.5, 'Image\nNot Found', ha='center', va='center')
            axes[1, i].axis('off')

    plt.suptitle('DFU Dataset Samples', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0.05, 0, 1, 0.95])
    plt.savefig('figures/dfu_samples.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('figures/dfu_samples.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Created: figures/dfu_samples.pdf")


def create_methodology_workflow():
    """Create enhanced methodology workflow diagram - professional academic style"""
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor('white')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Professional color palette
    colors = {
        'data': ('#E3F2FD', '#1565C0'),      # (fill, border)
        'teacher': ('#FFF8E1', '#F57F17'),
        'kd': ('#E8F5E9', '#2E7D32'),
        'student': ('#FCE4EC', '#C2185B'),
        'pruning': ('#F3E5F5', '#7B1FA2'),
        'result': ('#E0F7FA', '#00838F')
    }

    # Title
    ax.text(7, 11.5, 'Proposed Methodology: Knowledge Distillation + Pruning Pipeline',
            ha='center', fontsize=14, fontweight='bold', color='#212121')

    # ============ Stage 1: Data Preparation ============
    stage_y = 9.5
    ax.text(0.5, stage_y + 0.8, 'Stage 1', fontsize=10, fontweight='bold', color='#1565C0')

    # Dataset box
    ax.add_patch(FancyBboxPatch((0.5, stage_y - 0.5), 3, 1.3, boxstyle='round,pad=0.1',
                                 facecolor=colors['data'][0], edgecolor=colors['data'][1], linewidth=2))
    ax.text(2, stage_y + 0.3, 'DFU Dataset', ha='center', fontsize=11, fontweight='bold', color='#212121')
    ax.text(2, stage_y - 0.15, '1,055 images\n2 classes', ha='center', fontsize=10, color='#424242')

    ax.annotate('', xy=(4, stage_y + 0.15), xytext=(3.5, stage_y + 0.15),
                arrowprops=dict(arrowstyle='->', color='#424242', lw=2))

    # Preprocessing
    ax.add_patch(FancyBboxPatch((4, stage_y - 0.5), 3, 1.3, boxstyle='round,pad=0.1',
                                 facecolor=colors['data'][0], edgecolor=colors['data'][1], linewidth=2))
    ax.text(5.5, stage_y + 0.3, 'Preprocessing', ha='center', fontsize=11, fontweight='bold', color='#212121')
    ax.text(5.5, stage_y - 0.15, '224×224 resize\nImageNet norm', ha='center', fontsize=10, color='#424242')

    ax.annotate('', xy=(7.5, stage_y + 0.15), xytext=(7, stage_y + 0.15),
                arrowprops=dict(arrowstyle='->', color='#424242', lw=2))

    # K-Fold
    ax.add_patch(FancyBboxPatch((7.5, stage_y - 0.5), 3.5, 1.3, boxstyle='round,pad=0.1',
                                 facecolor=colors['data'][0], edgecolor=colors['data'][1], linewidth=2))
    ax.text(9.25, stage_y + 0.3, '5-Fold Cross-Validation', ha='center', fontsize=11, fontweight='bold', color='#212121')
    ax.text(9.25, stage_y - 0.15, 'Stratified splits\nRobust evaluation', ha='center', fontsize=10, color='#424242')

    # ============ Stage 2: Teacher Training ============
    stage_y = 7.2
    ax.text(0.5, stage_y + 0.8, 'Stage 2', fontsize=10, fontweight='bold', color='#F57F17')

    ax.add_patch(FancyBboxPatch((0.5, stage_y - 0.5), 6, 1.5, boxstyle='round,pad=0.1',
                                 facecolor=colors['teacher'][0], edgecolor=colors['teacher'][1], linewidth=2.5))
    ax.text(3.5, stage_y + 0.5, 'Teacher Model Training', ha='center', fontsize=12, fontweight='bold', color='#212121')
    ax.text(3.5, stage_y - 0.1, 'ResNet-50 (ImageNet pretrained) | 23.51M params | 15 epochs',
            ha='center', fontsize=9, color='#424242')

    # Teacher result box
    ax.add_patch(FancyBboxPatch((7, stage_y - 0.4), 4, 1.3, boxstyle='round,pad=0.1',
                                 facecolor=colors['result'][0], edgecolor=colors['result'][1], linewidth=2))
    ax.text(9, stage_y + 0.4, 'Teacher Performance', ha='center', fontsize=10, fontweight='bold', color=colors['result'][1])
    ax.text(9, stage_y - 0.05, 'Accuracy: 99.72% | Params: 23.51M', ha='center', fontsize=9, color='#424242')

    ax.annotate('', xy=(7, stage_y + 0.25), xytext=(6.5, stage_y + 0.25),
                arrowprops=dict(arrowstyle='->', color='#757575', lw=1.5, ls='--'))

    # ============ Stage 3: Knowledge Distillation ============
    stage_y = 4.8
    ax.text(0.5, stage_y + 1, 'Stage 3', fontsize=10, fontweight='bold', color='#2E7D32')

    ax.add_patch(FancyBboxPatch((0.5, stage_y - 0.6), 6, 1.8, boxstyle='round,pad=0.1',
                                 facecolor=colors['kd'][0], edgecolor=colors['kd'][1], linewidth=2.5))
    ax.text(3.5, stage_y + 0.7, 'Knowledge Distillation', ha='center', fontsize=12, fontweight='bold', color='#212121')
    ax.text(3.5, stage_y + 0.2, r'$\mathcal{L}_{KD} = \alpha \mathcal{L}_{soft} + (1-\alpha) \mathcal{L}_{hard}$',
            ha='center', fontsize=11, color='#424242')
    ax.text(3.5, stage_y - 0.25, r'Temperature $\tau$=10, $\alpha$=0.7', ha='center', fontsize=9, color='#616161')

    # Student result box
    ax.add_patch(FancyBboxPatch((7, stage_y - 0.5), 4, 1.5, boxstyle='round,pad=0.1',
                                 facecolor=colors['result'][0], edgecolor=colors['result'][1], linewidth=2))
    ax.text(9, stage_y + 0.5, 'Student Performance', ha='center', fontsize=10, fontweight='bold', color=colors['result'][1])
    ax.text(9, stage_y + 0.05, 'Accuracy: 99.81%', ha='center', fontsize=10, fontweight='bold', color='#2E7D32')
    ax.text(9, stage_y - 0.3, 'Params: 2.09M (11.2× reduction)', ha='center', fontsize=9, color='#424242')

    ax.annotate('', xy=(7, stage_y + 0.2), xytext=(6.5, stage_y + 0.2),
                arrowprops=dict(arrowstyle='->', color='#757575', lw=1.5, ls='--'))

    # ============ Stage 4: Pruning ============
    stage_y = 2.2
    ax.text(0.5, stage_y + 1, 'Stage 4', fontsize=10, fontweight='bold', color='#7B1FA2')

    ax.add_patch(FancyBboxPatch((0.5, stage_y - 0.6), 6, 1.8, boxstyle='round,pad=0.1',
                                 facecolor=colors['pruning'][0], edgecolor=colors['pruning'][1], linewidth=2.5))
    ax.text(3.5, stage_y + 0.7, 'History-Based Filter Pruning', ha='center', fontsize=12, fontweight='bold', color='#212121')
    ax.text(3.5, stage_y + 0.2, 'Track filter norms → Identify redundancy → Prune 20%',
            ha='center', fontsize=9, color='#424242')
    ax.text(3.5, stage_y - 0.2, 'Fine-tune for 3 epochs per round', ha='center', fontsize=9, color='#616161')

    # Pruning result boxes
    ax.add_patch(FancyBboxPatch((7, stage_y + 0.3), 2, 1, boxstyle='round,pad=0.1',
                                 facecolor='#E8F5E9', edgecolor='#43A047', linewidth=2))
    ax.text(8, stage_y + 0.95, 'Round 1', ha='center', fontsize=9, fontweight='bold', color='#43A047')
    ax.text(8, stage_y + 0.55, '98.67%', ha='center', fontsize=11, fontweight='bold', color='#2E7D32')

    ax.add_patch(FancyBboxPatch((9.5, stage_y + 0.3), 2, 1, boxstyle='round,pad=0.1',
                                 facecolor='#FFEBEE', edgecolor='#E53935', linewidth=2))
    ax.text(10.5, stage_y + 0.95, 'Round 2', ha='center', fontsize=9, fontweight='bold', color='#E53935')
    ax.text(10.5, stage_y + 0.55, '98.39%', ha='center', fontsize=11, fontweight='bold', color='#C62828')

    ax.text(8, stage_y - 0.1, '(Stable)', ha='center', fontsize=8, color='#43A047')
    ax.text(10.5, stage_y - 0.1, '(Unstable)', ha='center', fontsize=8, color='#E53935')

    ax.annotate('', xy=(7, stage_y + 0.5), xytext=(6.5, stage_y + 0.5),
                arrowprops=dict(arrowstyle='->', color='#757575', lw=1.5, ls='--'))

    # ============ Vertical flow arrows ============
    ax.annotate('', xy=(3.5, 8.7), xytext=(3.5, 9),
                arrowprops=dict(arrowstyle='->', color='#424242', lw=2))
    ax.annotate('', xy=(3.5, 6.4), xytext=(3.5, 7.2),
                arrowprops=dict(arrowstyle='->', color='#424242', lw=2))
    ax.annotate('', xy=(3.5, 4), xytext=(3.5, 4.8),
                arrowprops=dict(arrowstyle='->', color='#424242', lw=2))

    # Key metrics summary at bottom
    ax.add_patch(FancyBboxPatch((0.5, 0.2), 12.5, 1.2, boxstyle='round,pad=0.1',
                                 facecolor='#FAFAFA', edgecolor='#BDBDBD', linewidth=1.5))
    ax.text(6.75, 1.0, 'Key Results Summary', ha='center', fontsize=11, fontweight='bold', color='#424242')
    ax.text(2.5, 0.5, '11.2× compression', ha='center', fontsize=10, fontweight='bold', color='#1565C0')
    ax.text(6.75, 0.5, '2.65× CPU speedup', ha='center', fontsize=10, fontweight='bold', color='#2E7D32')
    ax.text(11, 0.5, '99.81% accuracy', ha='center', fontsize=10, fontweight='bold', color='#7B1FA2')

    plt.savefig('figures/methodology_workflow.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('figures/methodology_workflow.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Created: figures/methodology_workflow.pdf")


def create_kd_architecture():
    """Create Knowledge Distillation architecture diagram"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')

    colors = {
        'input': '#E3F2FD',
        'teacher': '#FFECB3',
        'student': '#E8F5E9',
        'loss': '#FCE4EC',
        'output': '#F3E5F5'
    }

    # Input image
    ax.add_patch(FancyBboxPatch((4, 5.5), 2, 1, boxstyle='round,pad=0.1',
                                 facecolor=colors['input'], edgecolor='black', linewidth=1.5))
    ax.text(5, 6, 'Input Image\n(224x224)', ha='center', va='center', fontsize=9, fontweight='bold')

    ax.annotate('', xy=(2.5, 5), xytext=(4.5, 5.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=(7.5, 5), xytext=(5.5, 5.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # Teacher network
    ax.add_patch(FancyBboxPatch((1, 3.5), 3, 1.5, boxstyle='round,pad=0.1',
                                 facecolor=colors['teacher'], edgecolor='black', linewidth=1.5))
    ax.text(2.5, 4.25, 'Teacher Network', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(2.5, 3.8, 'ResNet-50\n23.51M parameters', ha='center', va='center', fontsize=8)

    # Student network
    ax.add_patch(FancyBboxPatch((6, 3.5), 3, 1.5, boxstyle='round,pad=0.1',
                                 facecolor=colors['student'], edgecolor='black', linewidth=1.5))
    ax.text(7.5, 4.25, 'Student Network', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(7.5, 3.8, 'MobileNetV2\n2.23M parameters', ha='center', va='center', fontsize=8)

    # Soft labels arrow
    ax.annotate('', xy=(6, 4.25), xytext=(4, 4.25),
                arrowprops=dict(arrowstyle='->', color='#FF5722', lw=2))
    ax.text(5, 4.6, r'Soft Labels ($\tau$=10)', ha='center', va='center', fontsize=9,
            color='#FF5722', fontweight='bold')

    ax.annotate('', xy=(2.5, 3), xytext=(2.5, 3.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=(7.5, 3), xytext=(7.5, 3.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # Hard labels
    ax.add_patch(FancyBboxPatch((1, 1.8), 3, 1, boxstyle='round,pad=0.1',
                                 facecolor=colors['output'], edgecolor='black', linewidth=1.5))
    ax.text(2.5, 2.3, 'Hard Labels\n(Ground Truth)', ha='center', va='center', fontsize=9, fontweight='bold')

    # Loss computation
    ax.add_patch(FancyBboxPatch((5.5, 1.5), 4, 1.5, boxstyle='round,pad=0.1',
                                 facecolor=colors['loss'], edgecolor='black', linewidth=1.5))
    ax.text(7.5, 2.25, 'Loss Computation', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(7.5, 1.8, r'$L_{KD} = 0.7 \cdot L_{soft} + 0.3 \cdot L_{hard}$',
            ha='center', va='center', fontsize=9)

    ax.annotate('', xy=(5.5, 2.25), xytext=(4, 2.25),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    legend_elements = [
        mpatches.Patch(facecolor=colors['teacher'], edgecolor='black', label='Teacher (frozen after training)'),
        mpatches.Patch(facecolor=colors['student'], edgecolor='black', label='Student (trainable)'),
        mpatches.Patch(facecolor=colors['loss'], edgecolor='black', label='KD Loss')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig('figures/kd_architecture.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('figures/kd_architecture.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Created: figures/kd_architecture.pdf")


def create_pruning_workflow():
    """Create enhanced HBFP pruning workflow diagram with visual illustrations"""
    fig = plt.figure(figsize=(14, 12))
    fig.patch.set_facecolor('white')

    # Main axis for the workflow - expanded left margin for repeat box
    ax = fig.add_subplot(111)
    ax.set_xlim(-2, 14)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Color scheme
    step_colors = {
        1: '#E3F2FD',  # Light blue
        2: '#FFF3E0',  # Light orange
        3: '#E8F5E9',  # Light green
        4: '#FCE4EC',  # Light pink
        5: '#F3E5F5',  # Light purple
        6: '#E0F7FA',  # Light cyan
    }
    accent_colors = {
        1: '#1976D2',  # Blue
        2: '#E65100',  # Orange
        3: '#388E3C',  # Green
        4: '#C2185B',  # Pink
        5: '#7B1FA2',  # Purple
        6: '#00838F',  # Cyan
    }

    # Title
    ax.text(7, 11.5, 'History-Based Filter Pruning (HBFP) Workflow', ha='center',
            fontsize=16, fontweight='bold', color='#212121')

    # Step positions (left column for steps, right for illustrations)
    step_y = [9.5, 7.5, 5.5, 3.5, 1.5]
    step_height = 1.6
    step_width = 5.5
    illust_x = 8.5

    # ============ Step 1: Track Filter Norms ============
    y = step_y[0]
    # Step box
    ax.add_patch(FancyBboxPatch((0.5, y - 0.6), step_width, step_height,
                                 boxstyle='round,pad=0.1', facecolor=step_colors[1],
                                 edgecolor=accent_colors[1], linewidth=2.5))
    # Step number circle
    circle = plt.Circle((1.2, y + 0.2), 0.4, facecolor=accent_colors[1], edgecolor='white', linewidth=2)
    ax.add_patch(circle)
    ax.text(1.2, y + 0.2, '1', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    # Text - centered inside box (box center is at x = 0.5 + 5.5/2 = 3.25, shifted right for circle)
    text_center_x = 3.5
    ax.text(text_center_x, y + 0.5, 'Track Filter Norms', ha='center', va='center', fontsize=12, fontweight='bold', color='#212121')
    ax.text(text_center_x, y - 0.1, r'Record $\ell_1$-norm of each filter', ha='center', va='center', fontsize=10, color='#424242')
    ax.text(text_center_x, y - 0.4, 'across T training epochs', ha='center', va='center', fontsize=10, color='#424242')

    # Illustration: Mini line chart showing filter norms - aligned closer to step box
    mini_ax1_pos = [0.48, 0.77, 0.15, 0.08]
    mini_ax1 = fig.add_axes(mini_ax1_pos)
    epochs_mini = np.arange(1, 11)
    np.random.seed(42)
    mini_ax1.plot(epochs_mini, 0.8 + 0.05 * np.sin(epochs_mini * 0.5) + np.random.normal(0, 0.01, 10),
                  'o-', color=accent_colors[1], linewidth=1.5, markersize=3, label='Filter 1')
    mini_ax1.plot(epochs_mini, 0.6 + 0.03 * epochs_mini / 10 + np.random.normal(0, 0.01, 10),
                  's-', color='#43A047', linewidth=1.5, markersize=3, label='Filter 2')
    mini_ax1.plot(epochs_mini, 0.4 + 0.02 * np.log(epochs_mini + 1) + np.random.normal(0, 0.01, 10),
                  '^-', color='#E53935', linewidth=1.5, markersize=3, label='Filter 3')
    mini_ax1.set_xlabel('Epoch', fontsize=7)
    mini_ax1.set_ylabel('$\ell_1$-norm', fontsize=7)
    mini_ax1.tick_params(axis='both', labelsize=6)
    mini_ax1.set_facecolor('#F5F5F5')
    mini_ax1.grid(True, alpha=0.3)

    # ============ Step 2: Compute Differences ============
    y = step_y[1]
    ax.add_patch(FancyBboxPatch((0.5, y - 0.6), step_width, step_height,
                                 boxstyle='round,pad=0.1', facecolor=step_colors[2],
                                 edgecolor=accent_colors[2], linewidth=2.5))
    circle = plt.Circle((1.2, y + 0.2), 0.4, facecolor=accent_colors[2], edgecolor='white', linewidth=2)
    ax.add_patch(circle)
    ax.text(1.2, y + 0.2, '2', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    ax.text(text_center_x, y + 0.5, 'Compute Pairwise Differences', ha='center', va='center', fontsize=12, fontweight='bold', color='#212121')
    ax.text(text_center_x, y - 0.2, r'$D_{ij} = \sum_{t=1}^{T} |\ell_1(f_i^t) - \ell_1(f_j^t)|$', ha='center', va='center', fontsize=11, color='#424242')

    # Illustration: Difference matrix
    mini_ax2_pos = [0.62, 0.60, 0.15, 0.08]
    mini_ax2 = fig.add_axes(mini_ax2_pos)
    diff_matrix = np.array([[0, 0.2, 0.8, 0.5], [0.2, 0, 0.6, 0.3], [0.8, 0.6, 0, 0.9], [0.5, 0.3, 0.9, 0]])
    im = mini_ax2.imshow(diff_matrix, cmap='YlOrRd', aspect='auto')
    mini_ax2.set_xticks([0, 1, 2, 3])
    mini_ax2.set_yticks([0, 1, 2, 3])
    mini_ax2.set_xticklabels(['F1', 'F2', 'F3', 'F4'], fontsize=7)
    mini_ax2.set_yticklabels(['F1', 'F2', 'F3', 'F4'], fontsize=7)
    mini_ax2.set_title('Difference Matrix', fontsize=8, fontweight='bold')
    # Highlight low values
    mini_ax2.add_patch(plt.Rectangle((-0.5, 0.5), 2, 2, fill=False, edgecolor='green', linewidth=2))

    # ============ Step 3: Identify Similar Pairs ============
    y = step_y[2]
    ax.add_patch(FancyBboxPatch((0.5, y - 0.6), step_width, step_height,
                                 boxstyle='round,pad=0.1', facecolor=step_colors[3],
                                 edgecolor=accent_colors[3], linewidth=2.5))
    circle = plt.Circle((1.2, y + 0.2), 0.4, facecolor=accent_colors[3], edgecolor='white', linewidth=2)
    ax.add_patch(circle)
    ax.text(1.2, y + 0.2, '3', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    ax.text(text_center_x, y + 0.5, 'Identify Redundant Filters', ha='center', va='center', fontsize=12, fontweight='bold', color='#212121')
    ax.text(text_center_x, y - 0.1, 'Select top-M% pairs with', ha='center', va='center', fontsize=10, color='#424242')
    ax.text(text_center_x, y - 0.4, 'lowest cumulative difference', ha='center', va='center', fontsize=10, color='#424242')

    # Illustration: Filter pairs
    illust_y = 5.7
    for i in range(3):
        x_base = 8.5 + i * 1.8
        # Filter pair boxes
        ax.add_patch(FancyBboxPatch((x_base, illust_y - 0.3), 0.6, 0.8,
                                     boxstyle='round,pad=0.05', facecolor='#C8E6C9',
                                     edgecolor=accent_colors[3], linewidth=1.5))
        ax.add_patch(FancyBboxPatch((x_base + 0.8, illust_y - 0.3), 0.6, 0.8,
                                     boxstyle='round,pad=0.05', facecolor='#C8E6C9',
                                     edgecolor=accent_colors[3], linewidth=1.5))
        ax.text(x_base + 0.3, illust_y + 0.1, f'F{i*2+1}', ha='center', fontsize=8, fontweight='bold')
        ax.text(x_base + 1.1, illust_y + 0.1, f'F{i*2+2}', ha='center', fontsize=8, fontweight='bold')
        ax.text(x_base + 0.7, illust_y + 0.7, 'Similar', ha='center', fontsize=7, color=accent_colors[3])
        # Connection line
        ax.plot([x_base + 0.6, x_base + 0.8], [illust_y + 0.1, illust_y + 0.1],
                color=accent_colors[3], linewidth=2, linestyle='--')

    # ============ Step 4: Apply Regularizer ============
    y = step_y[3]
    ax.add_patch(FancyBboxPatch((0.5, y - 0.6), step_width, step_height,
                                 boxstyle='round,pad=0.1', facecolor=step_colors[4],
                                 edgecolor=accent_colors[4], linewidth=2.5))
    circle = plt.Circle((1.2, y + 0.2), 0.4, facecolor=accent_colors[4], edgecolor='white', linewidth=2)
    ax.add_patch(circle)
    ax.text(1.2, y + 0.2, '4', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    ax.text(text_center_x, y + 0.5, 'Prune Redundant Filters', ha='center', va='center', fontsize=12, fontweight='bold', color='#212121')
    ax.text(text_center_x, y - 0.1, 'Remove 20% of filters per round', ha='center', va='center', fontsize=10, color='#424242')
    ax.text(text_center_x, y - 0.4, '(weaker filter from each pair)', ha='center', va='center', fontsize=10, color='#424242')

    # Illustration: Before/After pruning
    # Before
    ax.text(9, 4.0, 'Before', ha='center', fontsize=9, fontweight='bold', color='#C62828')
    for i in range(5):
        ax.add_patch(plt.Rectangle((8.2 + i * 0.35, 3.3), 0.3, 0.5,
                                     facecolor='#EF9A9A', edgecolor='#C62828', linewidth=1))
    # Arrow
    ax.annotate('', xy=(11.2, 3.55), xytext=(10.5, 3.55),
                arrowprops=dict(arrowstyle='->', color='#424242', lw=2))
    # After
    ax.text(12.5, 4.0, 'After', ha='center', fontsize=9, fontweight='bold', color='#2E7D32')
    for i in range(3):
        ax.add_patch(plt.Rectangle((11.5 + i * 0.45, 3.3), 0.4, 0.5,
                                     facecolor='#A5D6A7', edgecolor='#2E7D32', linewidth=1))

    # ============ Step 5: Fine-tune ============
    y = step_y[4]
    ax.add_patch(FancyBboxPatch((0.5, y - 0.6), step_width, step_height,
                                 boxstyle='round,pad=0.1', facecolor=step_colors[6],
                                 edgecolor=accent_colors[6], linewidth=2.5))
    circle = plt.Circle((1.2, y + 0.2), 0.4, facecolor=accent_colors[6], edgecolor='white', linewidth=2)
    ax.add_patch(circle)
    ax.text(1.2, y + 0.2, '5', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    ax.text(text_center_x, y + 0.5, 'Fine-tune Model', ha='center', va='center', fontsize=12, fontweight='bold', color='#212121')
    ax.text(text_center_x, y - 0.1, 'Train pruned model for 3 epochs', ha='center', va='center', fontsize=10, color='#424242')
    ax.text(text_center_x, y - 0.4, 'to recover accuracy', ha='center', va='center', fontsize=10, color='#424242')

    # Illustration: Accuracy recovery curve
    mini_ax3_pos = [0.62, 0.095, 0.15, 0.08]
    mini_ax3 = fig.add_axes(mini_ax3_pos)
    epochs_ft = np.arange(0, 4)
    acc_recovery = [96.5, 97.8, 98.4, 98.67]
    mini_ax3.plot(epochs_ft, acc_recovery, 'o-', color=accent_colors[6], linewidth=2, markersize=5)
    mini_ax3.fill_between(epochs_ft, 96, acc_recovery, alpha=0.3, color=accent_colors[6])
    mini_ax3.set_xlabel('Epoch', fontsize=7)
    mini_ax3.set_ylabel('Accuracy', fontsize=7)
    mini_ax3.set_ylim(96, 99.5)
    mini_ax3.tick_params(axis='both', labelsize=6)
    mini_ax3.set_facecolor('#F5F5F5')
    mini_ax3.grid(True, alpha=0.3)
    mini_ax3.set_title('Accuracy Recovery', fontsize=8, fontweight='bold')

    # ============ Arrows between steps ============
    arrow_x = 3.5
    for i in range(len(step_y) - 1):
        ax.annotate('', xy=(arrow_x, step_y[i+1] + 0.7), xytext=(arrow_x, step_y[i] - 0.6),
                    arrowprops=dict(arrowstyle='->', color='#616161', lw=2,
                                   connectionstyle='arc3,rad=0'))

    # ============ Repeat arrow (feedback loop on left side) ============
    # Create a feedback arrow on the left side going from step 5 back to step 1
    # The arrow should be clearly visible and aligned with the step boxes

    # Position for the feedback loop - to the left of step boxes (which start at x=0.5)
    feedback_x = -0.5  # X position for the vertical part of the feedback arrow

    # Text box dimensions - aligned with step 3 (middle step)
    box_width = 1.0
    box_height = 1.4
    box_x = feedback_x - box_width/2  # Center box on the feedback line
    box_y = step_y[2] - box_height/2 + 0.2  # Aligned with step 3 (y=5.5)

    # Gap positions for the vertical line (where box is)
    gap_bottom = box_y - 0.1
    gap_top = box_y + box_height + 0.1

    # Draw the feedback arrow with gap for text box:
    # 1. Horizontal line from step 5 going left
    ax.annotate('', xy=(feedback_x, step_y[-1] + 0.2), xytext=(0.5, step_y[-1] + 0.2),
                arrowprops=dict(arrowstyle='-', color='#7B1FA2', lw=2.5))

    # 2. Vertical line going up - BOTTOM segment (from step 5 to box bottom)
    ax.plot([feedback_x, feedback_x], [step_y[-1] + 0.2, gap_bottom],
            color='#7B1FA2', lw=2.5, solid_capstyle='round')

    # 3. Vertical line going up - TOP segment (from box top to step 1)
    ax.plot([feedback_x, feedback_x], [gap_top, step_y[0] + 0.2],
            color='#7B1FA2', lw=2.5, solid_capstyle='round')

    # 4. Horizontal arrow from left back to step 1
    ax.annotate('', xy=(0.5, step_y[0] + 0.2), xytext=(feedback_x, step_y[0] + 0.2),
                arrowprops=dict(arrowstyle='-|>', color='#7B1FA2', lw=2.5, mutation_scale=15))

    # Add text box for repeat label - in the gap of the vertical line
    ax.add_patch(FancyBboxPatch((box_x, box_y), box_width, box_height,
                                 boxstyle='round,pad=0.1',
                                 facecolor='#F3E5F5', edgecolor='#7B1FA2',
                                 linewidth=2, alpha=0.95, zorder=10))

    # Centered text inside the box
    text_x = box_x + box_width/2
    text_y = box_y + box_height/2
    ax.text(text_x, text_y, 'Repeat\nfor\nmultiple\nrounds',
            ha='center', va='center',
            fontsize=8, color='#7B1FA2', fontweight='bold',
            linespacing=1.1, zorder=11)

    plt.savefig('figures/pruning_workflow.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('figures/pruning_workflow.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Created: figures/pruning_workflow.pdf")


def create_efficiency_comparison():
    """Create model efficiency comparison - HBFP style bar chart with Before/After"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Data - like HBFP Figure 3 style
    metrics = ['Parameters\n(Millions)', 'CPU Latency\n(ms)', 'GPU Latency\n(ms)']
    teacher_vals = [23.51, 13.57, 1.42]
    student_vals = [2.09, 5.11, 0.94]

    x = np.arange(len(metrics))
    width = 0.35

    # Create bars - red for teacher (before), green for student (after)
    bars1 = ax.bar(x - width/2, teacher_vals, width, label='Teacher (ResNet-50)',
                   color=COLORS['before'], edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, student_vals, width, label='Student (MobileNetV2)',
                   color=COLORS['after'], edgecolor='black', linewidth=1.2)

    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Model Efficiency Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(loc='upper right', fontsize=10)

    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(bars1)
    autolabel(bars2)

    ax.set_ylim(0, max(teacher_vals) * 1.15)

    plt.tight_layout()
    plt.savefig('figures/efficiency_comparison.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('figures/efficiency_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Created: figures/efficiency_comparison.pdf")


def create_kfold_comparison():
    """Create enhanced K-fold cross-validation visualization with visual representation"""
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3, height_ratios=[1, 1.2])

    # Enhanced colors
    colors_kfold = {
        'train': '#2196F3',      # Blue for training
        'test': '#FF9800',       # Orange for test
        'teacher': '#C62828',    # Dark red
        'student': '#2E7D32',    # Dark green
        'pruned_r1': '#1565C0',  # Blue
        'pruned_r2': '#6A1B9A',  # Purple
    }

    # ============ Top Left: K-Fold Visual Representation ============
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(0, 12)
    ax1.set_ylim(0, 7)
    ax1.axis('off')
    ax1.set_title('(a) 5-Fold Cross-Validation Strategy', fontweight='bold', fontsize=12, pad=10)

    # Draw the 5 folds
    fold_height = 0.8
    fold_width = 2.0
    gap = 0.15

    for fold_idx in range(5):
        y_pos = 5.5 - fold_idx * 1.2

        # Fold label
        ax1.text(0.3, y_pos + 0.3, f'Fold {fold_idx + 1}', fontsize=10, fontweight='bold',
                va='center', ha='center')

        # Draw 5 data segments
        for seg_idx in range(5):
            x_pos = 1.2 + seg_idx * (fold_width + gap)

            if seg_idx == fold_idx:
                # Test set (highlighted)
                rect = FancyBboxPatch((x_pos, y_pos), fold_width, fold_height,
                                       boxstyle='round,pad=0.02',
                                       facecolor=colors_kfold['test'],
                                       edgecolor='#E65100', linewidth=2)
                ax1.add_patch(rect)
                ax1.text(x_pos + fold_width/2, y_pos + fold_height/2, 'TEST',
                        ha='center', va='center', fontsize=8, fontweight='bold', color='white')
            else:
                # Training set
                rect = FancyBboxPatch((x_pos, y_pos), fold_width, fold_height,
                                       boxstyle='round,pad=0.02',
                                       facecolor=colors_kfold['train'],
                                       edgecolor='#1565C0', linewidth=1.5, alpha=0.8)
                ax1.add_patch(rect)
                ax1.text(x_pos + fold_width/2, y_pos + fold_height/2, 'TRAIN',
                        ha='center', va='center', fontsize=7, color='white')

    # Legend
    train_patch = mpatches.Patch(facecolor=colors_kfold['train'], edgecolor='#1565C0',
                                  label='Training Data (80%)')
    test_patch = mpatches.Patch(facecolor=colors_kfold['test'], edgecolor='#E65100',
                                 label='Test Data (20%)')
    ax1.legend(handles=[train_patch, test_patch], loc='upper right', fontsize=9)

    # Add annotation
    ax1.text(6, 0.2, 'Each fold uses different 20% as test set', ha='center',
            fontsize=9, style='italic', color='#616161')

    # ============ Top Right: Summary Statistics ============
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 8)
    ax2.axis('off')
    ax2.set_title('(b) Cross-Validation Summary Statistics', fontweight='bold', fontsize=12, pad=10)

    # Background
    ax2.add_patch(FancyBboxPatch((0.2, 0.5), 9.6, 7, boxstyle='round,pad=0.1',
                                  facecolor='#F5F5F5', edgecolor='#BDBDBD', linewidth=1))

    # Statistics cards
    stats = [
        ('Teacher (ResNet-50)', '99.72%', '±0.38%', colors_kfold['teacher']),
        ('Student (MobileNetV2)', '99.81%', '±0.38%', colors_kfold['student']),
        ('Pruned Round 1', '98.67%', '±0.63%', colors_kfold['pruned_r1']),
        ('Pruned Round 2', '98.39%', '±2.54%', colors_kfold['pruned_r2']),
    ]

    for i, (name, acc, std, color) in enumerate(stats):
        y_pos = 6.3 - i * 1.5

        # Card background
        ax2.add_patch(FancyBboxPatch((0.5, y_pos - 0.5), 9, 1.3, boxstyle='round,pad=0.1',
                                      facecolor='white', edgecolor=color, linewidth=2))

        # Color bar
        ax2.add_patch(plt.Rectangle((0.5, y_pos - 0.5), 0.3, 1.3, facecolor=color))

        # Model name
        ax2.text(1.1, y_pos + 0.3, name, fontsize=10, fontweight='bold', va='center')

        # Accuracy
        ax2.text(6.5, y_pos + 0.1, acc, fontsize=16, fontweight='bold', va='center',
                ha='center', color=color)

        # Std deviation
        ax2.text(8.5, y_pos + 0.1, std, fontsize=11, va='center', ha='center',
                color='#757575' if '2.54' not in std else '#D32F2F',
                fontweight='bold' if '2.54' in std else 'normal')

    # Header labels
    ax2.text(6.5, 7.2, 'Mean Acc', fontsize=10, fontweight='bold', ha='center', color='#424242')
    ax2.text(8.5, 7.2, 'Std Dev', fontsize=10, fontweight='bold', ha='center', color='#424242')

    # Warning for high variance
    ax2.annotate('High variance!', xy=(8.5, 1.4), xytext=(7.5, 0.6),
                fontsize=9, color='#D32F2F', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#D32F2F', lw=1.5))

    # ============ Bottom: Per-Fold Results Bar Chart ============
    ax3 = fig.add_subplot(gs[1, :])
    ax3.set_facecolor('#FAFAFA')

    folds = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5']
    x = np.arange(len(folds))
    width = 0.18

    teacher = [99.05, 99.53, 100.00, 100.00, 100.00]
    student = [99.05, 100.00, 100.00, 100.00, 100.00]
    pruned_r1 = [99.05, 98.58, 99.53, 98.58, 97.63]
    pruned_r2 = [100.00, 99.05, 99.53, 93.36, 100.00]

    bars1 = ax3.bar(x - 1.5*width, teacher, width, label='Teacher (ResNet-50)',
                    color=colors_kfold['teacher'], edgecolor='#8B0000', linewidth=1.5)
    bars2 = ax3.bar(x - 0.5*width, student, width, label='Student (MobileNetV2)',
                    color=colors_kfold['student'], edgecolor='#1B5E20', linewidth=1.5)
    bars3 = ax3.bar(x + 0.5*width, pruned_r1, width, label='Pruned R1',
                    color=colors_kfold['pruned_r1'], edgecolor='#0D47A1', linewidth=1.5)
    bars4 = ax3.bar(x + 1.5*width, pruned_r2, width, label='Pruned R2',
                    color=colors_kfold['pruned_r2'], edgecolor='#4A148C', linewidth=1.5)

    # Add value labels on bars - only show labels for notable values to reduce clutter
    for bars, data in [(bars1, teacher), (bars2, student), (bars3, pruned_r1), (bars4, pruned_r2)]:
        for bar, height in zip(bars, data):
            # Only show labels for values that are notably different (not 99-100 range except extremes)
            if height < 95 or height == 100:
                color = '#D32F2F' if height < 95 else '#2E7D32'
                weight = 'bold'
                label = f'{height:.0f}' if height < 100 else '100'
                ax3.annotate(label,
                            xy=(bar.get_x() + bar.get_width()/2, height),
                            xytext=(0, 2), textcoords="offset points",
                            ha='center', va='bottom', fontsize=7, color=color,
                            fontweight=weight)

    ax3.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Cross-Validation Fold', fontsize=12, fontweight='bold')
    ax3.set_title('(c) Per-Fold Accuracy Results', fontsize=12, fontweight='bold', pad=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels(folds, fontweight='bold')
    ax3.set_ylim(91, 103)  # Increased y-limit to give more space for labels
    ax3.legend(loc='upper left', fontsize=9, ncol=2, framealpha=0.95)  # Moved legend to upper left, 2 columns
    ax3.grid(True, axis='y', alpha=0.4, linestyle='--')

    # Highlight problematic Fold 4 for R2
    ax3.add_patch(plt.Rectangle((3 + 0.8*width, 91), width + 0.1, 3,
                                  facecolor='none', edgecolor='#D32F2F',
                                  linewidth=2.5, linestyle='--'))
    ax3.annotate('Unstable!\n(93.36%)', xy=(3 + 1.5*width, 93.36), xytext=(3.8, 92.5),
                fontsize=9, color='#D32F2F', fontweight='bold', ha='center',
                arrowprops=dict(arrowstyle='->', color='#D32F2F', lw=1.5))

    # Add horizontal line for 99% threshold
    ax3.axhline(y=99, color='#43A047', linestyle='--', alpha=0.7, linewidth=1.5)
    ax3.text(-0.4, 99.3, '99% threshold', fontsize=9, color='#43A047', fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/kfold_comparison.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('figures/kfold_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Created: figures/kfold_comparison.pdf")


def create_accuracy_params_comparison():
    """Create enhanced accuracy vs parameters scatter plot - professional academic style"""
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#FAFAFA')

    # Define methods with more detail
    methods = {
        'DFUNet': {'accuracy': 94.5, 'params': 25, 'color': '#78909C', 'marker': 's',
                   'year': '2020', 'ref': '[22]', 'category': 'baseline'},
        'DFU_QUTNet': {'accuracy': 96.0, 'params': 15, 'color': '#78909C', 'marker': 's',
                       'year': '2020', 'ref': '[23]', 'category': 'baseline'},
        'EfficientNet Ensemble': {'accuracy': 97.2, 'params': 66, 'color': '#78909C', 'marker': 's',
                                   'year': '2021', 'ref': '[31]', 'category': 'baseline'},
        'Ours (Teacher)': {'accuracy': 99.72, 'params': 23.51, 'color': COLORS['teacher'], 'marker': 'o',
                           'year': '2025', 'ref': '', 'category': 'ours'},
        'Ours (Student)': {'accuracy': 99.81, 'params': 2.09, 'color': COLORS['student'], 'marker': 'o',
                           'year': '2025', 'ref': '', 'category': 'ours'},
        'Ours (Pruned)': {'accuracy': 98.67, 'params': 1.67, 'color': COLORS['pruned_r1'], 'marker': 'o',
                          'year': '2025', 'ref': '', 'category': 'ours'},
    }

    # Create background regions for efficiency zones
    ax.axvspan(0, 5, alpha=0.15, color='#4CAF50', zorder=1)
    ax.axvspan(5, 15, alpha=0.08, color='#FFC107', zorder=1)
    ax.axvspan(15, 80, alpha=0.05, color='#F44336', zorder=1)

    # Add zone labels at bottom
    ax.text(2.5, 93.3, 'Edge-Deployable\n(<5M params)', ha='center', fontsize=9,
            color='#2E7D32', fontweight='bold', style='italic')
    ax.text(10, 93.3, 'Mobile\n(5-15M)', ha='center', fontsize=9,
            color='#F57F17', fontweight='bold', style='italic')
    ax.text(45, 93.3, 'Server-Only\n(>15M)', ha='center', fontsize=9,
            color='#C62828', fontweight='bold', style='italic')

    # Plot baseline methods
    for name, data in methods.items():
        if data['category'] == 'baseline':
            ax.scatter(data['params'], data['accuracy'], s=250, c=data['color'],
                       marker=data['marker'], edgecolors='#424242', linewidth=2,
                       alpha=0.7, zorder=4)

    # Plot our methods with larger markers
    for name, data in methods.items():
        if data['category'] == 'ours':
            ax.scatter(data['params'], data['accuracy'], s=400, c=data['color'],
                       marker=data['marker'], edgecolors='#212121', linewidth=2.5,
                       zorder=5)

    # Add detailed annotations for baseline methods
    ax.annotate('DFUNet [22]\n94.5%, 25M', (25, 94.5), textcoords="offset points",
                xytext=(10, -25), fontsize=9, ha='left', color='#546E7A',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#90A4AE', alpha=0.9))
    ax.annotate('DFU_QUTNet [23]\n96.0%, 15M', (15, 96.0), textcoords="offset points",
                xytext=(10, 10), fontsize=9, ha='left', color='#546E7A',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#90A4AE', alpha=0.9))
    ax.annotate('EfficientNet\nEnsemble [31]\n97.2%, 66M', (66, 97.2), textcoords="offset points",
                xytext=(-55, 15), fontsize=9, ha='center', color='#546E7A',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#90A4AE', alpha=0.9))

    # Add detailed annotations for our methods with colored boxes - carefully positioned to avoid overlap
    ax.annotate('Teacher (ResNet-50)\n99.72%, 23.51M', (23.51, 99.72), textcoords="offset points",
                xytext=(15, -50), fontsize=9, ha='left', color=COLORS['teacher'], fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE', edgecolor=COLORS['teacher'], alpha=0.95),
                arrowprops=dict(arrowstyle='->', color=COLORS['teacher'], lw=1.2, alpha=0.7))
    ax.annotate('Student (MobileNetV2)\n99.81%, 2.09M', (2.09, 99.81), textcoords="offset points",
                xytext=(20, 15), fontsize=9, ha='left', color=COLORS['student'], fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', edgecolor=COLORS['student'], alpha=0.95),
                arrowprops=dict(arrowstyle='->', color=COLORS['student'], lw=1.2, alpha=0.7))
    ax.annotate('Pruned R1\n98.67%, 1.67M', (1.67, 98.67), textcoords="offset points",
                xytext=(-90, 40), fontsize=9, ha='center', color=COLORS['pruned_r1'], fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD', edgecolor=COLORS['pruned_r1'], alpha=0.95),
                arrowprops=dict(arrowstyle='->', color=COLORS['pruned_r1'], lw=1.2, alpha=0.7))

    # Knowledge Distillation arrow with annotation - positioned above to avoid overlap
    ax.annotate('', xy=(2.09, 99.81), xytext=(23.51, 99.72),
                arrowprops=dict(arrowstyle='-|>', color=COLORS['student'], lw=2.5,
                               connectionstyle='arc3,rad=0.15', mutation_scale=12))

    # KD label box positioned clearly above the arrow
    ax.add_patch(FancyBboxPatch((10, 100.2), 6, 0.5, boxstyle='round,pad=0.1',
                                 facecolor='#E8F5E9', edgecolor=COLORS['student'], linewidth=1.5))
    ax.text(13, 100.45, 'Knowledge Distillation (11.2× reduction)', ha='center', fontsize=9,
            color=COLORS['student'], fontweight='bold')

    # Pruning arrow - smaller, cleaner
    ax.annotate('', xy=(1.67, 98.67), xytext=(2.09, 99.81),
                arrowprops=dict(arrowstyle='-|>', color=COLORS['pruned_r1'], lw=2,
                               connectionstyle='arc3,rad=-0.25', mutation_scale=10))
    ax.text(-0.8, 99.1, 'Pruning', ha='center', fontsize=8, rotation=0,
            color=COLORS['pruned_r1'], fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#E3F2FD', edgecolor=COLORS['pruned_r1'], alpha=0.9))

    # Pareto frontier line (connecting our optimal points)
    pareto_x = [1.67, 2.09, 23.51]
    pareto_y = [98.67, 99.81, 99.72]
    ax.plot(pareto_x, pareto_y, '--', color='#9E9E9E', lw=1.5, alpha=0.5, zorder=2)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', zorder=0)
    ax.set_axisbelow(True)

    # Labels and title
    ax.set_xlabel('Parameters (Millions)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Accuracy vs Model Size: Comparison with State-of-the-Art DFU Classification Methods',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlim(-2, 75)
    ax.set_ylim(93, 101)

    # Custom legend
    legend_elements = [
        plt.scatter([], [], s=150, c='#78909C', marker='s', edgecolors='#424242',
                   linewidth=1.5, label='Prior Methods'),
        plt.scatter([], [], s=200, c=COLORS['teacher'], marker='o', edgecolors='#212121',
                   linewidth=2, label='Ours (Teacher)'),
        plt.scatter([], [], s=200, c=COLORS['student'], marker='o', edgecolors='#212121',
                   linewidth=2, label='Ours (Student)'),
        plt.scatter([], [], s=200, c=COLORS['pruned_r1'], marker='o', edgecolors='#212121',
                   linewidth=2, label='Ours (Pruned)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10,
              framealpha=0.95, edgecolor='#BDBDBD', title='Methods', title_fontsize=11)

    # Add efficiency arrow at top - within visible range
    ax.annotate('', xy=(5, 100.75), xytext=(65, 100.75),
                arrowprops=dict(arrowstyle='<-', color='#616161', lw=1.5))
    ax.text(35, 100.9, 'More Efficient', ha='center', fontsize=9, color='#616161', style='italic')

    plt.tight_layout()
    plt.savefig('figures/accuracy_params_comparison.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('figures/accuracy_params_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Created: figures/accuracy_params_comparison.pdf")


def create_pruning_stability():
    """Create pruning stability comparison - clean bar chart with error bars"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    models = ['Teacher', 'Student\n(KD)', 'Pruned\nR1', 'Pruned\nR2']
    means = [99.72, 99.81, 98.67, 98.39]
    stds = [0.38, 0.38, 0.63, 2.54]
    colors = [COLORS['teacher'], COLORS['student'], COLORS['pruned_r1'], COLORS['pruned_r2']]

    # Left: Mean accuracy with error bars
    bars = axes[0].bar(models, means, yerr=stds, capsize=6, color=colors,
                       edgecolor='black', linewidth=1.2, error_kw={'linewidth': 1.5})
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].set_title('Mean Accuracy with Standard Deviation', fontsize=13, fontweight='bold')
    axes[0].set_ylim(94, 102)

    for i, (mean, std) in enumerate(zip(means, stds)):
        axes[0].text(i, mean + std + 0.3, f'{mean}%\n±{std}%', ha='center', fontsize=8, fontweight='bold')

    axes[0].annotate('Unstable!', xy=(3, 98.39-2.54), xytext=(3.3, 95.5),
                     arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                     fontsize=10, color='red', fontweight='bold')

    # Right: Standard deviation comparison
    axes[1].bar(models, stds, color=colors, edgecolor='black', linewidth=1.2)
    axes[1].set_ylabel('Standard Deviation (%)', fontsize=12)
    axes[1].set_title('Variance Across Folds (Lower is Better)', fontsize=13, fontweight='bold')
    axes[1].set_ylim(0, 3.5)

    for i, std in enumerate(stds):
        axes[1].text(i, std + 0.1, f'±{std}%', ha='center', fontsize=10, fontweight='bold')

    axes[1].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    axes[1].text(3.5, 1.15, 'Acceptable\nthreshold', fontsize=9, color='red')

    plt.tight_layout()
    plt.savefig('figures/pruning_stability.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('figures/pruning_stability.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Created: figures/pruning_stability.pdf")


def create_speed_comparison():
    """Create inference speed comparison - clean HBFP-style bar chart"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    models = ['ResNet-50\n(Teacher)', 'MobileNetV2\n(Student)']
    colors = [COLORS['before'], COLORS['after']]

    # CPU Latency
    cpu_latency = [13.57, 5.11]
    bars1 = axes[0].bar(models, cpu_latency, color=colors, edgecolor='black', linewidth=1.2)
    axes[0].set_ylabel('Inference Time (ms)', fontsize=12)
    axes[0].set_title('CPU Inference Latency', fontsize=13, fontweight='bold')
    axes[0].set_ylim(0, 18)

    for i, v in enumerate(cpu_latency):
        axes[0].text(i, v + 0.5, f'{v} ms', ha='center', fontsize=11, fontweight='bold')

    # Speedup annotation
    axes[0].annotate('', xy=(1, 7), xytext=(0, 12),
                     arrowprops=dict(arrowstyle='->', color=COLORS['after'], lw=2.5))
    axes[0].text(0.5, 10, '2.65x\nfaster', ha='center', fontsize=12,
                 fontweight='bold', color=COLORS['after'])

    # GPU Latency
    gpu_latency = [1.42, 0.94]
    bars2 = axes[1].bar(models, gpu_latency, color=colors, edgecolor='black', linewidth=1.2)
    axes[1].set_ylabel('Inference Time (ms)', fontsize=12)
    axes[1].set_title('GPU Inference Latency', fontsize=13, fontweight='bold')
    axes[1].set_ylim(0, 2)

    for i, v in enumerate(gpu_latency):
        axes[1].text(i, v + 0.08, f'{v} ms', ha='center', fontsize=11, fontweight='bold')

    axes[1].annotate('', xy=(1, 1.05), xytext=(0, 1.35),
                     arrowprops=dict(arrowstyle='->', color=COLORS['after'], lw=2.5))
    axes[1].text(0.5, 1.25, '1.51x\nfaster', ha='center', fontsize=12,
                 fontweight='bold', color=COLORS['after'])

    plt.tight_layout()
    plt.savefig('figures/speed_comparison.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('figures/speed_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Created: figures/speed_comparison.pdf")


def create_tradeoff_chart():
    """Create enhanced accuracy-efficiency trade-off chart - professional style"""
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor('white')

    # Create main axis
    ax = fig.add_subplot(111)
    ax.set_facecolor('#FAFAFA')

    # Data with comprehensive metrics
    data = [
        {'name': 'Teacher (ResNet-50)', 'params': 23.51, 'acc': 99.72, 'latency': 13.57,
         'color': COLORS['teacher'], 'memory': 94.04, 'stage': 'baseline'},
        {'name': 'Student (MobileNetV2)', 'params': 2.09, 'acc': 99.81, 'latency': 5.11,
         'color': COLORS['student'], 'memory': 8.36, 'stage': 'kd'},
        {'name': 'Pruned Round 1', 'params': 1.67, 'acc': 98.67, 'latency': 4.5,
         'color': COLORS['pruned_r1'], 'memory': 6.68, 'stage': 'pruned'},
        {'name': 'Pruned Round 2', 'params': 1.34, 'acc': 98.39, 'latency': 4.2,
         'color': COLORS['pruned_r2'], 'memory': 5.36, 'stage': 'pruned'},
    ]

    # Background zones
    ax.axvspan(0, 3, alpha=0.12, color='#4CAF50', zorder=1)
    ax.axvspan(3, 10, alpha=0.06, color='#FFC107', zorder=1)
    ax.axvspan(10, 30, alpha=0.04, color='#F44336', zorder=1)

    # Zone labels
    ax.text(1.5, 97.55, 'Highly Efficient\n(<3M)', ha='center', fontsize=9,
            color='#2E7D32', fontweight='bold', style='italic')
    ax.text(6.5, 97.55, 'Moderate\n(3-10M)', ha='center', fontsize=9,
            color='#F57F17', fontweight='bold', style='italic')
    ax.text(20, 97.55, 'Heavy\n(>10M)', ha='center', fontsize=9,
            color='#C62828', fontweight='bold', style='italic')

    # Plot data points with bubble size based on speed (larger = faster)
    for d in data:
        # Size inversely proportional to latency (faster = bigger bubble)
        size = (18 - d['latency']) * 60
        ax.scatter(d['params'], d['acc'], s=size, c=d['color'],
                   edgecolors='#212121', linewidth=2.5, alpha=0.85, zorder=5)

    # Add simplified annotations for each point - clean, non-overlapping layout
    # KD arrow from Teacher to Student
    ax.annotate('', xy=(2.09, 99.81), xytext=(23.51, 99.72),
                arrowprops=dict(arrowstyle='-|>', color=COLORS['student'], lw=2.5,
                               connectionstyle='arc3,rad=0.12', mutation_scale=15),
                zorder=4)

    # Teacher annotation - positioned to the right
    ax.annotate('Teacher\n99.72%\n23.51M', (23.51, 99.72), textcoords="offset points",
                xytext=(12, -25), fontsize=9, ha='left', fontweight='bold',
                color=COLORS['teacher'],
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE',
                         edgecolor=COLORS['teacher'], linewidth=1.5, alpha=0.95))

    # Student annotation - positioned above
    ax.annotate('Student\n99.81%\n2.09M', (2.09, 99.81), textcoords="offset points",
                xytext=(12, 12), fontsize=9, ha='left', fontweight='bold',
                color=COLORS['student'],
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9',
                         edgecolor=COLORS['student'], linewidth=1.5, alpha=0.95))

    # Pruning arrows
    ax.annotate('', xy=(1.67, 98.67), xytext=(2.09, 99.81),
                arrowprops=dict(arrowstyle='-|>', color=COLORS['pruned_r1'], lw=2,
                               connectionstyle='arc3,rad=-0.15', mutation_scale=12),
                zorder=4)
    ax.annotate('', xy=(1.34, 98.39), xytext=(1.67, 98.67),
                arrowprops=dict(arrowstyle='-|>', color=COLORS['pruned_r2'], lw=1.5,
                               connectionstyle='arc3,rad=-0.15', mutation_scale=10),
                zorder=4)

    # Pruned R1 annotation - positioned to the left and up
    ax.annotate('Pruned R1\n98.67%', (1.67, 98.67), textcoords="offset points",
                xytext=(-60, 25), fontsize=9, ha='center', fontweight='bold',
                color=COLORS['pruned_r1'],
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD',
                         edgecolor=COLORS['pruned_r1'], linewidth=1.5, alpha=0.95))

    # Pruned R2 annotation - positioned below and further left
    ax.annotate('Pruned R2\n98.39%\n(Unstable)', (1.34, 98.39), textcoords="offset points",
                xytext=(-70, -40), fontsize=8, ha='center', fontweight='bold',
                color=COLORS['pruned_r2'],
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3E0',
                         edgecolor=COLORS['pruned_r2'], linewidth=1.5, alpha=0.95))

    # Stage label for KD - positioned above the arrow
    ax.add_patch(FancyBboxPatch((10, 100.1), 7, 0.35, boxstyle='round,pad=0.1',
                                 facecolor='#E8F5E9', edgecolor=COLORS['student'], linewidth=1.5))
    ax.text(13.5, 100.27, 'Stage 1: Knowledge Distillation', ha='center', fontsize=9,
            color=COLORS['student'], fontweight='bold')

    # Stage label for Pruning
    ax.add_patch(FancyBboxPatch((-0.8, 98.85), 2.8, 0.3, boxstyle='round,pad=0.1',
                                 facecolor='#E3F2FD', edgecolor=COLORS['pruned_r1'], linewidth=1.5))
    ax.text(0.6, 99.0, 'Stage 2: HBFP', ha='center', fontsize=8,
            color=COLORS['pruned_r1'], fontweight='bold')

    # Recommended model indicator - positioned clearly
    ax.annotate('BEST', (2.09, 99.81), textcoords="offset points",
                xytext=(-20, 40), fontsize=9, ha='center', color='white',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#2E7D32', edgecolor='#1B5E20', linewidth=2),
                arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.5))

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', zorder=0)
    ax.set_axisbelow(True)

    # Labels
    ax.set_xlabel('Parameters (Millions)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Model Compression Pipeline: Accuracy-Efficiency Trade-off Analysis',
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_xlim(-1, 28)
    ax.set_ylim(97.4, 100.5)

    # Legend for bubble sizes
    legend_sizes = [(700, '~4ms (Fastest)'), (450, '~8ms'), (200, '~14ms (Slowest)')]
    legend_bubbles = []
    for size, label in legend_sizes:
        legend_bubbles.append(ax.scatter([], [], s=size, c='#BDBDBD',
                                         edgecolors='#616161', linewidth=1.5,
                                         alpha=0.7, label=label))

    # Model legend
    model_legend = [
        ax.scatter([], [], s=200, c=COLORS['teacher'], edgecolors='#212121',
                  linewidth=2, label='Teacher'),
        ax.scatter([], [], s=200, c=COLORS['student'], edgecolors='#212121',
                  linewidth=2, label='Student (KD)'),
        ax.scatter([], [], s=200, c=COLORS['pruned_r1'], edgecolors='#212121',
                  linewidth=2, label='Pruned R1'),
        ax.scatter([], [], s=200, c=COLORS['pruned_r2'], edgecolors='#212121',
                  linewidth=2, label='Pruned R2'),
    ]

    # Create two legends
    legend1 = ax.legend(handles=model_legend, loc='upper right', fontsize=9,
                        framealpha=0.95, edgecolor='#BDBDBD',
                        title='Models', title_fontsize=10)
    ax.add_artist(legend1)

    legend2 = ax.legend(handles=legend_bubbles, loc='lower right', fontsize=9,
                        framealpha=0.95, edgecolor='#BDBDBD',
                        title='Inference Speed\n(Bubble Size)', title_fontsize=9)

    # Summary metrics box - positioned in lower left corner with better formatting
    summary_text = ("Key Results\n"
                    "─────────────\n"
                    "11.2× param ↓\n"
                    "2.65× faster\n"
                    "99.81% acc")
    ax.text(0.02, 0.18, summary_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', family='monospace', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                     edgecolor='#4CAF50', linewidth=2, alpha=0.98))

    plt.tight_layout()
    plt.savefig('figures/tradeoff_chart.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('figures/tradeoff_chart.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Created: figures/tradeoff_chart.pdf")


def create_kfold_vs_single_split():
    """Create K-Fold vs Single Split comparison figure"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    models = ['Single-Split', 'K-Fold Mean']
    teacher_acc = [100.00, 99.72]
    student_acc = [100.00, 99.81]

    x = np.arange(len(models))
    width = 0.35

    bars1 = axes[0].bar(x - width/2, teacher_acc, width, label='Teacher (ResNet-50)',
                        color=COLORS['teacher'], edgecolor='black', linewidth=1.2)
    bars2 = axes[0].bar(x + width/2, student_acc, width, label='Student (MobileNetV2)',
                        color=COLORS['student'], edgecolor='black', linewidth=1.2)

    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].set_title('Single-Split vs K-Fold Evaluation', fontsize=13, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models)
    axes[0].set_ylim(99, 100.5)
    axes[0].legend(loc='lower left', fontsize=9)

    for bar in bars1:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.05,
                     f'{height}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.05,
                     f'{height}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    axes[0].annotate('Single-split\noverestimates!', xy=(0, 100.05), xytext=(0.5, 100.3),
                     arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                     fontsize=9, color='red', fontweight='bold', ha='center')

    # Right: K-Fold distribution with scatter points showing individual folds
    fold_accuracies = {
        'Teacher': [99.05, 99.53, 100.00, 100.00, 100.00],
        'Student': [99.05, 100.00, 100.00, 100.00, 100.00]
    }

    positions = [1, 2]
    bp = axes[1].boxplot([fold_accuracies['Teacher'], fold_accuracies['Student']],
                          positions=positions, widths=0.5, patch_artist=True)

    colors_bp = [COLORS['teacher'], COLORS['student']]
    for patch, color in zip(bp['boxes'], colors_bp):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add individual fold points with jitter for visibility
    np.random.seed(42)
    for i, (key, vals) in enumerate(fold_accuracies.items()):
        jitter = np.random.uniform(-0.1, 0.1, len(vals))
        axes[1].scatter([positions[i] + j for j in jitter], vals,
                       color=colors_bp[i], edgecolor='white', s=60, zorder=4, alpha=0.8)

    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('K-Fold Accuracy Distribution', fontsize=13, fontweight='bold')
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(['Teacher', 'Student'])
    axes[1].set_ylim(98.5, 100.5)

    # Add mean lines with clearer labels
    for i, (key, vals) in enumerate(fold_accuracies.items()):
        mean_val = np.mean(vals)
        axes[1].scatter(positions[i], mean_val, marker='D', color='black', s=100, zorder=5)
        axes[1].axhline(y=mean_val, xmin=(positions[i]-0.3)/3, xmax=(positions[i]+0.3)/3,
                       color='black', linestyle='--', alpha=0.5, linewidth=1.5)
        axes[1].text(positions[i] + 0.35, mean_val, f'Mean: {mean_val:.2f}%',
                     fontsize=10, fontweight='bold', va='center')

    plt.tight_layout()
    plt.savefig('figures/kfold_vs_single_split.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('figures/kfold_vs_single_split.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Created: figures/kfold_vs_single_split.pdf")


def create_efficiency_gains():
    """Create efficiency gains summary - clean HBFP-style bar chart"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))

    categories = ['Teacher\n(ResNet-50)', 'Student\n(MobileNetV2)']
    colors = [COLORS['before'], COLORS['after']]

    # Parameter Reduction
    params = [23.51, 2.09]
    axes[0].bar(categories, params, color=colors, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('Parameters (Millions)', fontsize=12)
    axes[0].set_title('Parameter Reduction', fontsize=13, fontweight='bold')
    axes[0].set_ylim(0, 30)  # Increased to give more room
    for i, v in enumerate(params):
        axes[0].text(i, v + 0.8, f'{v}M', ha='center', fontsize=11, fontweight='bold')

    # Arrow positioned to not overlap with bars or text
    axes[0].annotate('', xy=(1, 6), xytext=(0, 18),
                     arrowprops=dict(arrowstyle='->', color=COLORS['after'], lw=2.5))
    # Label with background box for better readability
    axes[0].text(0.5, 12.5, '11.2x', ha='center', fontsize=13,
                 fontweight='bold', color=COLORS['after'],
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.8))

    # CPU Speedup
    latencies = [13.57, 5.11]
    axes[1].bar(categories, latencies, color=colors, edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('Inference Time (ms)', fontsize=12)
    axes[1].set_title('CPU Inference Speedup', fontsize=13, fontweight='bold')
    axes[1].set_ylim(0, 18)
    for i, v in enumerate(latencies):
        axes[1].text(i, v + 0.5, f'{v}ms', ha='center', fontsize=11, fontweight='bold')

    # Arrow positioned to not overlap
    axes[1].annotate('', xy=(1, 7), xytext=(0, 11.5),
                     arrowprops=dict(arrowstyle='->', color=COLORS['after'], lw=2.5))
    axes[1].text(0.5, 9.5, '2.65x', ha='center', fontsize=13,
                 fontweight='bold', color=COLORS['after'],
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.8))

    # Memory Footprint
    memory = [94.04, 8.36]
    axes[2].bar(categories, memory, color=colors, edgecolor='black', linewidth=1.5)
    axes[2].set_ylabel('Memory (MB)', fontsize=12)
    axes[2].set_title('Memory Footprint', fontsize=13, fontweight='bold')
    axes[2].set_ylim(0, 115)  # Increased to give more room
    for i, v in enumerate(memory):
        axes[2].text(i, v + 2.5, f'{v}MB', ha='center', fontsize=11, fontweight='bold')

    # Arrow positioned to not overlap
    axes[2].annotate('', xy=(1, 18), xytext=(0, 80),
                     arrowprops=dict(arrowstyle='->', color=COLORS['after'], lw=2.5))
    axes[2].text(0.5, 50, '11.2x', ha='center', fontsize=13,
                 fontweight='bold', color=COLORS['after'],
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.8))

    plt.tight_layout()
    plt.savefig('figures/efficiency_gains.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('figures/efficiency_gains.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Created: figures/efficiency_gains.pdf")


def create_model_comparison():
    """Create comparison with related methods"""
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = ['DFUNet', 'DFU_QUTNet', 'EfficientNet\nEnsemble', 'Ours\n(Teacher)', 'Ours\n(Student)', 'Ours\n(Pruned R1)']
    accuracy = [94.5, 96.0, 97.2, 99.72, 99.81, 98.67]
    params = [25, 15, 66, 23.51, 2.09, 2.09]

    sizes = [p * 15 for p in params]
    colors = [COLORS['gray'], COLORS['gray'], COLORS['gray'],
              COLORS['teacher'], COLORS['student'], COLORS['pruned_r1']]

    scatter = ax.scatter(range(len(methods)), accuracy, s=sizes, c=colors,
                         edgecolors='black', linewidth=1.5, alpha=0.8)

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Comparison with Related DFU Classification Methods\n(Bubble size = Parameter count)',
                 fontsize=13, fontweight='bold')
    ax.set_ylim(93, 101)

    for i, (acc, param) in enumerate(zip(accuracy, params)):
        ax.annotate(f'{param}M', (i, acc), textcoords="offset points",
                    xytext=(0, 15), ha='center', fontsize=8)

    ax.axvspan(2.5, 5.5, alpha=0.1, color='green')
    ax.text(4, 94, 'Our Methods', ha='center', fontsize=10, fontweight='bold', color=COLORS['after'])

    plt.tight_layout()
    plt.savefig('figures/model_comparison.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('figures/model_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Created: figures/model_comparison.pdf")


def create_compression_summary():
    """Create compression summary infographic"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(6, 7.5, 'Model Compression Summary', ha='center', fontsize=16, fontweight='bold')

    # Teacher box
    ax.add_patch(FancyBboxPatch((0.5, 4), 3, 2.5, boxstyle='round,pad=0.1',
                                 facecolor='#FFCDD2', edgecolor='black', linewidth=2))
    ax.text(2, 6, 'Teacher Model', ha='center', fontsize=12, fontweight='bold')
    ax.text(2, 5.5, 'ResNet-50', ha='center', fontsize=10)
    ax.text(2, 5, '23.51M params', ha='center', fontsize=10)
    ax.text(2, 4.5, '99.72% accuracy', ha='center', fontsize=10)
    ax.text(2, 4.2, '13.57ms CPU', ha='center', fontsize=9, color='gray')

    ax.annotate('', xy=(4.5, 5.25), xytext=(3.5, 5.25),
                arrowprops=dict(arrowstyle='->', color=COLORS['after'], lw=3))
    ax.text(4, 5.8, 'Knowledge\nDistillation', ha='center', fontsize=9, fontweight='bold', color=COLORS['after'])

    # Student box
    ax.add_patch(FancyBboxPatch((4.5, 4), 3, 2.5, boxstyle='round,pad=0.1',
                                 facecolor='#C8E6C9', edgecolor='black', linewidth=2))
    ax.text(6, 6, 'Student Model', ha='center', fontsize=12, fontweight='bold')
    ax.text(6, 5.5, 'MobileNetV2', ha='center', fontsize=10)
    ax.text(6, 5, '2.09M params', ha='center', fontsize=10)
    ax.text(6, 4.5, '99.81% accuracy', ha='center', fontsize=10)
    ax.text(6, 4.2, '5.11ms CPU', ha='center', fontsize=9, color='gray')

    ax.annotate('', xy=(8.5, 5.25), xytext=(7.5, 5.25),
                arrowprops=dict(arrowstyle='->', color=COLORS['pruned_r1'], lw=3))
    ax.text(8, 5.8, 'HBFP\nPruning', ha='center', fontsize=9, fontweight='bold', color=COLORS['pruned_r1'])

    # Pruned box
    ax.add_patch(FancyBboxPatch((8.5, 4), 3, 2.5, boxstyle='round,pad=0.1',
                                 facecolor='#BBDEFB', edgecolor='black', linewidth=2))
    ax.text(10, 6, 'Pruned Model', ha='center', fontsize=12, fontweight='bold')
    ax.text(10, 5.5, 'MobileNetV2-P', ha='center', fontsize=10)
    ax.text(10, 5, '~1.7M params', ha='center', fontsize=10)
    ax.text(10, 4.5, '98.67% accuracy', ha='center', fontsize=10)
    ax.text(10, 4.2, '~4.5ms CPU', ha='center', fontsize=9, color='gray')

    # Summary metrics
    metrics = [
        ('Parameter Reduction', '11.2x', COLORS['after']),
        ('CPU Speedup', '2.65x', COLORS['pruned_r1']),
        ('Accuracy Retained', '99.81%', COLORS['teacher']),
    ]

    for i, (label, value, color) in enumerate(metrics):
        x = 2 + i * 4
        ax.add_patch(FancyBboxPatch((x-1.5, 1), 3, 2, boxstyle='round,pad=0.1',
                                     facecolor=color, edgecolor='black', linewidth=2, alpha=0.3))
        ax.text(x, 2.3, value, ha='center', fontsize=18, fontweight='bold', color=color)
        ax.text(x, 1.5, label, ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/compression_summary.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('figures/compression_summary.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Created: figures/compression_summary.pdf")


def create_pruning_motivation():
    """Create figure showing why we prune and the benefits - Enhanced visual design"""
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor('white')

    # Create grid for subplots with better spacing
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    # Enhanced color palette
    colors_enhanced = {
        'redundant1': '#E53935',      # Vibrant red
        'redundant2': '#FF7043',      # Orange-red
        'important': '#43A047',       # Green
        'unique': '#1E88E5',          # Blue
        'before': '#C62828',          # Dark red
        'after': '#2E7D32',           # Dark green
        'bg_light': '#FAFAFA',        # Light background
    }

    # ============ Top Left: Filter Redundancy Visualization ============
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#FAFAFA')

    np.random.seed(42)
    epochs = np.arange(1, 16)

    # Create smoother filter trajectories
    filter1 = 0.85 + 0.08 * np.sin(epochs * 0.4) + np.random.normal(0, 0.015, len(epochs))
    filter2 = 0.82 + 0.08 * np.sin(epochs * 0.4 + 0.05) + np.random.normal(0, 0.015, len(epochs))
    filter3 = 0.55 + 0.04 * epochs / 15 + np.random.normal(0, 0.015, len(epochs))
    filter4 = 0.52 + 0.04 * epochs / 15 + np.random.normal(0, 0.015, len(epochs))
    filter5 = 0.35 + 0.15 * np.log(epochs + 1) / np.log(16) + np.random.normal(0, 0.015, len(epochs))

    # Plot with enhanced styling
    ax1.plot(epochs, filter1, 'o-', color=colors_enhanced['redundant1'], linewidth=2.5,
             markersize=6, label='Filter A', markeredgecolor='white', markeredgewidth=0.5)
    ax1.plot(epochs, filter2, 's-', color=colors_enhanced['redundant2'], linewidth=2.5,
             markersize=6, label='Filter B (similar to A)', markeredgecolor='white', markeredgewidth=0.5)
    ax1.plot(epochs, filter3, 'o-', color=colors_enhanced['important'], linewidth=2.5,
             markersize=6, label='Filter C', markeredgecolor='white', markeredgewidth=0.5)
    ax1.plot(epochs, filter4, 's-', color='#66BB6A', linewidth=2.5,
             markersize=6, label='Filter D (similar to C)', markeredgecolor='white', markeredgewidth=0.5)
    ax1.plot(epochs, filter5, '^-', color=colors_enhanced['unique'], linewidth=2.5,
             markersize=7, label='Filter E (unique)', markeredgecolor='white', markeredgewidth=0.5)

    # Shade redundant pairs with gradient effect
    ax1.fill_between(epochs, filter1, filter2, alpha=0.25, color=colors_enhanced['redundant1'],
                     label='_nolegend_')
    ax1.fill_between(epochs, filter3, filter4, alpha=0.25, color=colors_enhanced['important'],
                     label='_nolegend_')

    ax1.set_xlabel('Training Epoch', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Filter $\ell_1$-norm', fontweight='bold', fontsize=11)
    ax1.set_title('(a) Filter Importance Trajectories During Training', fontweight='bold', fontsize=12, pad=10)
    ax1.legend(loc='upper right', fontsize=8, framealpha=0.95, edgecolor='gray')
    ax1.grid(True, alpha=0.4, linestyle='--')
    ax1.set_xlim(1, 15)
    ax1.set_ylim(0.25, 1.0)

    # Enhanced annotation with box
    ax1.annotate('Redundant Pair\n(Prune one)', xy=(13, 0.835), xytext=(9, 0.68),
                fontsize=9, ha='center', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#424242', lw=2,
                               connectionstyle='arc3,rad=0.2'),
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFEB3B',
                         edgecolor='#F57F17', alpha=0.9, linewidth=1.5))

    # ============ Top Right: Before vs After Pruning Network ============
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(0, 12)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('(b) Network Architecture: Before vs After Pruning', fontweight='bold', fontsize=12, pad=10)

    # Before pruning section
    before_x = 1
    ax2.add_patch(FancyBboxPatch((before_x - 0.3, 0.8), 4.5, 8.5, boxstyle='round,pad=0.1',
                                  facecolor='#FFEBEE', edgecolor=colors_enhanced['before'],
                                  linewidth=2, alpha=0.5))
    ax2.text(before_x + 1.9, 9.5, 'Before Pruning', ha='center', fontsize=12,
             fontweight='bold', color=colors_enhanced['before'])

    # Draw neural network layers - before
    layer_y = [7, 5, 3]
    before_filters = [8, 6, 4]
    for i, (y, n_f) in enumerate(zip(layer_y, before_filters)):
        # Layer box
        width = n_f * 0.4 + 0.3
        ax2.add_patch(FancyBboxPatch((before_x + 0.2, y - 0.4), width, 1.2,
                                      boxstyle='round,pad=0.05',
                                      facecolor=colors_enhanced['before'],
                                      edgecolor='#B71C1C', linewidth=1.5, alpha=0.8))
        # Filter circles
        for j in range(n_f):
            circle = plt.Circle((before_x + 0.6 + j * 0.4, y + 0.2), 0.15,
                               facecolor='white', edgecolor='#B71C1C', linewidth=1)
            ax2.add_patch(circle)
        ax2.text(before_x + 0.2 + width/2, y - 0.7, f'Conv {i+1}: {n_f} filters',
                ha='center', fontsize=10, fontweight='bold')

    # Connections between layers - before
    for i in range(2):
        ax2.annotate('', xy=(before_x + 2, layer_y[i+1] + 0.8),
                    xytext=(before_x + 2, layer_y[i] - 0.4),
                    arrowprops=dict(arrowstyle='->', color='#B71C1C', lw=1.5))

    # After pruning section
    after_x = 7
    ax2.add_patch(FancyBboxPatch((after_x - 0.3, 0.8), 4.5, 8.5, boxstyle='round,pad=0.1',
                                  facecolor='#E8F5E9', edgecolor=colors_enhanced['after'],
                                  linewidth=2, alpha=0.5))
    ax2.text(after_x + 1.9, 9.5, 'After Pruning', ha='center', fontsize=12,
             fontweight='bold', color=colors_enhanced['after'])

    # Draw neural network layers - after (pruned)
    after_filters = [5, 4, 3]
    for i, (y, n_f) in enumerate(zip(layer_y, after_filters)):
        width = n_f * 0.45 + 0.3
        ax2.add_patch(FancyBboxPatch((after_x + 0.2, y - 0.4), width, 1.2,
                                      boxstyle='round,pad=0.05',
                                      facecolor=colors_enhanced['after'],
                                      edgecolor='#1B5E20', linewidth=1.5, alpha=0.8))
        for j in range(n_f):
            circle = plt.Circle((after_x + 0.6 + j * 0.45, y + 0.2), 0.17,
                               facecolor='white', edgecolor='#1B5E20', linewidth=1)
            ax2.add_patch(circle)
        ax2.text(after_x + 0.2 + width/2, y - 0.7, f'Conv {i+1}: {n_f} filters',
                ha='center', fontsize=10, fontweight='bold')

    # Connections between layers - after
    for i in range(2):
        ax2.annotate('', xy=(after_x + 1.5, layer_y[i+1] + 0.8),
                    xytext=(after_x + 1.5, layer_y[i] - 0.4),
                    arrowprops=dict(arrowstyle='->', color='#1B5E20', lw=1.5))

    # Arrow between networks
    ax2.annotate('', xy=(6.4, 5), xytext=(5.2, 5),
                arrowprops=dict(arrowstyle='-|>', color='#424242', lw=3,
                               mutation_scale=20))
    ax2.text(5.8, 6, 'HBFP', ha='center', fontsize=11, fontweight='bold', color='#424242')
    ax2.text(5.8, 5.5, 'Pruning', ha='center', fontsize=10, color='#616161')

    # Reduction labels
    ax2.text(5.8, 3.8, '-33%', ha='center', fontsize=10, fontweight='bold',
             color=colors_enhanced['after'],
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=colors_enhanced['after']))

    # ============ Bottom Left: Improvement Metrics ============
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor('#FAFAFA')

    metrics = ['Parameters', 'FLOPs', 'Memory', 'Latency']
    before_values = [100, 100, 100, 100]  # Normalized to 100%
    after_values = [80, 80, 80, 88]  # After pruning percentages
    reduction = [20, 20, 20, 12]

    x = np.arange(len(metrics))
    width = 0.35

    # Create bars with shadow effect
    bars1 = ax3.bar(x - width/2, before_values, width, label='Before Pruning',
                    color=colors_enhanced['before'], edgecolor='#8B0000', linewidth=1.5,
                    zorder=3)
    bars2 = ax3.bar(x + width/2, after_values, width, label='After Pruning',
                    color=colors_enhanced['after'], edgecolor='#1B5E20', linewidth=1.5,
                    zorder=3)

    # Add percentage labels
    for i, (bar1, bar2, red) in enumerate(zip(bars1, bars2, reduction)):
        ax3.annotate('100%', xy=(bar1.get_x() + bar1.get_width()/2, 102),
                    ha='center', va='bottom', fontsize=10, fontweight='bold',
                    color=colors_enhanced['before'])
        ax3.annotate(f'{after_values[i]}%', xy=(bar2.get_x() + bar2.get_width()/2, after_values[i] + 2),
                    ha='center', va='bottom', fontsize=10, fontweight='bold',
                    color=colors_enhanced['after'])
        # Reduction arrow
        ax3.annotate(f'-{red}%', xy=(x[i], 50), ha='center', fontsize=11,
                    fontweight='bold', color='#E65100',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3E0',
                             edgecolor='#E65100', linewidth=1.5))

    ax3.set_ylabel('Relative Value (%)', fontweight='bold', fontsize=11)
    ax3.set_title('(c) Resource Reduction After Pruning', fontweight='bold', fontsize=12, pad=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics, fontweight='bold')
    ax3.set_ylim(0, 120)
    ax3.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax3.grid(True, axis='y', alpha=0.4, linestyle='--', zorder=0)
    ax3.axhline(y=100, color='gray', linestyle='--', alpha=0.5, zorder=1)

    # ============ Bottom Right: Key Benefits Summary ============
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)
    ax4.axis('off')
    ax4.set_title('(d) Key Benefits of Pruning', fontweight='bold', fontsize=12, pad=10)

    # Background
    ax4.add_patch(FancyBboxPatch((0.2, 0.5), 9.6, 9, boxstyle='round,pad=0.1',
                                  facecolor='#F5F5F5', edgecolor='#BDBDBD', linewidth=1))

    benefits = [
        ('Computational Efficiency', '20% fewer FLOPs', '#E53935', 'Faster processing'),
        ('Memory Savings', '20% less storage', '#1E88E5', 'Edge deployment'),
        ('Faster Inference', '12% speedup', '#43A047', 'Real-time capable'),
        ('Accuracy Preserved', '98.67% maintained', '#7B1FA2', 'Only 1.1% drop'),
    ]

    for i, (title, value, color, subtitle) in enumerate(benefits):
        y_pos = 8.2 - i * 2

        # Benefit card
        ax4.add_patch(FancyBboxPatch((0.5, y_pos - 0.7), 9, 1.7, boxstyle='round,pad=0.1',
                                      facecolor='white', edgecolor=color, linewidth=2.5))

        # Color indicator bar on left
        ax4.add_patch(plt.Rectangle((0.5, y_pos - 0.7), 0.4, 1.7, facecolor=color, alpha=0.9))

        # Icon circle
        circle = plt.Circle((1.5, y_pos + 0.15), 0.35, facecolor=color, edgecolor='white', linewidth=2)
        ax4.add_patch(circle)
        ax4.text(1.5, y_pos + 0.15, str(i+1), ha='center', va='center',
                fontsize=12, fontweight='bold', color='white')

        # Text content
        ax4.text(2.2, y_pos + 0.45, title, fontsize=11, fontweight='bold', va='center', color='#212121')
        ax4.text(2.2, y_pos - 0.1, value, fontsize=12, fontweight='bold', va='center', color=color)
        ax4.text(9.2, y_pos + 0.15, subtitle, fontsize=9, va='center', ha='right',
                color='#757575', style='italic')

    # Bottom note
    ax4.text(5, 0.1, 'HBFP identifies consistently redundant filters across training history',
            ha='center', fontsize=9, style='italic', color='#616161')

    plt.savefig('figures/pruning_motivation.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('figures/pruning_motivation.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Created: figures/pruning_motivation.pdf")


if __name__ == '__main__':
    print("Generating figures for DFU Classification paper...")
    print("Style inspired by HBFP paper (Basha et al.)")
    print("=" * 50)

    # DFU sample images
    create_dfu_samples()

    # Methodology figures
    create_methodology_workflow()
    create_kd_architecture()
    create_pruning_motivation()
    create_pruning_workflow()

    # Results figures
    create_efficiency_comparison()
    create_kfold_comparison()
    create_model_comparison()

    # Comparison figures
    create_accuracy_params_comparison()
    create_pruning_stability()
    create_speed_comparison()
    create_tradeoff_chart()
    create_compression_summary()

    # Discussion figures
    create_kfold_vs_single_split()
    create_efficiency_gains()

    print("=" * 50)
    print("All figures generated successfully!")
    print("Figures saved in: figures/")
