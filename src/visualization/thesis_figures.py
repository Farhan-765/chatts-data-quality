"""
Thesis figure generators.
All figures are saved to figures/ directory and shown inline.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_approach_comparison(
    results_approach1: list[dict],
    results_approach2: list[dict],
    results_approach3: list[dict],
    ground_truth: dict,
    save_path: str | None = None,
) -> None:
    """
    Figure 1 - Side-by-side accuracy bar chart for all three approaches.
    """
    approaches = ['Approach 1\n(Statistical)', 'Approach 2\n(Stats Embedded)', 'Approach 3\n(Hybrid)']
    result_sets = [results_approach1, results_approach2, results_approach3]

    accuracies = []
    for results in result_sets:
        n_correct = sum(
            1 for r in results
            if r.get('Category') == ground_truth.get(r.get('Tag', ''))
        )
        accuracies.append(100 * n_correct / max(len(results), 1))

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(approaches, accuracies, color=['#4472C4', '#ED7D31', '#70AD47'],
                  edgecolor='black', linewidth=0.7)
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylim(0, 110)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('ChatTS-14B - Classification Accuracy by Approach\n(55 industrial signals)',
                 fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.4)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved → {save_path}')
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    title: str = 'Confusion Matrix',
    save_path: str | None = None,
) -> None:
    """
    Figure 2 - Confusion matrix heatmap.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap='Blues')

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, str(cm[i, j]),
                           ha='center', va='center', fontsize=10,
                           color='white' if cm[i, j] > cm.max() / 2 else 'black')

    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True',      fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved → {save_path}')
    plt.show()


def plot_prescreener_analysis(
    tags: list[str],
    detected_flags: list[list[str]],
    save_path: str | None = None,
) -> None:
    """
    Figure 3 - Pre-screener flag frequency bar chart.
    """
    flag_counts = {'stale': 0, 'drift': 0, 'spikes': 0,
                   'var_collapse': 0, 'intermittent': 0, 'clean': 0}
    for flags in detected_flags:
        for f in flags:
            if f in flag_counts:
                flag_counts[f] += 1

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {
        'stale': '#2196F3', 'drift': '#FF9800', 'spikes': '#F44336',
        'var_collapse': '#9C27B0', 'intermittent': '#795548', 'clean': '#4CAF50',
    }
    labels = list(flag_counts.keys())
    counts = [flag_counts[l] for l in labels]
    bars   = ax.bar(labels, counts, color=[colors[l] for l in labels],
                    edgecolor='black', linewidth=0.6)
    for bar, cnt in zip(bars, counts):
        if cnt > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    str(cnt), ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('Number of signals flagged', fontsize=11)
    ax.set_title('Statistical Pre-Screener Flag Distribution (55 signals)',
                 fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.4)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved → {save_path}')
    plt.show()


def plot_mlp_training_curve(
    history: dict,
    save_path: str | None = None,
) -> None:
    """
    Figure 4 - MLP training loss and F1 curves.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], label='Train loss',  color='#2196F3')
    ax1.plot(epochs, history['val_loss'],   label='Val loss',    color='#F44336')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('BCE Loss')
    ax1.set_title('MLP Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.4)

    ax2.plot(epochs, history['val_f1'], label='Val F1', color='#4CAF50')
    ax2.axhline(0.699, color='orange', linestyle='--',
                label='Final F1=0.699 (threshold=0.65)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('MLP Validation F1')
    ax2.legend()
    ax2.grid(True, alpha=0.4)

    plt.suptitle('TSAISegmentationMLP Training - Combined Industrial + Synthetic Data',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved → {save_path}')
    plt.show()
