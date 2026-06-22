"""
Visualization for the segmentation pipeline.
Two plot functions extracted from the original notebook cells 93 and 116.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_segmentation(
    signal: np.ndarray,
    prob_map: np.ndarray,
    mask: np.ndarray,
    query_text: str,
    tag: str,
    true_start: int | None = None,
    true_end: int | None = None,
    save_path: str | None = None,
) -> None:
    """
    Three-panel plot: signal + mask overlay, probability trace, binary mask.

    Parameters
    ----------
    signal     : raw signal array
    prob_map   : per-timestep anomaly probability from segment_signal()
    mask       : binary anomaly mask
    query_text : query string used for the run
    tag        : signal tag name
    true_start/end : ground-truth anomaly indices (optional, shown in green)
    save_path  : if provided, save figure to this path
    """
    fig, axes = plt.subplots(
        3, 1, figsize=(14, 10),
        gridspec_kw={'height_ratios': [3, 1, 1]},
    )
    t = np.arange(len(signal))

    # Panel 1 - signal with mask overlay
    ax1 = axes[0]
    ax1.plot(t, signal, color='steelblue', linewidth=0.8, label='Signal')

    in_anomaly = False
    for i in range(len(mask)):
        if mask[i] == 1 and not in_anomaly:
            anom_start = i
            in_anomaly = True
        elif mask[i] == 0 and in_anomaly:
            ax1.axvspan(anom_start, i, alpha=0.3, color='red',
                        label='Predicted anomaly')
            in_anomaly = False
    if in_anomaly:
        ax1.axvspan(anom_start, len(mask), alpha=0.3, color='red')

    if true_start is not None and true_end is not None:
        ax1.axvspan(true_start, true_end, alpha=0.15, color='green',
                    label='Ground truth')

    ax1.set_title(f'{tag}\nQuery: "{query_text}"', fontsize=11)
    ax1.set_ylabel('Value')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2 - probability
    ax2 = axes[1]
    ax2.fill_between(t, prob_map, alpha=0.7, color='orange')
    ax2.axhline(0.65, color='red', linestyle='--', linewidth=1,
                label='Threshold=0.65')
    ax2.set_ylabel('Anomaly\nProbability')
    ax2.set_ylim(0, 1)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Panel 3 - binary mask
    ax3 = axes[2]
    ax3.fill_between(t, mask, step='mid', alpha=0.8, color='red')
    ax3.set_ylabel('Binary\nMask')
    ax3.set_xlabel('Timestep (15-min intervals)')
    ax3.set_ylim(-0.1, 1.1)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved → {save_path}')
    plt.show()


def plot_segmentation_heatmap(
    vals: np.ndarray,
    tag: str,
    area: str,
    queries_to_test: list[str],
    mlp,
    encode_text_fn,
    get_chronos_fn,
    window_size: int = 128,
    threshold: float = 0.65,
    save_path: str | None = None,
) -> None:
    """
    Multi-query heatmap: one row per query, columns = timesteps.
    Red = high anomaly probability, green = low.

    Parameters
    ----------
    vals            : signal array (ideally downsampled to 1024 pts)
    tag, area       : signal metadata for plot title
    queries_to_test : list of natural-language query strings
    mlp             : trained TSAISegmentationMLP
    encode_text_fn  : callable(str) → np.ndarray[5120]
    get_chronos_fn  : callable(np.ndarray) → np.ndarray[512]
    window_size     : sliding window length
    threshold       : decision threshold (used for binary panel)
    save_path       : optional output file path
    """
    from src.preprocessing.windows import extract_windows
    import torch

    n            = len(vals)
    prob_matrix  = np.zeros((len(queries_to_test), n))

    print(f'Generating heatmap for {tag}...')
    for qi, query in enumerate(queries_to_test):
        txt_emb    = encode_text_fn(query)
        txt_tensor = torch.tensor(txt_emb, dtype=torch.float32).unsqueeze(0)

        windows, positions = extract_windows(vals, window_size=window_size)
        prob_map  = np.zeros(n)
        count_map = np.zeros(n)

        mlp.eval()
        with torch.no_grad():
            for window, (s, e) in zip(windows, positions):
                emb = get_chronos_fn(np.array(window, dtype=np.float32))
                if emb is None:
                    continue
                ts_tensor = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)
                logit     = mlp(ts_tensor, txt_tensor)
                prob      = torch.sigmoid(logit).item()
                prob_map[s:e]  += prob
                count_map[s:e] += 1

        mask = count_map > 0
        prob_map[mask] /= count_map[mask]
        prob_matrix[qi] = prob_map
        print(f'  [{qi+1}/{len(queries_to_test)}] {query[:40]}')

    fig, axes = plt.subplots(
        3, 1, figsize=(16, 11),
        gridspec_kw={'height_ratios': [2, 4, 1.5]},
    )
    ts_x = np.arange(n) * 15 / 60

    # Panel 1 - original signal
    ax1 = axes[0]
    ax1.plot(ts_x, vals, color='steelblue', linewidth=0.8)
    ax1.set_title(f'{tag}  ({area}) - Original Signal (90 days)',
                  fontsize=13, fontweight='bold')
    ax1.set_ylabel('Value', fontsize=11)
    ax1.set_xlim(0, ts_x[-1])
    ax1.grid(True, alpha=0.3)

    # Panel 2 - heatmap
    ax2 = axes[1]
    im  = ax2.imshow(
        prob_matrix,
        aspect='auto',
        cmap='RdYlGn_r',
        vmin=0, vmax=1,
        extent=[0, ts_x[-1], len(queries_to_test), 0],
        interpolation='nearest',
    )
    ax2.set_yticks(np.arange(len(queries_to_test)) + 0.5)
    short_q = [q[:38] + '..' if len(q) > 38 else q for q in queries_to_test]
    ax2.set_yticklabels(short_q, fontsize=9)
    ax2.set_title('Anomaly Probability Heatmap per Query (red=high, green=low)',
                  fontsize=12, fontweight='bold')
    for i in range(1, len(queries_to_test)):
        ax2.axhline(y=i, color='white', linewidth=0.5)
    plt.colorbar(im, ax=ax2, label='Anomaly Probability',
                 fraction=0.02, pad=0.02)

    # Panel 3 - binary mask for the first spike/anomaly query
    ax3 = axes[2]
    spike_prob = prob_matrix[2] if len(queries_to_test) > 2 else prob_matrix[0]
    binary     = (spike_prob > threshold).astype(float)
    ax3.fill_between(ts_x, binary, color='red', alpha=0.7, step='mid',
                     label='Anomaly detected')
    ax3.plot(ts_x, spike_prob, color='darkred', linewidth=0.8, alpha=0.5)
    ax3.axhline(threshold, color='black', linestyle='--', linewidth=1,
                label=f'Threshold={threshold}')
    ax3.set_ylabel('Probability\n(query 3)', fontsize=9)
    ax3.set_xlabel('Time (hours from dataset start)', fontsize=11)
    ax3.set_ylim(-0.05, 1.1)
    ax3.legend(fontsize=8, loc='upper right')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved → {save_path}')
    plt.show()
