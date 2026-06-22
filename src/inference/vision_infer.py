"""
Vision-modality inference: render a time-series as a line chart and
send the image to GPT-4o for anomaly classification.

This module provides the chart rendering layer that sits on top of
call_gpt4o_vision() in openai_infer.py.
"""

import os
import tempfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.inference.openai_infer import call_gpt4o_vision
from src.prompts.templates import MCQ_CATEGORIES


_SYSTEM_VISION = (
    'You are an expert industrial process engineer reviewing sensor charts. '
    'Analyze the time-series line chart and classify the anomaly shown. '
    'Always start your answer with the single category letter.'
)


def render_series_chart(
    vals: np.ndarray,
    tag: str,
    title: str = '',
    figsize: tuple = (10, 3),
    save_path: str | None = None,
) -> str:
    """
    Render a time-series array as a clean line chart.

    Parameters
    ----------
    vals      : 1-D signal array
    tag       : sensor tag name (used as y-label)
    title     : optional chart title
    figsize   : matplotlib figure size
    save_path : explicit file path; if None, a temp PNG is created

    Returns
    -------
    str - path to the saved PNG
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(vals, linewidth=0.8, color='steelblue')
    ax.set_xlabel('Time index (15-min intervals)')
    ax.set_ylabel(tag)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is None:
        fd, save_path = tempfile.mkstemp(suffix='.png')
        os.close(fd)

    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return save_path


def render_annotated_chart(
    vals: np.ndarray,
    tag: str,
    stats: dict,
    detected: list[str],
    figsize: tuple = (10, 3.5),
    save_path: str | None = None,
) -> str:
    """
    Chart with statistics annotations (mean±std band, threshold lines).
    Used in the hybrid-vision variant that gives GPT-4o the same context
    as Approach 3 (stats embedded visually rather than in text).
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(vals, linewidth=0.8, color='steelblue', label=tag)

    mean = stats.get('mean', vals.mean())
    std  = stats.get('std',  vals.std())
    ax.axhline(mean, color='orange', linewidth=1, linestyle='--', label=f'mean={mean:.3f}')
    ax.axhspan(mean - std, mean + std, alpha=0.08, color='orange', label='±1σ')

    spike_hi = stats.get('spike_hi')
    spike_lo = stats.get('spike_lo')
    if spike_hi is not None:
        ax.axhline(spike_hi, color='red', linewidth=0.8, linestyle=':', label=f'+4σ={spike_hi:.3f}')
    if spike_lo is not None:
        ax.axhline(spike_lo, color='red', linewidth=0.8, linestyle=':', label=f'-4σ={spike_lo:.3f}')

    det_str = ', '.join(detected) if detected else 'none'
    ax.set_title(f'{tag}  |  pre-screen detected: {det_str}')
    ax.set_xlabel('Time index (15-min intervals)')
    ax.set_ylabel(tag)
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is None:
        fd, save_path = tempfile.mkstemp(suffix='.png')
        os.close(fd)

    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return save_path


def ask_vision_naive(
    vals: np.ndarray,
    tag: str,
    model: str = 'gpt-4o',
) -> str:
    """
    Naive vision baseline: plain chart, open-ended question, no MCQ.
    """
    chart_path = render_series_chart(vals, tag)
    question = (
        f'This is a time-series chart from industrial sensor [{tag}], '
        f'sampled every 15 minutes. '
        'Is there any anomaly visible? If yes, what type and where?'
    )
    try:
        return call_gpt4o_vision(chart_path, question, model=model)
    finally:
        os.unlink(chart_path)


def ask_vision_mcq(
    vals: np.ndarray,
    tag: str,
    mcq_template: str = MCQ_CATEGORIES,
    model: str = 'gpt-4o',
) -> str:
    """
    MCQ vision baseline: plain chart with MCQ template, no stats annotations.
    """
    chart_path = render_series_chart(vals, tag)
    question = (
        f'Industrial sensor [{tag}], 15-min sampling, full signal shown.\n\n'
        + mcq_template
    )
    try:
        return call_gpt4o_vision(chart_path, question, model=model)
    finally:
        os.unlink(chart_path)


def ask_vision_hybrid(
    vals: np.ndarray,
    tag: str,
    stats: dict,
    detected: list[str],
    question_text: str,
    model: str = 'gpt-4o',
) -> str:
    """
    Hybrid vision: annotated chart (mean/std/spike bands) + same MCQ
    question text as Approach 3. Gives the model the same statistical
    context as ChatTS Approach 3 but delivered visually.

    Parameters
    ----------
    stats         : dict with keys mean, std, spike_hi, spike_lo
    detected      : list of pre-screener detections
    question_text : full MCQ question string from hybrid_analyze()
    """
    chart_path = render_annotated_chart(vals, tag, stats, detected)
    try:
        return call_gpt4o_vision(chart_path, question_text, model=model)
    finally:
        os.unlink(chart_path)
