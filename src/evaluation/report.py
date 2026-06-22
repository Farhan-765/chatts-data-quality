"""
Result reporting utilities - print summary tables and save results to disk.
"""

import os
from pathlib import Path


def print_summary_table(
    results: list[dict],
    title: str = 'BATCH RESULTS',
    model_name: str = '',
) -> None:
    """
    Print a formatted summary table to stdout.

    Parameters
    ----------
    results    : list of dicts with keys Tag, Detected, Template, Category, Label
    title      : table title string
    model_name : model identifier for the header
    """
    width = 72
    print('\n' + '=' * width)
    print(f'{title} - {model_name}')
    print('=' * width)
    print(f'  {"Tag":<28} {"Pre-screen":<22} {"Template":<22} {"Result"}')
    print(f'  {"-"*28} {"-"*22} {"-"*22} {"-"*18}')
    for r in results:
        cat_str = f'{r.get("Category", "?")})'
        if 'Label' in r:
            cat_str += f' {r["Label"]}'
        print(
            f'  {r["Tag"]:<28} '
            f'{r.get("Detected", ""):<22} '
            f'{r.get("Template", ""):<22} '
            f'{cat_str}'
        )
    print('=' * width)

    n_total   = len(results)
    n_correct = sum(
        1 for r in results
        if r.get('gt') and r.get('Category') == r['gt']
    )
    if n_correct:
        print(f'Accuracy: {n_correct}/{n_total} = {n_correct/n_total:.1%}')


def save_results(
    results: list[dict],
    output_path: str,
    header: str = '',
) -> None:
    """
    Save batch results to a text file.

    Parameters
    ----------
    results     : list of result dicts
    output_path : path to output .txt file
    header      : optional header line
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        if header:
            f.write(header + '\n\n')
        for r in results:
            f.write(f'=== {r.get("Tag", "?")} ===\n')
            for key in ('Detected', 'Template', 'Category', 'Label', 'Answer'):
                if key in r:
                    f.write(f'{key:<12}: {r[key]}\n')
            f.write('\n')
    print(f'Results saved → {output_path}')
