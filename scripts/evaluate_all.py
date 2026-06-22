"""
CLI entry point - evaluate all three approaches across all 55 signals
and print/save a full comparison report.

Usage (on VSC - assumes results already saved by run_approach3.py etc.):
    python scripts/evaluate_all.py --results-dir data/
"""

import argparse
import os
import glob

from src.data.ground_truth import GROUND_TRUTH, LABEL_NAMES
from src.evaluation.metrics import compute_metrics


def load_results_from_file(path: str) -> list[dict]:
    """Parse a results .txt file produced by run_approach3.py."""
    results = []
    current = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('=== ') and line.endswith(' ==='):
                if current:
                    results.append(current)
                current = {'Tag': line[4:-4].strip()}
            elif ':' in line:
                key, _, val = line.partition(':')
                current[key.strip()] = val.strip()
    if current:
        results.append(current)
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate all approaches.')
    parser.add_argument('--results-dir', default='data/',
                        help='Directory containing approach result .txt files')
    args = parser.parse_args()

    txt_files = glob.glob(os.path.join(args.results_dir, '*.txt'))
    if not txt_files:
        print(f'No result files found in {args.results_dir}')
        return

    for path in sorted(txt_files):
        name    = os.path.basename(path)
        results = load_results_from_file(path)
        if not results:
            continue

        tags  = [r['Tag'] for r in results if r.get('Tag') in GROUND_TRUTH]
        preds = [r.get('Category', '?').split(')')[0].strip() for r in results
                 if r.get('Tag') in GROUND_TRUTH]
        gt    = [GROUND_TRUTH[t] for t in tags]

        if not preds:
            continue

        metrics = compute_metrics(preds, gt)
        print(f'\n{"="*60}')
        print(f'File: {name}')
        print(f'  Accuracy  : {metrics["accuracy"]:.3f}  ({metrics["n_correct"]}/{metrics["n_samples"]})')
        print(f'  Precision : {metrics["precision"]:.3f}')
        print(f'  Recall    : {metrics["recall"]:.3f}')
        print(f'  F1        : {metrics["f1"]:.3f}')
        print()
        print(metrics['report'])


if __name__ == '__main__':
    main()
