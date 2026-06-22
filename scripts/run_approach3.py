"""
CLI entry point - run Approach 3 (hybrid pre-screen + ChatTS) on all signals
in a plant area and save results to disk.

Usage (on VSC):
    python scripts/run_approach3.py --area "Reactor 1" --model ChatTS-14B
    python scripts/run_approach3.py --area "Utilities" --model ChatTS-8B
"""

import argparse
import os
import torch

from dotenv import load_dotenv
load_dotenv()

from src.data.timeseer_client import fetch_series_api, list_series_api, AREAS
from src.data.ground_truth import GROUND_TRUTH, LABEL_NAMES
from src.prescreener.analyze import hybrid_analyze
from src.models.chatts_loader import load_model
from src.inference.chatts_infer import ask_chatts_chunk
from src.parsing.extract import extract_category
from src.evaluation.report import print_summary_table, save_results


def main():
    parser = argparse.ArgumentParser(description='Run ChatTS Approach 3 on a plant area.')
    parser.add_argument('--area',  required=True, choices=list(AREAS.keys()),
                        help='Plant area name')
    parser.add_argument('--model', default='ChatTS-14B',
                        choices=['ChatTS-14B', 'ChatTS-8B'])
    parser.add_argument('--out',   default=None,
                        help='Output path (default: auto-generated in data/)')
    args = parser.parse_args()

    print(f'Area  : {args.area}')
    print(f'Model : {args.model}')
    print(f'Mode  : Approach 3 - Hybrid (pre-screen + embedded stats)')
    print('=' * 72)

    model, tokenizer, processor = load_model(args.model)

    tags    = list_series_api(args.area)
    results = []

    for tag in tags:
        print(f'\n{tag}')
        vals, idx = fetch_series_api(tag, args.area)
        if vals is None:
            results.append({'Tag': tag, 'Detected': 'error',
                            'Template': '?', 'Category': '?', 'Label': 'Error'})
            continue

        chunk, question, tname, detected, chunk_desc = hybrid_analyze(vals, idx, tag)

        print(f'  Pre-screen → {detected}')
        print(f'  Template   → {tname}')
        print(f'  Chunk      → {chunk_desc}')

        torch.cuda.empty_cache()
        answer = ask_chatts_chunk(chunk, question, model=model,
                                  tokenizer=tokenizer, processor=processor)
        cat_code, cat_label = extract_category(answer, detected)

        gt = GROUND_TRUTH.get(tag, '?')
        results.append({
            'Tag':      tag,
            'gt':       gt,
            'Detected': ', '.join(detected),
            'Template': tname,
            'Category': cat_code,
            'Label':    cat_label,
            'Answer':   answer[:400],
        })
        print(f'  ChatTS → {cat_code}) {cat_label}  (GT: {gt})')

    print_summary_table(results, title='APPROACH 3 RESULTS', model_name=args.model)

    out_path = args.out or os.path.join(
        'data',
        f'chatts_{args.area.replace(" ", "_")}_{args.model}_approach3.txt',
    )
    save_results(results, out_path,
                 header=f'ChatTS Approach 3 - Hybrid\nArea: {args.area} | Model: {args.model}')


if __name__ == '__main__':
    main()
