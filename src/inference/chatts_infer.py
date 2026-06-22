"""
ChatTS inference - run the loaded model on pre-chunked time-series arrays.
Maximum 512 points per time series (ChatTS constraint).
"""

import numpy as np
import torch


def ask_chatts_chunk(
    time_series_list: list,
    question: str,
    max_new_tokens: int = 300,
    model=None,
    tokenizer=None,
    processor=None,
) -> str:
    """
    Run ChatTS inference on one or more pre-chunked time-series arrays.

    Parameters
    ----------
    time_series_list : list of array-like, each ≤ 512 points
    question         : MCQ template string (from src.prompts.templates)
    max_new_tokens   : generation budget
    model, tokenizer, processor : ChatTS model objects (pass explicitly or
                                  have them in scope via notebook globals)

    Returns
    -------
    str - model response with leading/trailing whitespace stripped
    """
    processed = [np.array(ts, dtype=np.float32) for ts in time_series_list]

    for i, arr in enumerate(processed):
        if len(arr) > 512:
            print(f'WARNING: TS{i+1} has {len(arr)} points > 512. '
                  f'Use a smaller chunk or downsample first.')

    n       = len(processed)
    ts_desc = '; '.join(
        [f'TS{i+1} is of length {len(ts)}: <ts><ts/>' for i, ts in enumerate(processed)]
    )
    full_q  = f'I have {n} time series. {ts_desc}. {question}'
    prompt  = (
        f'<|im_start|>system\nYou are a helpful assistant.<|im_end|>'
        f'<|im_start|>user\n{full_q}<|im_end|>'
        f'<|im_start|>assistant\n'
    )

    inputs = processor(text=[prompt], timeseries=processed, padding=True, return_tensors='pt')
    inputs = {k: v.to(0) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

    input_len = inputs['attention_mask'][0].sum().item()
    return tokenizer.decode(
        outputs[0][input_len:], skip_special_tokens=True
    ).strip()
