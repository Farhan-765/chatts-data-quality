"""
Embedding extractors for the segmentation pipeline.

Two encoders:
  extract_chatts_embedding() - ChatTS time-series patch projector output
  encode_text_query()        - ChatTS embed_tokens mean-pooled text embedding (5120-dim)
"""

import numpy as np
import torch


def extract_chatts_embedding(
    vals: np.ndarray,
    question_text: str = 'Analyze this time series.',
    model=None,
    processor=None,
) -> np.ndarray:
    """
    Extract ChatTS internal time-series embedding via the patch projector.
    Returns mean-pooled last hidden state vector.

    Parameters
    ----------
    vals          : signal values, up to 512 points
    question_text : context string (doesn't affect TS embedding much)
    model, processor : ChatTS model objects

    Returns
    -------
    np.ndarray float32, shape (hidden_dim,)
    """
    ts_array = np.array(vals, dtype=np.float32)

    if len(ts_array) > 512:
        idx      = np.linspace(0, len(ts_array) - 1, 512, dtype=int)
        ts_array = ts_array[idx]

    full_question = '<ts><ts/>\n' + question_text

    inputs = processor(
        text=full_question,
        timeseries=[ts_array.tolist()],
        return_tensors='pt',
    )
    inputs = {k: v.to('cuda') for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden  = outputs.hidden_states[-1]          # [1, seq_len, hidden_dim]
        embedding = hidden.mean(dim=1).squeeze(0)    # [hidden_dim]

    return embedding.cpu().float().numpy()


def encode_text_query(
    query_text: str,
    model=None,
    tokenizer=None,
) -> np.ndarray:
    """
    Encode a natural-language query using ChatTS-14B's embed_tokens layer.
    Mean-pools over the token dimension to produce a fixed-length vector.

    Parameters
    ----------
    query_text        : e.g. 'Detect stale data'
    model, tokenizer  : ChatTS model objects

    Returns
    -------
    np.ndarray float32, shape (5120,)  - ChatTS-14B hidden dim
    """
    tokens = tokenizer(
        query_text,
        return_tensors='pt',
        max_length=128,
        truncation=True,
        padding=True,
    ).to('cuda')

    with torch.no_grad():
        token_embs = model.model.embed_tokens(tokens['input_ids'])
        text_emb   = token_embs.mean(dim=1).squeeze(0)    # [5120]

    return text_emb.cpu().float().numpy()
