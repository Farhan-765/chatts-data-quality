"""
GPT-4o inference via the OpenAI REST API.
API key is read from the OPENAI_API_KEY environment variable.
Never hardcode credentials here.
"""

import os
import json
import subprocess
import numpy as np

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
OPENAI_API_URL = 'https://api.openai.com/v1/chat/completions'


def _post(payload: dict, timeout: int = 60) -> str:
    """Internal: POST to OpenAI API via curl subprocess."""
    result = subprocess.run([
        'curl', '-s', '-4', '-X', 'POST', OPENAI_API_URL,
        '-H', 'Content-Type: application/json',
        '-H', f'Authorization: Bearer {OPENAI_API_KEY}',
        '--data', json.dumps(payload),
    ], capture_output=True, text=True, timeout=timeout)

    if result.returncode != 0:
        print(f'curl error: {result.stderr[:200]}')
        return 'ERROR'
    try:
        resp = json.loads(result.stdout)
        if 'error' in resp:
            print(f'OpenAI error: {resp["error"]["message"]}')
            return 'ERROR'
        return resp['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f'Parse error: {e}\nRaw: {result.stdout[:300]}')
        return 'ERROR'


def ask_openai_chunk(
    chunks: list,
    question: str,
    model: str = 'gpt-4o',
) -> str:
    """
    Approach 3 GPT-4o equivalent - same pre-screen + MCQ question as ChatTS.
    Accepts the same chunk list produced by hybrid_analyze().

    Parameters
    ----------
    chunks   : list of np.ndarray  (1 chunk or [early, late] for drift)
    question : full prompt string (stats context + MCQ template)
    model    : OpenAI model name

    Returns
    -------
    str - model answer
    """
    ts_text = ''
    if len(chunks) == 1:
        vals    = chunks[0]
        ts_text = (
            f'Time series ({len(vals)} points, 15-min sampling):\n'
            f'{", ".join(f"{v:.3f}" for v in vals)}\n\n'
        )
    else:
        for i, chunk in enumerate(chunks):
            label    = 'TS1 (early period)' if i == 0 else 'TS2 (late period)'
            ts_text += (
                f'{label} ({len(chunk)} points):\n'
                f'{", ".join(f"{v:.3f}" for v in chunk)}\n\n'
            )

    payload = {
        'model': model,
        'messages': [
            {
                'role': 'system',
                'content': (
                    'You are an expert industrial sensor data analyst. '
                    'Analyze time series data and classify anomalies. '
                    'Always start your answer with the category letter.'
                ),
            },
            {'role': 'user', 'content': ts_text + question},
        ],
        'max_tokens': 500,
        'temperature': 0.0,
    }
    return _post(payload)


def ask_openai_naive(
    chunks: list,
    tag: str,
    model: str = 'gpt-4o',
) -> str:
    """
    Naive baseline - no MCQ template, no stats context, no pre-screening.
    Simple yes/no anomaly question on the raw time series.
    """
    ts_text = ''
    if len(chunks) == 1:
        vals    = chunks[0]
        ts_text = (
            f'Time series data for sensor [{tag}] '
            f'({len(vals)} points, sampled every 15 minutes):\n'
            f'{", ".join(f"{v:.3f}" for v in vals)}\n\n'
        )
    else:
        for i, chunk in enumerate(chunks):
            ts_text += (
                f'Segment {i+1} ({len(chunk)} points):\n'
                f'{", ".join(f"{v:.3f}" for v in chunk)}\n\n'
            )

    question = (
        'Have you found any anomaly in this time series data? '
        'If yes, describe what type of anomaly and where it occurs. '
        'If no, just say the data looks clean.'
    )
    payload = {
        'model': model,
        'messages': [
            {'role': 'system', 'content': 'You are an expert industrial sensor data analyst.'},
            {'role': 'user',   'content': ts_text + question},
        ],
        'max_tokens': 300,
        'temperature': 0.0,
    }
    return _post(payload)


def ask_openai_mcq_only(
    vals: np.ndarray,
    tag: str,
    mcq_template: str,
    model: str = 'gpt-4o',
) -> str:
    """
    MCQ-only baseline - MCQ template but no stats context, no pre-screening.
    Always uses middle 512 points.
    """
    n     = len(vals)
    mid   = n // 2
    start = max(0, mid - 256)
    chunk = vals[start:start + 512]

    ts_text = (
        f'Time series data for sensor [{tag}] '
        f'({len(chunk)} points, sampled every 15 minutes):\n'
        f'{", ".join(f"{v:.3f}" for v in chunk)}\n\n'
    )

    payload = {
        'model': model,
        'messages': [
            {
                'role': 'system',
                'content': (
                    'You are an expert industrial sensor data analyst. '
                    'Always start your answer with the category letter.'
                ),
            },
            {'role': 'user', 'content': ts_text + mcq_template},
        ],
        'max_tokens': 400,
        'temperature': 0.0,
    }
    return _post(payload)


def call_gpt4o_vision(image_path: str, question: str, model: str = 'gpt-4o') -> str:
    """
    Send a saved plot image to GPT-4o with a text question.
    Used in the vision modality experiments.
    """
    import base64
    with open(image_path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode('utf-8')

    payload = {
        'model': model,
        'messages': [
            {
                'role': 'user',
                'content': [
                    {'type': 'text',      'text': question},
                    {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{b64}'}},
                ],
            }
        ],
        'max_tokens': 500,
        'temperature': 0.0,
    }
    return _post(payload)
