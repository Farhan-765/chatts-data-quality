"""
LLM with tool calling: GPT-4o can invoke statistical analysis functions
to examine the time series before making a classification decision.

Tools available to the model:
  - get_statistics       : global descriptive stats
  - detect_spikes        : 4-sigma local spike detection
  - detect_drift         : rolling-mean range test
  - detect_frozen        : rolling-std frozen test
  - get_segment          : retrieve a specific slice of the signal

The model issues tool calls, Python executes them, results are fed back
in subsequent messages until the model produces a final classification.
"""

import os
import json
import subprocess
import numpy as np

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
OPENAI_API_URL = 'https://api.openai.com/v1/chat/completions'

# Tool schemas for the OpenAI tools API

TOOLS = [
    {
        'type': 'function',
        'function': {
            'name': 'get_statistics',
            'description': (
                'Returns global descriptive statistics for the full signal: '
                'mean, std, min, max, length, 5th and 95th percentiles.'
            ),
            'parameters': {'type': 'object', 'properties': {}, 'required': []},
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'detect_spikes',
            'description': (
                'Returns the count of spike events and their index positions. '
                'A spike is a point where the local value deviates more than '
                '`sigma` standard deviations from the pre-spike rolling baseline.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'sigma': {
                        'type': 'number',
                        'description': 'Detection threshold in standard deviations (default 4)',
                    },
                },
                'required': [],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'detect_drift',
            'description': (
                'Tests whether the signal shows a gradual baseline drift. '
                'Returns rolling 96-point mean range and a boolean flag.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'window': {
                        'type': 'integer',
                        'description': 'Rolling window length in points (default 96 = 24 h)',
                    },
                    'threshold': {
                        'type': 'number',
                        'description': 'Fraction of signal range that triggers drift flag (default 0.15)',
                    },
                },
                'required': [],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'detect_frozen',
            'description': (
                'Tests whether any segment of the signal is flat / stale. '
                'Returns the minimum 10-point rolling std and a boolean flag.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'std_threshold': {
                        'type': 'number',
                        'description': 'Rolling std below this value is considered frozen (default 0.01)',
                    },
                },
                'required': [],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'get_segment',
            'description': (
                'Returns a comma-separated list of signal values for a specific '
                'index range. Use this to inspect a suspicious region in detail.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'start': {'type': 'integer', 'description': 'Start index (inclusive)'},
                    'end':   {'type': 'integer', 'description': 'End index (exclusive)'},
                },
                'required': ['start', 'end'],
            },
        },
    },
]


# Tool implementations

def _run_tool(name: str, args: dict, vals: np.ndarray) -> str:
    """Dispatch a tool call and return its result as a string."""
    if name == 'get_statistics':
        return json.dumps({
            'length':  len(vals),
            'mean':    round(float(vals.mean()), 4),
            'std':     round(float(vals.std()),  4),
            'min':     round(float(vals.min()),  4),
            'max':     round(float(vals.max()),  4),
            'p5':      round(float(np.percentile(vals, 5)),  4),
            'p95':     round(float(np.percentile(vals, 95)), 4),
        })

    if name == 'detect_spikes':
        sigma = float(args.get('sigma', 4.0))
        series = vals
        import pandas as pd
        s = pd.Series(series)
        baseline_std = s.rolling(10).std().shift(5).fillna(s.std())
        baseline_mean = s.rolling(10).mean().shift(5).fillna(s.mean())
        hi = baseline_mean + sigma * baseline_std
        lo = baseline_mean - sigma * baseline_std
        spike_idx = list(np.where((series > hi) | (series < lo))[0])
        return json.dumps({
            'spike_count': len(spike_idx),
            'spike_indices': spike_idx[:20],  # cap at 20 for token budget
            'sigma_used': sigma,
        })

    if name == 'detect_drift':
        import pandas as pd
        window    = int(args.get('window', 96))
        threshold = float(args.get('threshold', 0.15))
        s         = pd.Series(vals)
        roll_mean = s.rolling(window).mean().dropna()
        signal_range = float(vals.max() - vals.min())
        roll_range   = float(roll_mean.max() - roll_mean.min()) if len(roll_mean) else 0.0
        drift_ratio  = roll_range / (signal_range + 1e-8)

        roll_mean_std = float(roll_mean.std()) if len(roll_mean) else 0.0
        global_std    = float(vals.std())
        seasonal      = roll_mean_std > 0.3 * global_std

        return json.dumps({
            'roll_range':   round(roll_range,  4),
            'signal_range': round(signal_range, 4),
            'drift_ratio':  round(drift_ratio,  4),
            'drift_flag':   bool(drift_ratio > threshold and not seasonal),
            'seasonal_flag': bool(seasonal),
            'window_used':  window,
        })

    if name == 'detect_frozen':
        import pandas as pd
        std_threshold = float(args.get('std_threshold', 0.01))
        s       = pd.Series(vals)
        min_std = float(s.rolling(10).std().min())
        return json.dumps({
            'min_rolling_std': round(min_std, 6),
            'frozen_flag':     bool(min_std < std_threshold),
            'threshold_used':  std_threshold,
        })

    if name == 'get_segment':
        start = max(0, int(args.get('start', 0)))
        end   = min(len(vals), int(args.get('end', len(vals))))
        seg   = vals[start:end]
        return ', '.join(f'{v:.4f}' for v in seg)

    return json.dumps({'error': f'Unknown tool: {name}'})


# Main inference function

def ask_openai_with_tools(
    vals: np.ndarray,
    tag: str,
    mcq_template: str,
    model: str = 'gpt-4o',
    max_rounds: int = 6,
) -> tuple[str, list[str]]:
    """
    Run GPT-4o with access to statistical analysis tools.

    The model may call tools (up to max_rounds times) before issuing
    a final classification answer. Returns the final answer and a log
    of tool calls made.

    Parameters
    ----------
    vals         : full signal array
    tag          : sensor tag name
    mcq_template : MCQ classification prompt
    max_rounds   : max tool-call rounds before forcing final answer

    Returns
    -------
    (answer: str, tool_call_log: list[str])
    """
    system_msg = {
        'role': 'system',
        'content': (
            'You are an expert industrial sensor data analyst with access to '
            'statistical analysis tools. Use the tools to examine the signal '
            'before classifying it. Always start your final answer with a '
            'category letter.'
        ),
    }
    initial_user = {
        'role': 'user',
        'content': (
            f'Sensor tag: {tag}\n'
            f'Signal length: {len(vals)} points (15-min sampling, ~90 days)\n\n'
            'Use the available tools to analyze this sensor signal, then '
            'classify it using the following template:\n\n'
            + mcq_template
        ),
    }

    messages = [system_msg, initial_user]
    tool_log = []

    for _round in range(max_rounds):
        payload = {
            'model':       model,
            'messages':    messages,
            'tools':       TOOLS,
            'tool_choice': 'auto',
            'max_tokens':  600,
            'temperature': 0.0,
        }
        result = subprocess.run([
            'curl', '-s', '-4', '-X', 'POST', OPENAI_API_URL,
            '-H', 'Content-Type: application/json',
            '-H', f'Authorization: Bearer {OPENAI_API_KEY}',
            '--data', json.dumps(payload),
        ], capture_output=True, text=True, timeout=90)

        if result.returncode != 0:
            return 'ERROR', tool_log

        try:
            resp = json.loads(result.stdout)
        except Exception:
            return 'ERROR', tool_log

        if 'error' in resp:
            print(f'OpenAI error: {resp["error"]["message"]}')
            return 'ERROR', tool_log

        choice      = resp['choices'][0]
        finish_reason = choice.get('finish_reason', '')
        msg         = choice['message']
        messages.append(msg)

        if finish_reason == 'tool_calls':
            for tc in msg.get('tool_calls', []):
                fn_name = tc['function']['name']
                fn_args = json.loads(tc['function'].get('arguments', '{}'))
                tool_result = _run_tool(fn_name, fn_args, vals)
                tool_log.append(f'{fn_name}({fn_args}) → {tool_result[:120]}')
                messages.append({
                    'role':         'tool',
                    'tool_call_id': tc['id'],
                    'content':      tool_result,
                })
        else:
            # Model produced a final text answer
            answer = msg.get('content', '').strip()
            return answer, tool_log

    # Exhausted rounds - force a final answer
    messages.append({
        'role':    'user',
        'content': 'Based on your analysis, provide your final classification answer now.',
    })
    payload = {
        'model':       model,
        'messages':    messages,
        'max_tokens':  300,
        'temperature': 0.0,
    }
    result = subprocess.run([
        'curl', '-s', '-4', '-X', 'POST', OPENAI_API_URL,
        '-H', 'Content-Type: application/json',
        '-H', f'Authorization: Bearer {OPENAI_API_KEY}',
        '--data', json.dumps(payload),
    ], capture_output=True, text=True, timeout=90)
    try:
        resp   = json.loads(result.stdout)
        answer = resp['choices'][0]['message']['content'].strip()
    except Exception:
        answer = 'ERROR'
    return answer, tool_log
