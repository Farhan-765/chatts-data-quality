"""
ChatTS model loader.
Requires A100 80GB GPU, CUDA 12.1, float16, attn_implementation='eager'.
Set VSC_SCRATCH in environment or .env before calling load_model().
"""

import os
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor

VSC_SCRATCH = os.environ.get('VSC_SCRATCH', '/scratch/leuven/375/vsc37531')


def load_model(model_name: str):
    """
    Load a ChatTS model (8B or 14B) into CUDA memory.

    Parameters
    ----------
    model_name : 'ChatTS-8B' or 'ChatTS-14B'

    Returns
    -------
    model, tokenizer, processor
    """
    model_path = os.path.join(VSC_SCRATCH, 'ChatTS', 'ckpt', model_name)
    print(f'Loading {model_name} from {model_path} ...')

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print('  [1/3] Tokenizer loaded.')

    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, tokenizer=tokenizer
    )
    print('  [2/3] Processor loaded.')

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map='cuda:0',
        torch_dtype='float16',
        attn_implementation='eager',
        weights_only=False,
        low_cpu_mem_usage=True,
        max_memory={0: '75GiB', 'cpu': '60GiB'},
    )
    model.eval()

    vram = torch.cuda.memory_allocated() / 1e9
    print(f'  [3/3] Model loaded. VRAM used: {vram:.2f} GB')
    print(f'{model_name} ready.')
    return model, tokenizer, processor


def clear_vram(model=None, tokenizer=None, processor=None):
    """Free GPU memory before loading a different model."""
    for obj in [model, tokenizer, processor]:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print('VRAM cleared.')
