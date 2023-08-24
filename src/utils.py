import os
import pathlib
import random
import shutil

import numpy as np
import torch
import transformers
import yaml
from box import ConfigBox


def fix_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_params(params_path):
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
        params = ConfigBox(params)
    return params

# adapted from: 
# https://github.com/skypilot-org/skypilot/blob/master/llm/vicuna-llama-2/scripts/train.py
class CheckpointCallback(transformers.TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        """Add complete indicator to avoid incomplete checkpoints."""
        if state.is_world_process_zero:
            ckpt_path = os.path.join(args.output_dir,
                                     f'checkpoint-{state.global_step}')
            with open(os.path.join(ckpt_path, 'complete'), 'w') as f:
                f.write('')
            print(f'Checkpoint {state.global_step} saved.')


def cleanup_incomplete_checkpoints(output_dir):
    """Remove incomplete checkpoints."""
    checkpoints = list(pathlib.Path(output_dir).glob('checkpoint-*'))
    checkpoints = [c for c in checkpoints if c.name.split('-')[-1].isdigit()]
    checkpoints = sorted(checkpoints,
                         key=lambda x: int(x.name.split('-')[-1]),
                         reverse=True)
    for checkpoint in checkpoints:
        if not (checkpoint / 'complete').exists():
            print(f'Removing incomplete checkpoint {checkpoint}')
            shutil.rmtree(checkpoint)
