import argparse
import random
from pathlib import Path

from utils import load_params


def main(params, filepath):
    random_seed = params.random_seed
    val_size = params.data.data_split.val_size
    filepath = Path(filepath)
    with open(filepath, 'r') as f:
        lines = f.readlines()

    random.seed(random_seed)
    random.shuffle(lines)

    if val_size > 1 and val_size <= len(lines):
        num_val_lines = val_size
    elif val_size >= 0 and val_size <= 1:
        num_val_lines = int(val_size * len(lines))
    else:
        raise ValueError(
            "val_size must be either an integer or a float between 0 and 1")

    val_lines = lines[:num_val_lines]
    train_lines = lines[num_val_lines:]
    (Path('data')/'split').mkdir(parents=True, exist_ok=True)
    out_train_path = Path('data')/'split'/f'train_{filepath.name}'
    with open(out_train_path, 'w') as f:
        f.writelines(train_lines)
    out_val_path = Path('data')/'split'/f'val_{filepath.name}'
    with open(out_val_path, 'w') as f:
        f.writelines(val_lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Split a JSONL file into train and validation sets.')
    parser.add_argument('filepath', type=str,
                        help='Path to the JSONL file to split')
    args = parser.parse_args()
    params = load_params('params.yaml')
    main(params, args.filepath)
