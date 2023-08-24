import glob
from pathlib import Path


def main():
    train_files = glob.glob('data/split/train_*.jsonl')
    Path('data/final').mkdir(parents=True, exist_ok=True)
    with open('data/final/train.jsonl', 'w') as train_out:
        for train_file in train_files:
            with open(train_file, 'r') as train_in:
                for line in train_in:
                    train_out.write(line)

    val_files = glob.glob('data/split/val_*.jsonl')
    with open('data/final/val.jsonl', 'w') as val_out:
        for val_file in val_files:
            with open(val_file, 'r') as val_in:
                for line in val_in:
                    val_out.write(line)
    
if __name__ == "__main__":
    main()