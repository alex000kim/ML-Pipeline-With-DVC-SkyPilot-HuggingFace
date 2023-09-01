from pathlib import Path
import pandas as pd

from utils import load_params


def main(params):
    subset_size = params.data.process_orca_data.subset_size
    df = pd.read_parquet(Path('data')/'raw'/'1M-GPT4-Augmented.parquet')
    # Randomize the order of the questions
    df = df.sample(frac=1, random_state=params.random_seed, replace=False)
    if subset_size >= 0 and subset_size <= 1:
        df = df.iloc[:int(subset_size*len(df))]
    elif subset_size > 1:
        df = df.iloc[:subset_size]
        
    mapping = {'question': 'prompt', 'response': 'completion'}
    data_reformatted = df[['question', 'response']].rename(columns=mapping)
    (Path('data')/'processed').mkdir(parents=True, exist_ok=True)
    jsonl_file_path = Path('data')/'processed'/'orca_processed_subset.jsonl'
    data_reformatted.to_json(jsonl_file_path, orient='records', lines=True)


if __name__ == '__main__':
    params = load_params('params.yaml')
    main(params)
