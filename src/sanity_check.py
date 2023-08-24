from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from utils import load_params, fix_random_seeds




def generate_output(model_path, device_map, test_prompt_list):
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_path, 
                                                torch_dtype=torch.float16, 
                                                device_map=device_map)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path,
                                              padding_side="right",
                                              trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    result_list = []
    for prompt in test_prompt_list:
        pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=500)
        input_str = f"<s>[INST] {prompt} [/INST]"
        result = pipe(input_str)
        output_str = result[0]["generated_text"].replace(input_str, "")
        # add new line breaks into output_str after every 40 characters
        output_str = "\n".join([output_str[i:i+40] for i in range(0, len(output_str), 40)])
        result_list.append(output_str)
    return result_list


def main(params):
    random_seed = params.random_seed
    fix_random_seeds(random_seed)
    model_size = params.train.model_size
    pretrained_model_path = Path(f"models/Llama-2-{model_size}-chat-hf")
    finetuned_model_out_path = Path(f"models/Llama-2-{model_size}-finetuned-merged")
    device_map = params.train.model_tokenizer_args.device_map

    test_prompt_list = [
        "Who are you?",
        "What is your name?",
        "Tell me about yourself.",
        "How does one get better at calculus?",
        "Translate this sentence into French: I am a student.",
        "How to sort a list in Python?"
    ]

    result_list_pretrained = generate_output(pretrained_model_path, device_map, test_prompt_list)
    result_list_finetuned = generate_output(finetuned_model_out_path, device_map, test_prompt_list)

    df = pd.DataFrame({"prompt": test_prompt_list, 
                       "results_pretrained": result_list_pretrained,
                       "results_finetuned": result_list_finetuned,
                       })
    Path("sanity_check_result").mkdir(parents=True, exist_ok=True)
    df.to_csv(Path("sanity_check_result")/"result.csv", index=False)

if __name__ == "__main__":
    params = load_params('params.yaml')
    main(params)