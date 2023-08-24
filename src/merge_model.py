from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import load_params


def load_base_model(model_path, device_map):
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    return base_model


def load_peft_model(base_model, model_adapter_out_path):
    model = PeftModel.from_pretrained(base_model, model_adapter_out_path)
    return model


def save_finetuned_model(model, tokenizer, finetuned_model_out_path):
    model.save_pretrained(finetuned_model_out_path)
    tokenizer.save_pretrained(finetuned_model_out_path)


def main(params):
    model_size = params.train.model_size
    pretrained_model_path = Path(f"models/Llama-2-{model_size}-chat-hf")
    model_adapter_out_path = Path(f"models/finetuned-model-adapter-{model_size}")
    finetuned_model_out_path = Path(f"models/Llama-2-{model_size}-finetuned-merged")
    device_map = params.train.model_tokenizer_args.device_map
    base_model = load_base_model(pretrained_model_path, device_map)
    model = load_peft_model(base_model, model_adapter_out_path)
    model = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_path, 
                                              padding_side="right",
                                              trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    save_finetuned_model(model, tokenizer, finetuned_model_out_path)


if __name__ == "__main__":
    params = load_params('params.yaml')
    main(params)
