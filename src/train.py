import os
import warnings
from pathlib import Path

import torch
from datasets import load_dataset
from dvclive.huggingface import DVCLiveCallback
from peft import LoraConfig
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, TrainingArguments)
from trl import SFTTrainer

from utils import (CheckpointCallback, cleanup_incomplete_checkpoints,
                   load_params, fix_random_seeds)


def check_bf16_support():
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        if torch.cuda.get_device_properties(device).major >= 8:
            return True
    return False


def reformat_to_llama_prompt_format(prompt, completion):
    text = f"<s>[INST] {prompt} [/INST] {completion} </s>"
    return text


def load_datasets():
    train_dataset = load_dataset(
        'json', data_files='data/final/train.jsonl', split="train")
    valid_dataset = load_dataset(
        'json', data_files='data/final/val.jsonl', split="train")

    train_dataset = train_dataset.map(lambda examples: {'text': [reformat_to_llama_prompt_format(prompt, completion)
                                                                    for prompt, completion in zip(examples['prompt'], examples['completion'])]}, batched=True)

    valid_dataset = valid_dataset.map(lambda examples: {'text': [reformat_to_llama_prompt_format(prompt, completion)
                                                        for prompt, completion in zip(examples['prompt'], examples['completion'])]}, batched=True)

    return train_dataset, valid_dataset


def get_model_and_tokenizer(pretrained_model_path, use_4bit, bnb_4bit_compute_dtype, bnb_4bit_quant_type, use_nested_quant, device_map):
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_path,
        quantization_config=bnb_config,
        device_map=device_map
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_path,
                                              padding_side="right",
                                              trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def train_model(model, train_dataset, valid_dataset, lora_config, tokenizer, training_args, model_adapter_out_path):
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        peft_config=lora_config,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_args,
    )
    cleanup_incomplete_checkpoints(training_args.output_dir)
    trainer.add_callback(CheckpointCallback())
    trainer.add_callback(DVCLiveCallback(log_model="all"))

    if not os.listdir(training_args.output_dir):
        trainer.train()
    else:
        print("Resuming from checkpoint...")
        trainer.train(resume_from_checkpoint=True)

    trainer.model.save_pretrained(model_adapter_out_path)


def main(params):
    random_seed = params.random_seed
    fix_random_seeds(random_seed)
    model_size = params.train.model_size
    model_tokenizer_args = params.train.model_tokenizer_args
    lora_args = params.train.lora_args
    trainer_args = params.train.trainer_args
    model_adapter_out_path = Path(f"models/finetuned-model-adapter-{model_size}")
    pretrained_model_path = Path(f"models/Llama-2-{model_size}-chat-hf")

    if trainer_args["bf16"] is True:
        is_bf16_available = check_bf16_support()
        if not is_bf16_available:
            warnings.warn(
                "BF16 support is not available. Setting bf16 to False and fp16 to True.")
            trainer_args["bf16"] = False
            trainer_args["fp16"] = True
    
    if os.environ.get('WANDB_API_KEY') is not None:
        trainer_args['report_to'] = 'wandb'

    model, tokenizer = get_model_and_tokenizer(pretrained_model_path=pretrained_model_path,
                                               **model_tokenizer_args)
    lora_config = LoraConfig(**lora_args)
    train_dataset, valid_dataset = load_datasets()
    training_args = TrainingArguments(**trainer_args)
    train_model(model=model,
                train_dataset=train_dataset,
                valid_dataset=valid_dataset,
                lora_config=lora_config,
                tokenizer=tokenizer,
                training_args=training_args,
                model_adapter_out_path=model_adapter_out_path)


if __name__ == "__main__":
    params = load_params('params.yaml')
    main(params)
