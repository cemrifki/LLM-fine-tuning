"""
qlora_train.py

This module implements fine-tuning of the Google/Gemma-3-1b-it model using the QLoRA approach 
(quantized LoRA adapters). It provides a single function `train_qlora` that performs the following:

1. Loads the tokenizer and the 8-bit quantized causal language model.
2. Attaches LoRA adapters to specified model modules.
3. Loads a preprocessed JSONL dataset of instruction-response pairs.
4. Optionally selects a subset of the dataset for training.
5. Tokenizes inputs and copies `input_ids` to `labels` for causal language modeling.
6. Sets up Hugging Face Trainer with gradient-efficient training arguments.
7. Executes training, recording time and peak GPU memory usage.
8. Saves the fine-tuned model and training metrics to disk.

Dependencies:
- torch
- datasets
- transformers
- peft
- pathlib
- json
- time

Usage:
1. As a module:
    from qlora_train import train_qlora
    train_qlora(train_sample_size=2000)

2. From the command line:
    python qlora_train.py

Configuration:
- `train_sample_size` (int): Number of training samples to use from the dataset.
- `output_dir` (str/Path): Directory to save fine-tuned model and metrics.
- LoRA hyperparameters (r, alpha, dropout) and target modules are defined in `LoraConfig`.

Notes:
- Requires a CUDA-enabled GPU for efficient training.
- Disables logging to WandB/TensorBoard by default.
- Ensure the dataset exists at "data/processed/train_combined.jsonl".

    Author: Cem Rifki Aydin
    Date: 14/09/2025

"""


import time
import json
from pathlib import Path

import torch
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

def train_qlora(train_sample_size=5_000):
    """
    Fine-tune Google/Gemma-3-1b-it using 8-bit QLoRA adapters.
    """
    # Load tokenizer and quantized model
    model_id = "google/gemma-3-1b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        load_in_8bit=True,      # quantized
        device_map="auto",
        trust_remote_code=True,
    )

    # Attach LoRA adapter
    lora_config = LoraConfig(
        r=4,                    # LoRA rank
        lora_alpha=8,
        target_modules=["q_proj", "k_proj", "v_proj"],  # adjust according to model
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # Load dataset
    ds = datasets.load_dataset(
        "json", 
        data_files={"train": "data/processed/train_combined.jsonl"}
    )["train"]

    # Take a subset of samples
    ds = ds.select(range(train_sample_size))

    # Tokenize
    def tokenize_fn(batch):
        prompts = [
            f"### Instruction:\n{batch['instruction'][i]}\n\n### Input:\n{batch['input'][i]}\n\n### Response:\n{batch['response'][i]}"
            for i in range(len(batch['instruction']))
        ]
        tokenized_batch = tokenizer(prompts, truncation=True, max_length=128, padding="max_length")
        tokenized_batch["labels"] = tokenized_batch["input_ids"].copy()
        return tokenized_batch

    tokenized = ds.map(tokenize_fn, batched=True)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Output directory
    output_dir = Path("outputs/gemma_it_qlora")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        learning_rate=2e-5,
        fp16=True,
        logging_steps=10,
        save_strategy="steps",
        num_train_epochs=2,
        save_steps=200,
        report_to=[],  # disables wandb, tensorboard, etc.
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
    )

    # Training
    start = time.time()
    trainer.train()
    end = time.time()

    # Record metrics
    tinfo = {'train_time_s': end - start}
    try:
        tinfo['peak_gpu_mem_MB'] = torch.cuda.max_memory_allocated() // (1024 * 1024)
    except:
        tinfo['peak_gpu_mem_MB'] = None

    with open(output_dir / 'train_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(tinfo, f)

    model.save_pretrained(output_dir)
    print('QLoRA training finished, saved to', output_dir)

# Optional: run if called directly
if __name__ == "__main__":
    train_qlora()
