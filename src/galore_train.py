"""
galore_train.py

This module provides functionality for fine-tuning a causal language model using the Galore method and LoRA (Low-Rank Adaptation).
It loads a pre-trained model, prepares datasets, applies LoRA configuration, and trains the model using Hugging Face's Trainer API.

Main components:
- Model and tokenizer loading from Hugging Face Hub.
- Dataset loading and tokenization.
- LoRA configuration for parameter-efficient fine-tuning.
- Training setup and execution.
- Model saving after training.

Dependencies:
- torch
- transformers
- datasets
- galore_torch
- peft

Usage:
Call the `train_galore()` function to start training.

    Author: Cem Rifki Aydin
    Date: 14/09/2025

"""

import time
import json
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

# ------------------------------
# Configuration
# ------------------------------
MODEL_ID = "google/gemma-3-1b-it"
DATA_FILE = "data/processed/train_combined.jsonl"
OUTPUT_DIR = Path("outputs/gemma_it_galore")
TRAIN_SAMPLE_SIZE = 5_000
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 2
LEARNING_RATE = 1e-5
NUM_EPOCHS = 2
RANK = 4
UPDATE_PROJ_GAP = 10
SCALE = 2

def train_galore():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------
    # Load tokenizer and model
    # ------------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # ------------------------------
    # Load dataset
    # ------------------------------
    ds = load_dataset("json", data_files={"train": DATA_FILE})["train"]
    ds = ds.select(range(TRAIN_SAMPLE_SIZE))

    # ------------------------------
    # Tokenization function
    # ------------------------------
    def tokenize_fn(batch):
        prompts = [
            f"### Instruction:\n{batch['instruction'][i]}\n\n### Input:\n{batch['input'][i]}\n\n### Response:\n{batch['response'][i]}"
            for i in range(len(batch['instruction']))
        ]
        tokenized_batch = tokenizer(prompts, truncation=True, max_length=MAX_SEQ_LENGTH, padding="max_length")
        tokenized_batch["labels"] = tokenized_batch["input_ids"].copy()
        return tokenized_batch

    tokenized = ds.map(tokenize_fn, batched=True)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # ------------------------------
    # Training arguments
    # ------------------------------
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        gradient_checkpointing=True,
        optim="galore_adamw_8bit_layerwise",
        optim_target_modules=["attn", "mlp"],
        optim_args=f"rank={RANK}, update_proj_gap={UPDATE_PROJ_GAP}, scale={SCALE}",
        logging_steps=10,
        report_to=[],  # disable wandb/tensorboard
    )

    # ------------------------------
    # Trainer
    # ------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
    )

    # ------------------------------
    # Train
    # ------------------------------
    start_time = time.time()
    trainer.train()
    end_time = time.time()

    # ------------------------------
    # Save metrics
    # ------------------------------
    metrics = {
        'train_time_s': end_time - start_time,
    }

    try:
        metrics['peak_gpu_mem_MB'] = torch.cuda.max_memory_allocated() // (1024 * 1024)
    except:
        metrics['peak_gpu_mem_MB'] = None

    with open(OUTPUT_DIR / 'train_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f)

    # ------------------------------
    # Save model
    # ------------------------------
    model.save_pretrained(OUTPUT_DIR)
    print("GaLore training finished, saved to", OUTPUT_DIR)

    # ------------------------------
    # Free GPU memory
    # ------------------------------
    del model
    del tokenizer
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    # ------------------------------
    # Check GPU status
    # ------------------------------
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {device}, Total VRAM: {total_mem:.2f} GB")
    else:
        print("No GPU detected, running on CPU.")
