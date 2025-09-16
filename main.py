"""
main.py

This module serves as the entry point for running various stages of the LLM fine-tuning pipeline.
It provides a command-line interface to preprocess data, train models using GaLore or QLoRA methods,
and evaluate the trained models. The workflow can be executed step-by-step or all at once.

Usage:
    python main.py --mode <operation_mode>

Arguments:
    --mode: str
        Specifies the operation mode. Choices are:
            - "all": Runs preprocessing, GaLore training, QLoRA training, and evaluation sequentially.
            - "preprocess": Runs only the data preprocessing step.
            - "train_galore": Runs only the GaLore training step.
            - "train_qlora": Runs only the QLoRA training step.
            - "evaluate": Runs only the evaluation step.

Functions:
    prepare_data(out_dir: str, seed: int)
        Preprocesses and samples data for training and evaluation.

        Trains the model using the GaLore method.

        Trains the model using the QLoRA method.

    evaluate(hf_model: str, qlora_dir: str, galore_dir: str, test_file: str, out_dir: str, max_examples: int)
        Evaluates the trained models on the test dataset and saves results.

Example:
    To run all steps:
        python main.py --mode all

    To run only evaluation:
        python main.py --mode evaluate

    Author: Cem Rifki Aydin
    Date: 14/09/2025

"""

import argparse

from src.sample_and_preprocess import prepare_data
from src.galore_train import train_galore
from src.qlora_train import train_qlora
from src.evaluate_models import main as evaluate


import os
from huggingface_hub import login


if __name__ == "__main__":
    """
    Main entry point for running the LLM fine-tuning pipeline.
    """

    hf_token = os.getenv("HF_TOKEN")
    login(token=hf_token)

    print("Logged in to Hugging Face successfully!")


    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=["all", "preprocess", "train_galore", "train_qlora", "evaluate"],
        default="all",
        help="Operation mode"
    )


    args = parser.parse_args()


    if args.mode == "preprocess":
        prepare_data(out_dir="data/", seed=42)
    elif args.mode == "train_galore":
        train_galore()
    elif args.mode == "train_qlora":
        train_qlora()
    elif args.mode == "evaluate":
        evaluate(
            hf_model='google/gemma-3-1b-it',
            qlora_dir='outputs/gemma_it_qlora',
            galore_dir='outputs/gemma_it_galore',
            test_file='data/processed/test_combined.jsonl',
            out_dir='results',
            max_examples=200
        )
    elif args.mode == "all":
        print("=== Step 1: Preprocessing ===")
        prepare_data(out_dir="data/", seed=42)

        print("\n=== Step 2: GaLore Training ===")
        train_galore()

        print("\n=== Step 3: QLoRA Training ===")
        train_qlora()

        print("\n=== Step 4: Evaluation ===")
        evaluate(
            hf_model='google/gemma-3-1b-it',
            qlora_dir='outputs/gemma_it_qlora',
            galore_dir='outputs/gemma_it_galore',
            test_file='data/processed/test_combined.jsonl',
            out_dir='results',
            max_examples=200
        )
