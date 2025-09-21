"""
Module for evaluating multiple language models on a test dataset using BLEU and ROUGE metrics.

This script loads three models (baseline, galore, qlora), generates predictions for each example in the test dataset,
and computes BLEU and ROUGE scores for the generated outputs against reference responses. The results are saved as a JSON file.

Dependencies:
    - torch
    - transformers
    - datasets
    - pathlib
    - rouge_score
    - sacrebleu
    - json

Functions:
    main(hf_model, qlora_dir, galore_dir, test_file, out_dir, max_examples=2000):
        Evaluates the specified models on the provided test dataset and saves the results.

Args:
    hf_model (str): Path or Hugging Face identifier for the baseline model.
    qlora_dir (str): Path to the QLoRA fine-tuned model directory.
    galore_dir (str): Path to the GaLore fine-tuned model directory.
    test_file (str): Path to the test dataset file in JSON format.
    out_dir (str): Directory to save the evaluation results.
    max_examples (int, optional): Maximum number of examples to evaluate. Defaults to 200.

Outputs:
    - Prints BLEU and ROUGE scores for each model.
    - Saves a JSON file with evaluation results in the specified output directory.


    Author: Cem Rifki Aydin
    Date: 14/09/2025

"""

import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from pathlib import Path
from rouge_score import rouge_scorer
import sacrebleu
import json

def normalize_text(text, max_sentences=3):
    """
    Normalize text:
    - Strip whitespace
    - Collapse repeated sentences
    - Keep only first `max_sentences` unique sentences
    """
    text = text.strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)

    seen = set()
    cleaned = []
    for s in sentences:
        s = s.strip()
        if s and s not in seen:
            cleaned.append(s)
            seen.add(s)
        if len(cleaned) >= max_sentences:
            break
    return ' '.join(cleaned)


def main(hf_model, qlora_dir, galore_dir, test_file, out_dir, max_examples=2000):
    """Evaluates multiple models on a test dataset and saves the results."""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_ds = load_dataset("json", data_files=test_file, split="train")
    if max_examples: test_ds = test_ds.select(range(min(max_examples, len(test_ds))))

    models = {
        "baseline": hf_model,
        "galore": galore_dir,
        "qlora": qlora_dir
    }

    results = {}
    for name, path in models.items():
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path).to(device)
        predictions, references = [], []
        for record in test_ds:
            prompt = record['instruction'] + " " + record['input']
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            out = model.generate(**inputs, max_new_tokens=128)

            pred = tokenizer.decode(out[0], skip_special_tokens=True)
            predictions.append(normalize_text(pred))
            references.append(normalize_text(record["response"]))


        bleu_score = sacrebleu.corpus_bleu(predictions, [references], tokenize='intl').score if len(predictions) > 0 else 0.0
        rouge = rouge_scorer.RougeScorer(['rouge1','rougeL'], use_stemmer=True)
        rouge_scores = [rouge.score(p,r) for p, r in zip(predictions,references)]

        avg_rouge1 = sum([x['rouge1'].fmeasure for x in rouge_scores])/len(rouge_scores)
        avg_rougeL = sum([x['rougeL'].fmeasure for x in rouge_scores])/len(rouge_scores)

        results[name] = {"BLEU": bleu_score, "ROUGE1": avg_rouge1, "ROUGEL": avg_rougeL}
        print(f"Evaluation for {name}: BLEU={bleu_score:.2f}, ROUGE1={avg_rouge1:.4f}, ROUGEL={avg_rougeL:.4f}")

    with open(out_dir / "evaluation.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
