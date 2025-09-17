"""
This module provides utilities for sampling, preprocessing, and aggregating instruction-following datasets for fine-tuning large language models.

Functions:
    map_record(ds_name, record):
        Maps a dataset record to a standardized dictionary with 'instruction', 'input', and 'response' fields, handling variations in dataset schemas.

    sample_dataset(name, train_n=5000, test_n=2000, seed=SEED):
        Samples train and test splits from a HuggingFace dataset, supporting special handling for UltraChat datasets. Returns shuffled and size-limited splits.

    write_jsonl(records, path):
        Writes a list of records (dictionaries) to a JSON Lines (.jsonl) file at the specified path.

    prepare_data(out_dir, seed=SEED):
        Aggregates and preprocesses multiple instruction datasets, filters out empty responses, shuffles the data, and writes combined train and test splits to disk in JSONL format.

Constants:
    SEED:
        Default random seed for reproducibility.

Usage:
    Call `prepare_data(out_dir)` to sample, preprocess, and save combined train/test splits from supported datasets into the specified output directory.

    Author: Cem Rifki Aydin
    Date: 14/09/2025

"""

import random, json
from datasets import load_dataset
from pathlib import Path

SEED = 42

def map_record(ds_name, record):
    instruction = ""
    inp = ""
    response = ""

    if 'alpaca' in ds_name:
        instruction = record.get('instruction') or record.get('prompt') or ''
        response = record.get('output') or record.get('response') or ''

    elif 'tulu' in ds_name or 'ultrachat' in ds_name.lower():
        # Prefer messages/messages list
        msgs = record.get('messages') or record.get('message') or record.get('conversations') or []
        if isinstance(msgs, list) and len(msgs) >= 2:
            # Find last user/assistant pair
            user_msg = next((m for m in reversed(msgs) if m.get('role') == 'user'), None)
            assistant_msg = next((m for m in reversed(msgs) if m.get('role') == 'assistant'), None)

            if user_msg:
                instruction = user_msg.get('content','')
            if assistant_msg:
                response = assistant_msg.get('content','')

        # Fallbacks in case messages missing or format unusual
        instruction = instruction or record.get('instruction') or record.get('prompt') or record.get('text','')
        response = response or record.get('output') or record.get('response') or record.get('answer','') or record.get('completion','')

    # Generic fallback
    if not instruction:
        instruction = record.get('prompt') or record.get('input') or record.get('text','')

    return {'instruction': instruction, 'input': inp, 'response': response}


def sample_dataset(name, train_n=5000, test_n=2000, seed=SEED):
    print(f"Loading {name}...")

    # special case: ultrachat
    if "ultrachat" in name.lower():
        ds_train = load_dataset(name, split="train_sft")
        ds_test  = load_dataset(name, split="test_sft")

        # Shuffle and subsample
        ds_train = ds_train.shuffle(seed=seed).select(range(min(train_n, len(ds_train))))
        ds_test  = ds_test.shuffle(seed=seed).select(range(min(test_n, len(ds_test))))
        return ds_train, ds_test

    # default case (alpaca, tulu, etc.)
    else:
        ds = load_dataset(name, split="train")
        ds = ds.shuffle(seed=seed)
        total = len(ds)
        train_n = min(train_n, total)
        test_n = min(test_n, max(0, total-train_n))
        train = ds.select(range(train_n))
        test = ds.select(range(train_n, train_n+test_n))
        return train, test

def write_jsonl(records, path):
    with open(path,'w',encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

def prepare_data(out_dir, seed):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        'tatsu-lab/alpaca',
        'allenai/tulu-v2-sft-mixture',
        'HuggingFaceH4/ultrachat_200k'
    ]

    train_agg = []
    test_agg = []

    for name in datasets:
        try:
            t, v = sample_dataset(name, seed=seed)
        except Exception as e:
            print('Error loading', name, e)
            continue

        # map records to unified format
        mapped_train = [map_record(name, dict(r)) for r in t]
        mapped_test  = [map_record(name, dict(r)) for r in v]

        train_agg.extend(mapped_train)
        test_agg.extend(mapped_test)
        # print(len(test_agg))

    # # remove records with empty response
    train_agg = [r for r in train_agg if r.get('response')]
    test_agg  = [r for r in test_agg if r.get('response')]

    # shuffle the combined datasets
    random.seed(seed)
    random.shuffle(train_agg)
    random.shuffle(test_agg)

    # save to JSONL
    proc_dir = out_dir / 'processed'
    proc_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(train_agg, proc_dir / 'train_combined.jsonl')
    write_jsonl(test_agg, proc_dir / 'test_combined.jsonl')

    print('Wrote', len(train_agg), 'train and', len(test_agg), 'test records to', proc_dir)
