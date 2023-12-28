import argparse
import os
import sys
import json
import time
import warnings
from logging import getLogger
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np
from tqdm import tqdm

from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer,
    T5Tokenizer, 
    T5ForConditionalGeneration,
)
from peft import (
    PeftModel,
    PeftConfig,
    LoraConfig,
)
from utils.data_helper import (
    load_e2e_dataset, 
    load_viggo_dataset,
    get_split_dataset,
)
from utils.cal_metrics import (
    cal_rouge,
    cal_bleu,
    cal_bertscore,
    cal_parent,
    calculate_metrics,
)

logger = getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# peft_t5_3b:
## python eval.py --model_path checkpoints/peft_t5_3b_0 --type 0 --split 'test' --lora True
## python eval.py --model_path checkpoints/peft_t5_3b_1 --type 1 --split 'test' --lora True
## python eval.py --model_path checkpoints/peft_t5_3b_2 --type 0 --split 'test' --lora True
## python eval.py --model_path checkpoints/peft_t5_3b_2 --type 1 --split 'test' --lora True

# t5_large or t5_base:
## python eval.py --model_path checkpoints/t5_base_0(t5_large_0) --type 0 --split 'test'
## python eval.py --model_path checkpoints/t5_base_1(t5_large_0) --type 1 --split 'test'
## python eval.py --model_path checkpoints/t5_base_2(t5_large_0) --type 0 --split 'test'
## python eval.py --model_path checkpoints/t5_base_2(t5_large_0) --type 1 --split 'test'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--type", type=int, default=0) # 0 for only e2e and 1 for only viggo
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--save_path", type=str, default="checkpoints/")    # checkpoints dir
    parser.add_argument("--metrics_path", type=str, default="metrics.json")   # metrics path
    parser.add_argument("--batch_size", type=int, default=32, required=False)
    parser.add_argument("--lora", type=bool, default=False)

    return parser.parse_args()

def handle_metrics(split, metrics, output_dir):
    """
    Log and save metrics
    
    input:
    - split: one of train, validation, test
    - metrics: metrics dict
    - output_dir: 
    """
    logger.info(f"------ {split} metrics ------")
    for key in sorted(metrics.keys()):
        logger.info(f"  {key} = {metrics[key]}")
    with open(os.path.join(output_dir, f"{split}_results.json"), 'w') as f:
        json.dump(metrics, f)


def eval_lora(args):
    # Load peft model for evaluation
    model_id = args.model_path
    config = PeftConfig.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path, load_in_8bit=True, device_map='auto')
    model = PeftModel.from_pretrained(model, model_id, device_map='auto')
    model.eval()

    # Load dataset for evaluation
    if args.type == 0:
        tokenized_dataset = get_split_dataset(tokenizer, 0, args.split)
        dataset = load_e2e_dataset(args.split)
    elif args.type == 1:
        tokenized_dataset = get_split_dataset(tokenizer, 1, args.split)
        dataset = load_viggo_dataset(args.split)
    else:
        raise ValueError(f"Invalid dataset name, should be 0 (e2e) or 1 (viggo)")

    # Evaluation
    generations = []
    references = []
    tables = []
    for sample, data in tqdm(zip(tokenized_dataset, dataset)):
        inputs = torch.tensor(sample['input_ids'])
        outputs = model.generate(input_ids=inputs.unsqueeze(0).cuda(), do_sample=True, top_p=0.9, max_new_tokens=64)
        prediction = tokenizer.decode(outputs[0].detach().cpu().numpy(), skip_special_tokens=True)
        reference = data['references']
        table = data['linearized_input']
        generations.append(prediction)
        references.append(reference)
        tables.append(table)

    results = calculate_metrics(generations=generations, references=references, tables=tables)
    handle_metrics(f'{args.type}_{args.split}', results, model_id)


def eval_ft(args):
    # Load fine-tuned model for evaluation
    model_id = args.model_path
    base_model = model_id.split('/')[1][:-2] # t5_large or t5_base
    tokenizer = T5Tokenizer.from_pretrained()
    model = T5ForConditionalGeneration.from_pretrained(model_id, return_dict=True, config=f'./{base_model}/config.json').to(device)
    model.eval()

    # Load dataset for evaluation
    if args.type == 0:
        dataset = load_e2e_dataset(args.split)
    elif args.type == 1:
        dataset = load_viggo_dataset(args.split)
    else:
        raise ValueError(f"Invalid dataset name, should be 0 (e2e) or 1 (viggo)")

    # Evaluation
    with torch.no_grad():
        generations = []
        references = []
        tables = []
        for data in tqdm(dataset):
            input_ids = tokenizer.encode(data['linearized_input'], return_tensors="pt").to(device)
            outputs = model.generate(input_ids, do_sample=True, top_p=0.9, max_new_tokens=64)
            prediction = tokenizer.decode(outputs[0].detach().cpu().numpy(), skip_special_tokens=True)
            reference = data['references']
            table = data['linearized_input']
            generations.append(prediction)
            references.append(reference)
            tables.append(table)

    results = calculate_metrics(generations=generations, references=references, tables=tables)
    handle_metrics(f'{args.type}_{args.split}', results, model_id)


def main():
    args = parse_args()
    if args.lora == True:
        eval_lora(args)
    else:
        eval_ft(args)


if __name__ == "__main__":
    main()
