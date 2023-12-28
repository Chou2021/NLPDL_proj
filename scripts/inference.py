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
    load_mlb_dataset, load_nba_dataset,
    get_split_dataset, get_whole_dataset
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--input", type=str)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--lora", type=bool, default=False)

    return parser.parse_args()

def inference_lora(args):
    model_id = args.model_path
    input = args.input
    config = PeftConfig.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path, load_in_8bit=True, device_map='auto')
    model = PeftModel.from_pretrained(model, model_id, device_map='auto')
    model.eval()
    input_ids = torch.tensor(tokenizer.encode(input), device=device)
    outputs = model.generate(input_ids=input_ids, do_sample=True, top_p=0.9, max_new_tokens=args.max_new_tokens)
    outputs = tokenizer.decode(outputs[0].detach().cpu().numpy(), skip_special_tokens=True)
    print("============================================\noutput:\n" + outputs)

def inference_ft(args):
    model_id = args.model_path
    input = args.input
    tokenizer = T5Tokenizer.from_pretrained(model_id)
    model = T5ForConditionalGeneration.from_pretrained(model_id, device_map='auto')
    model.eval()
    input_ids = torch.tensor(tokenizer.encode(input), device=device)
    outputs = model.generate(input_ids=input_ids, do_sample=True, top_p=0.9, max_new_tokens=args.max_new_tokens)
    outputs = tokenizer.decode(outputs[0].detach().cpu().numpy(), skip_special_tokens=True)
    print("============================================\noutput:\n" + outputs)

def main():
    args = parse_args()
    if args.lora == True:
        inference_lora(args)
    else:
        inference_ft(args)

if __name__ == "__main__":
    main()