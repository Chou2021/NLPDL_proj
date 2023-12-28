import argparse
import json
import time
import os
import sys
from logging import getLogger
import torch
import numpy as np
import evaluate
from tqdm import tqdm

from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM, 
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
    GenerationConfig,
    set_seed,
)

from peft import (
    LoraConfig,
    get_peft_model, 
    prepare_model_for_int8_training, 
    TaskType,
    PeftModel,
    PeftConfig
)

from utils.data_helper import (
    load_e2e_dataset, 
    load_viggo_dataset, 
    get_split_dataset,
)

from utils.cal_metrics import (
    cal_bleu, cal_rouge, 
    cal_bertscore, cal_parent,
    calculate_metrics
)

logger = getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# lora on e2e: python lora.py --type 0
# lora on viggo: python lora.py --type 1
# lora on both: python lora.py --type 2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=int, default=0) # 0 for only e2e and 1 for only viggo and 2 for both
    parser.add_argument("--save_path", type=str, default="./checkpoints/")    # checkpoints dir
    parser.add_argument("--metrics_path", type=str, default="metrics.json")   # metrics path
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8, required=False)

    return parser.parse_args()


def main():
    """ Fine-tune t5-3b with PEFT """
    # Load t5-3b and tokenizer
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained('t5-3b')
    if args.type == 2:
        model = AutoModelForSeq2SeqLM.from_pretrained('t5-3b', load_in_8bit=True, device_map='auto')
        model = prepare_model_for_int8_training(model)
        model = PeftModel.from_pretrained(model, args.save_path + "peft_t5_3b_0", device_map='auto')
        model.train()
        model.print_trainable_parameters()
        # Prepare for training
        label_pad_token_id = -100
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8,
        )
        train_dataset = get_split_dataset(tokenizer, 1, "train")
        output_dir = args.save_path + "peft_t5_3b_2"
        peft_training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            auto_find_batch_size=True,
            learning_rate=1e-3,
            num_train_epochs=3,
            logging_dir=f"{output_dir}/logs",
            logging_strategy="steps",
            logging_steps=500,
            save_strategy="no",
            report_to="tensorboard",
        )
        peft_trainer = Seq2SeqTrainer(
            model=model,
            args=peft_training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
        )
        peft_trainer.train()
        peft_trainer.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained('t5-3b', load_in_8bit=True, device_map='auto')
        # Define LoRA config and load model with low-rank adapter
        lora_config = LoraConfig(
            r=16, 
            lora_alpha=32, 
            target_modules=["q", "v"],
            lora_dropout=0.05,
            bias="none", 
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        model = prepare_model_for_int8_training(model)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        # Prepare for training
        label_pad_token_id = -100
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8,
        )
        train_dataset = get_split_dataset(tokenizer, args.type, "train")
        output_dir = args.save_path + f"peft_t5_3b_{args.type}"
        peft_training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            auto_find_batch_size=True,
            learning_rate=1e-3,
            num_train_epochs=3,
            logging_dir=f"{output_dir}/logs",
            logging_strategy="steps",
            logging_steps=500,
            save_strategy="no",
            report_to="tensorboard",
        )
        peft_trainer = Seq2SeqTrainer(
            model=model,
            args=peft_training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
        )
        peft_trainer.train()
        peft_trainer.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
