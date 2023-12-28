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
import matplotlib.pyplot as plt

from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
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
from transformers.optimization import Adafactor

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

# ft on e2e: python ft.py --type 0
# ft on viggo: python ft.py --type 1
# ft on both: python ft.py --type 2

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
    """ Full fine-tune t5-base """
    args = parse_args()
    # Load t5-base and tokenizer
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    
    if args.type == 2:
        # Load fine-tuned model t5_base_0 as base model
        base_path = args.save_path + "t5_base_0"
        model = T5ForConditionalGeneration.from_pretrained(base_path, return_dict=True, config='./t5-base/config.json')
        model.to(device)
        
        # Prepare for training
        train_dataset = load_viggo_dataset("train")
        val_dataset = load_viggo_dataset("validation")
        num_epochs = args.num_epochs
        batch_size = args.batch_size
        num_batches = int(len(train_dataset) / batch_size)
        val_num_batches = int(len(val_dataset) / batch_size)
        optimizer = Adafactor(
            model.parameters(),
            lr=args.lr,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=0.0,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )
        
        # Training
        model.train()
        loss_per_steps = []

        for epoch in tqdm(range(1, num_epochs + 1)):
            print(f"Running epoch: {epoch}")
            total_loss = 0
            for i in range(num_batches):
                # Load data_batch
                batches = train_dataset[i * batch_size: i * batch_size + batch_size]
                inputs = batches['linearized_input']
                inputs = tokenizer.batch_encode_plus(inputs, padding=True, max_length=200, return_tensors="pt")['input_ids']
                inputs = inputs.to(device)
                labels = batches['target']
                labels = tokenizer.batch_encode_plus(labels, padding=True, max_length=200, return_tensors="pt")['input_ids']
                labels = labels.to(device)

                # zero optimizer's grad
                optimizer.zero_grad()

                # generation predictions
                outputs = model(input_ids=inputs, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                loss_per_steps.append(loss.item())

                # backward propagation and update
                loss.backward()
                optimizer.step()
                
                # validation
                if i % 100 == 0:
                    model.eval()
                    val_loss = []
                    with torch.no_grad():
                        for j in range(val_num_batches):
                            batches = val_dataset[j * batch_size: j * batch_size + batch_size]
                            inputs = batches['linearized_input']
                            inputs = tokenizer.batch_encode_plus(inputs, padding=True, max_length=200, return_tensors="pt")['input_ids']
                            inputs = inputs.to(device)
                            labels = batches['target']
                            labels = tokenizer.batch_encode_plus(labels, padding=True, max_length=200, return_tensors="pt")['input_ids']
                            labels = labels.to(device)
                            loss = model(input_ids=inputs, labels=labels).loss
                            val_loss.append(loss.item())
                    print(f"Batch_num {i} validation loss: {np.mean(val_loss)}")
                    model.train()
                        
            print(f"Epoch {epoch} training loss: {total_loss/num_batches}")        

        plt.plot(range(len(loss_per_steps)), loss_per_steps)
        plt.title(f'Training loss for t5-base-2')
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.savefig(args.save_path + 't5_base_2/loss.jpg')
        
        # Save model
        torch.save(model.state_dict(), args.save_path + 't5_base_2/pytorch_model.bin')    


    else:
        # Load fine-tuned model t5_base_0 as base model
        model = T5ForConditionalGeneration.from_pretrained('t5-base', return_dict=True)
        model.to(device)
        
        # Prepare for training
        assert args.type in [0, 1]
        if args.type == 0:
            train_dataset = load_e2e_dataset("train")
            val_dataset = load_e2e_dataset("validation")
        else:
            train_dataset = load_viggo_dataset("train")
            val_dataset = load_viggo_dataset("validation")
        num_epochs = args.num_epochs
        batch_size = args.batch_size
        num_batches = int(len(train_dataset) / batch_size)
        val_num_batches = int(len(val_dataset) / batch_size)
        optimizer = Adafactor(
            model.parameters(),
            lr=args.lr,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=0.0,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )
        
        # Training
        model.train()
        loss_per_steps = []

        for epoch in tqdm(range(1, num_epochs + 1)):
            print(f"Running epoch: {epoch}")
            total_loss = 0
            for i in range(num_batches):
                # Load data_batch
                batches = train_dataset[i * batch_size: i * batch_size + batch_size]
                inputs = batches['linearized_input']
                inputs = tokenizer.batch_encode_plus(inputs, padding=True, max_length=200, return_tensors="pt")['input_ids']
                inputs = inputs.to(device)
                labels = batches['target']
                labels = tokenizer.batch_encode_plus(labels, padding=True, max_length=200, return_tensors="pt")['input_ids']
                labels = labels.to(device)

                # zero optimizer's grad
                optimizer.zero_grad()

                # generation predictions
                outputs = model(input_ids=inputs, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                loss_per_steps.append(loss.item())

                # backward propagation and update
                loss.backward()
                optimizer.step()
                
                # validation
                if i % 100 == 0:
                    model.eval()
                    val_loss = []
                    with torch.no_grad():
                        for j in range(val_num_batches):
                            batches = val_dataset[j * batch_size: j * batch_size + batch_size]
                            inputs = batches['linearized_input']
                            inputs = tokenizer.batch_encode_plus(inputs, padding=True, max_length=200, return_tensors="pt")['input_ids']
                            inputs = inputs.to(device)
                            labels = batches['target']
                            labels = tokenizer.batch_encode_plus(labels, padding=True, max_length=200, return_tensors="pt")['input_ids']
                            labels = labels.to(device)
                            loss = model(input_ids=inputs, labels=labels).loss
                            val_loss.append(loss.item())
                    print(f"Batch_num {i} validation loss: {np.mean(val_loss)}")
                    model.train()
                        
            print(f"Epoch {epoch} training loss: {total_loss/num_batches}")        

        plt.plot(range(len(loss_per_steps)), loss_per_steps)
        plt.title(f'Training loss for t5-base-{args.type}')
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.savefig(args.save_path + f't5_base_{args.type}/loss.jpg')
        
        # Save model
        torch.save(model.state_dict(), args.save_path + f't5_base_{args.type}/pytorch_model.bin')

if __name__ == "__main__":
    main()
