import datasets
from datasets import load_dataset, load_from_disk
import numpy as np

"""
After analyzing the dataset:
viggo: max_source_length = 60 ; max_target_length = 50
e2e_nlg: max_source_length = 80 ; max_target_length = 64
"""

def load_e2e_dataset(split="train"):
    assert split in ['train', 'validation', 'test']
    dataset = load_from_disk('./e2e_nlg')[split]
    if split == 'train':
        dataset = dataset.select(list(np.arange(0, len(dataset), 6)))
    def preprocess(sample):
        inputs = "inform(" + sample['meaning_representation'] + ")"
        target = sample['target']
        references = [target]
        results = {
            'linearized_input': inputs,
            'target': target,
            'references': references,
        }
        return results
        
    dataset = dataset.map(preprocess, remove_columns=list(dataset.features))
    print(dataset)
    return dataset

def load_viggo_dataset(split="train"):
    assert split in ['train', 'validation', 'test']
    dataset = load_from_disk("./viggo")[split]
    def preprocess(sample):
        inputs = sample['meaning_representation']
        target = sample['target']
        references = sample['references']
        results = {
            'linearized_input': inputs,
            'target': target,
            'references': references,
        }
        return results

    dataset = dataset.map(preprocess, remove_columns=list(dataset.features))
    return dataset


def get_split_dataset(tokenizer, sign=0, split="train"):
    if sign == 0:
        dataset = load_e2e_dataset(split=split)
    elif sign == 1:
        dataset = load_viggo_dataset(split=split)
    else:
        raise ValueError(f"Invalid dataset name, should be 0 (e2e) or 1 (viggo)")

    def apply_prompt_template(sample):
        inputs = [item for item in sample['linearized_input']]

        model_inputs = tokenizer(inputs, max_length=100, padding="max_length", truncation=True)

        labels = tokenizer(text_target=sample['target'], max_length=100, padding="max_length", truncation=True)
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
        model_inputs['labels'] = labels['input_ids']

        return model_inputs
 
    tokenized_dataset = dataset.map(apply_prompt_template, batched=True, remove_columns=list(dataset.features))
    print(f"Keys of tokenized dataset: {list(tokenized_dataset.features)}")
    return tokenized_dataset
