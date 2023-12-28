# Can T5 be taught twice for data-to-text generation?

## Data-to-Text Dataset
### sportsett_basketball
Download from https://huggingface.co/datasets/GEM/sportsett_basketball

### mlb_data_to_text
Download from https://huggingface.co/datasets/GEM/mlb_data_to_text

### e2e_nlg
Download from https://github.com/tuetschek/e2e-dataset

### viggo
Download from https://huggingface.co/datasets/GEM/viggo

## Usage 
### Install Dependencies
```shell
pip install -r requirements.txt
```
### Training
```shell
python lora.py --type 0 # or 1, 2
python ft.py --type 0 # or 1, 2
```

### Evaluation
```bash
python eval.py --model_path checkpoints/t5_base_0 --type 0 --split test
```

## Reference:
[Text-to-Text Pre-Training for Data-to-Text Tasks](https://arxiv.org/pdf/2005.10433.pdf) \
[Biomedical Data-to-Text Generation via Fine-Tuning Transformers](https://arxiv.org/abs/2109.01518) \
[fine-tune-flan-t5-peft](https://www.philschmid.de/fine-tune-flan-t5-peft)