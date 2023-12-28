import numpy as np
import evaluate
from parent import parent

def cal_parent(generations, references, tables):
    """
    Calculate a specifying-table-to-text metrics PARENT (need structured data) 
    
    input:
    - generations: list of model generation (str)
    - references: list of target descriptions (str)
    - table: list of structured data (str)

    output:
    - result: dict of three scores: precision, recall, f_score
    """
    precision, recall, f_score = parent(
        predictions=generations, 
        references=references,
        tables=tables,
        avg_results=True,
        n_jobs=32
    )
    result = {'precision': precision, 'recall': recall, 'f_score': f_score}
    return result


def cal_rouge(generations, references):
    """
    Calculate ROUGE score

    input:
    - generations: list of model generation (str)
    - references: list of target descriptions (str)

    output:
    - rouge: ROUGE score (dict)
    """

    rouge_metric = evaluate.load('./metrics/rouge')
    rouge = rouge_metric.compute(predictions=generations, references=references, use_aggregator=True)

    return rouge


def cal_bleu(generations, references):
    """
    Calculate BLEU score

    input:
    - generations: list of model generation (str)
    - references: list of target descriptions (str)

    output:
    - bleu: BLEU score (float 0~1)
    """

    bleu_metric = evaluate.load('./metrics/sacrebleu')
    bleu =  bleu_metric.compute(predictions=generations, references=references)['score']

    return {'sacrebleu': bleu}


def cal_bertscore(generations, references):
    """
    Calculate BERTScore with Huggingface API

    input:
    - generations: list of model generation (str)
    - references: list of target descriptions (str)

    output:
    - bertscore: BERTScore
    """

    bertscore_metric = evaluate.load('./metrics/bertscore')
    bertscore = bertscore_metric.compute(predictions=generations, references=references, 
                                         lang='en')

    bertscore = np.mean(bertscore['f1'])

    return bertscore

def cal_meteor(generations, references):
    """
    Calcute meteor score
    """

    meteor = evaluate.load('./metrics/meteor')
    score = meteor.compute(predictions=generations, references=references)
   
    return score[0]

def calculate_metrics(generations, references, tables=None):
    """
    Calculate above three metrics and return a result dictionary
    
    input:
    - generations: list of model generation (str)
    - references: list of target descriptions (str)

    output:
    - results: {'bleu', 'rouge', 'bertscore', 'meteor', 'PARENT'}
    """

    results = dict()
    results.update(cal_bleu(generations=generations, references=references))
    results['rouge'] = cal_rouge(generations=generations, references=references)
    results['meteor'] = cal_meteor(generations, references)
    results['bertscore'] = cal_bertscore(generations=generations, references=references)
    if tables is not None:
        results['PARENT'] = cal_parent(generations=generations, references=references, tables=tables)
    return results
