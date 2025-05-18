import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from collections import defaultdict
from utils import extract_and_prepare_data, detrend_ue
from pathlib import Path
import pathlib

normalize = True

methods_dict = {
    'CometQE-wmt23-cometkiwi-da-xxl': 'Comet QE XXL',
    'CometQE-wmt23-cometkiwi-da-xl': 'Comet QE XL',
    'CometQE-wmt22-cometkiwi-da': 'Comet QE',
    'MaximumSequenceProbability': 'MSP',
    'Perplexity': 'PPL',
    'MeanTokenEntropy': 'MTE',
    'MonteCarloSequenceEntropy': 'MCSE',
    'MonteCarloNormalizedSequenceEntropy': 'MCNSE',
    'LexicalSimilarity_rougeL': 'LSRL',
}

MODELS = ['llama', 'gemma', 'eurollm']

DATASETS = {
    'metricx-metricx-24-hybrid-xxl-v2p6': [
        'wmt14_csen',
        'wmt14_deen',
        'wmt14_ruen',
        'wmt14_fren',
        'wmt19_deen',
        'wmt19_fien',
        'wmt19_lten',
        'wmt19_ruen',
    ],
    'XComet-XCOMET-XXL': [
        'wmt14_csen',
        'wmt14_deen',
        'wmt14_ruen',
        'wmt14_fren',
        'wmt19_deen',
        'wmt19_fien',
        'wmt19_lten',
        'wmt19_ruen',
    ],
    'Comet-wmt22-comet-da': [
        'wmt14_csen',
        'wmt14_deen',
        'wmt14_ruen',
        'wmt14_fren',
        'wmt19_deen',
        'wmt19_fien',
        'wmt19_lten',
        'wmt19_ruen',
    ],
    'bleu_proper': [
        'wmt14_csen',
        'wmt14_deen',
        'wmt14_ruen',
        'wmt14_fren',
        'wmt19_deen',
        'wmt19_fien',
        'wmt19_lten',
        'wmt19_ruen',
    ],
}

METRICS = {
    'metricx-metricx-24-hybrid-xxl-v2p6': 'MetricX XXL',
    'XComet-XCOMET-XXL': 'XComet XXL',
    'Comet-wmt22-comet-da': 'Comet WMT22',
    'bleu_proper': 'BLEU',
}

pathlib.Path('tables').mkdir(parents=True, exist_ok=True)
pathlib.Path('charts').mkdir(parents=True, exist_ok=True)

for model in MODELS:
    #for model_type in ['base', 'instruct']:
    for model_type in ['instruct_zeroshot']:
        prefix = '' if model_type == 'base' else '_instruct'
        if model_type == 'instruct_zeroshot':
            prefix = '_instruct_zeroshot'

        for metric, metric_name in METRICS.items():
            all_metrics = [metric]
            ue_methods = list(methods_dict.values())
            datasets = DATASETS[metric]

            ue_scores, ue_coefs, ave_test_metric_values = detrend_ue(datasets, model, model_type, all_metrics, ue_methods, methods_dict)

            def colname(dataset):
                if '_' in dataset:
                    return dataset.split('_')[1]
                return dataset

            columns = [colname(dataset) for dataset in datasets for metric in all_metrics] + ['raw_rank', 'detr_rank', 'rank']
            df = pd.DataFrame.from_dict(ue_scores, orient='index', columns=columns)
            name = f'tables/{model}{prefix}_{metric}_ue_scores_norm.tex'
            with open(name, 'w') as f:
                caption = f"PRRs for each method, with ranks for raw and detr methods, and total rank. Metric is {metric_name}, model is {model}{prefix}."
                latex = df.style.set_caption(caption).format(precision=2).to_latex()
                latex = latex.replace('_', '\_')

                lines = latex.split('\n')
                column_groups = ["&\\multicolumn{4}{c}{\\textbf{WMT14}}&\\multicolumn{4}{c}{\\textbf{WMT19}}\\\\", "\\cmidrule(lr){2-5}", "\\cmidrule(lr){6-9}"]
                base_quality_row = [''.join([f"&{val:.2f}" for val in list(ave_test_metric_values.values())]) + '&-&-&-\\\\']
                header = lines[0:1] + ['\\footnotesize'] + lines[1:3] + column_groups + base_quality_row
                body = lines[3:-3]
                footer = lines[-3:]
                latex = '\n'.join(header + [line if i % 2 != 0 else line + '\n\\midrule' for i, line in enumerate(body)] + footer)
                latex = latex.replace('table', 'table*')
                f.write(latex)

            columns = [colname(dataset) for dataset in datasets for metric in all_metrics]

            columns = []
            for dataset in datasets:
                for coef_type in ['train_c', 'test_c']:
                    columns.append(f'{dataset}_{coef_type}')
