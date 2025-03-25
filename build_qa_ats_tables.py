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
    'MaximumSequenceProbability': 'MSP',
    'Perplexity': 'PPL',
    'MeanTokenEntropy': 'MTE',
    'MonteCarloSequenceEntropy': 'MCSE',
    'MonteCarloNormalizedSequenceEntropy': 'MCNSE',
    'LexicalSimilarity_rougeL': 'LSRL',
}

MODELS = ['llama', 'gemma']
DATASETS = {
    'AlignScoreInputOutput': 'xsum',
    'Accuracy': 'gsm8k',
}

pathlib.Path('tables').mkdir(parents=True, exist_ok=True)
pathlib.Path('charts').mkdir(parents=True, exist_ok=True)

for model in MODELS:
    for model_type in ['base']:
        prefix = '' if model_type == 'base' else '_instruct'
        if model_type == 'instruct_zeroshot':
            prefix = '_instruct_zeroshot'

        datasets = DATASETS.values()
        all_metrics = DATASETS.keys()
        ue_methods = list(methods_dict.values())

        ue_scores, ue_coefs, ave_test_metric_values = detrend_ue(datasets, model, model_type, all_metrics, ue_methods, methods_dict, task='not_nmt')

        def colname(dataset):
            if '_' in dataset:
                return dataset.split('_')[1]
            return dataset

        columns = [colname(dataset) for dataset in datasets] + ['raw_rank', 'detr_rank', 'rank']
        df = pd.DataFrame.from_dict(ue_scores, orient='index', columns=columns)
        name = f'tables/{model}{prefix}_qa_ats_ue_scores_norm.tex'
        with open(name, 'w') as f:
            caption = f"PRRs for each method, with ranks for raw and detr methods, and total rank. Metrics are {all_metrics}, model is {model}{prefix}."
            latex = df.style.set_caption(caption).format(precision=2).to_latex()
            latex = latex.replace('_', '\_')

            lines = latex.split('\n')
            base_quality_row = [''.join([f"&{round(val,2)}" for val in ave_test_metric_values.values()]) + '&-&-&-\\\\']
            header = lines[0:1] + ['\\footnotesize'] + lines[1:3] + base_quality_row
            body = lines[3:-3]
            footer = lines[-3:]
            latex = '\n'.join(header + [line if i % 2 != 0 else line + '\n\\midrule' for i, line in enumerate(body)] + footer)
            f.write(latex)

        columns = [colname(dataset) for dataset in datasets]

        columns = []
        for dataset in datasets:
            for coef_type in ['train_c', 'test_c']:
                columns.append(f'{dataset}_{coef_type}')

        df = pd.DataFrame.from_dict(ue_coefs, orient='index', columns=columns)
        name = f'tables/{model}{prefix}_qa_ats_ue_trends.tex'
        if normalize:
            name = f'tables/{model}{prefix}_qa_ats_ue_trends_norm.tex'

        with open(name, 'w') as f:
            latex = df.to_latex(float_format="%.3f", escape=False)
            latex = latex.replace('_', '\_')
            f.write(latex)
