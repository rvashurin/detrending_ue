import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from collections import defaultdict
from utils import extract_and_prepare_data, detrend_ue
from pathlib import Path
import pathlib
import argparse

normalize = True

methods_dict = {
    #'CometQE-wmt23-cometkiwi-da-xxl': 'Comet QE XXL',
    'MaximumSequenceProbability': 'MSP',
    'Perplexity': 'PPL',
    'MeanTokenEntropy': 'MTE',
    'MonteCarloSequenceEntropy': 'MCSE',
    'MonteCarloNormalizedSequenceEntropy': 'MCNSE',
    'LexicalSimilarity_rougeL': 'LSRL',
}

MODELS = {
    'llama': 'Llama 3.1 8B',
    'gemma': 'Gemma 2 9B',
    'eurollm': 'EuroLLM 9B',
}

DATASETS = [
    'wmt14_csen',
    'wmt14_deen',
    'wmt14_ruen',
    'wmt14_fren',
    'wmt19_deen',
    'wmt19_fien',
    'wmt19_lten',
    'wmt19_ruen',
]

METRICS = {
    'metricx-metricx-24-hybrid-xxl-v2p6': 'MetricX XXL',
    'XComet-XCOMET-XXL': 'XComet XXL',
    'Comet-wmt22-comet-da': 'Comet WMT22',
    'bleu_proper': 'BLEU',
}

pathlib.Path('tables').mkdir(parents=True, exist_ok=True)
pathlib.Path('charts').mkdir(parents=True, exist_ok=True)

def get_header(caption):
    return (
        "\\begin{table*}\n"
        "\\footnotesize\n"
        f"\caption{{{caption}}}\n"
        "\\begin{tabular}{lcccccccc}\n"
        "&\multicolumn{4}{c}{\\textbf{WMT14}}&\multicolumn{4}{c}{\\textbf{WMT19}}\\\\\n"
        "\cmidrule(lr){2-5}\n"
        "\cmidrule(lr){6-9}\n"
    )

def footer():
    return (
        "\midrule\n"
        "\end{tabular}\n"
        "\end{table*}\n"
    )

def colname(dataset):
    if '_' in dataset:
        dataset = dataset.split('_')[1]

    return dataset[:2].capitalize() + '-' + dataset[2:].capitalize()


def main():
    datasets = DATASETS

    caption = f"Average metric scores"
    header = get_header(caption)
    header += "&" + "&".join([colname(dataset).replace('_', '\\_') for dataset in datasets]) + "\\\\\n"
    latex = header

    for model_type in ['base', 'instruct']:
        for model, model_name in MODELS.items():
            latex += "\midrule\n"
            model_title = model_name if model_type == 'base' else f"{model_name} Instruct"
            latex += "& \\multicolumn{7}{c}{" + model_title + "}\\\\\n"
            latex += "\midrule\n"

            for metric, metric_name in METRICS.items():
                ue_methods = list(methods_dict.values())
                all_metrics = [metric]
                ue_scores, ue_coefs, ave_test_metric_values = detrend_ue(datasets, model, model_type, all_metrics, ue_methods, methods_dict, return_unprocessed=True)

                str_metrics = [f"{value:.2f}" for value in ave_test_metric_values.values()]
                latex += f"{metric_name} & " + " & ".join(str_metrics) + "\\\\\n"

        latex += footer()

        name = f'tables/base_metric_values.tex'
        with open(name, 'w') as f:
            f.write(latex)

if __name__ == "__main__":
    main()
