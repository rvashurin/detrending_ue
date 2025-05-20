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

def create_subplot_chart(ax, datasets, raw_scores, detrended_scores, model_name, subplot_idx):
    """
    Create a bar chart subplot comparing raw and detrended scores for each dataset.
    """
    # Separate WMT14 and WMT19 datasets
    wmt14_indices = [i for i, d in enumerate(datasets) if 'wmt14' in d]
    wmt19_indices = [i for i, d in enumerate(datasets) if 'wmt19' in d]

    # Calculate x positions for grouped bars
    width = 0.25  # Slimmer bars
    x_wmt14 = np.arange(len(wmt14_indices))
    x_wmt19 = np.arange(len(wmt19_indices)) + len(wmt14_indices) + 0.5  # Add gap between groups

    # Plot WMT14 datasets
    rects1_wmt14 = ax.bar(x_wmt14 - width/2, [raw_scores[i] for i in wmt14_indices], 
                         width, label='Raw PRR', color='lightblue')
    rects2_wmt14 = ax.bar(x_wmt14 + width/2, [detrended_scores[i] for i in wmt14_indices], 
                         width, label='Detrended PRR', color='orange')

    # Plot WMT19 datasets
    rects1_wmt19 = ax.bar(x_wmt19 - width/2, [raw_scores[i] for i in wmt19_indices], 
                         width, color='lightblue')
    rects2_wmt19 = ax.bar(x_wmt19 + width/2, [detrended_scores[i] for i in wmt19_indices], 
                         width, color='orange')

    # Larger font sizes
    if subplot_idx % 3 == 0:
        ax.set_ylabel('PRR Score', fontsize=20)
    ax.set_title(model_name, fontsize=20)

    # Set x-ticks and labels
    all_x = np.concatenate([x_wmt14, x_wmt19])
    all_datasets = ([datasets[i] for i in wmt14_indices] + 
                   [datasets[i] for i in wmt19_indices])
    ax.set_xticks(all_x)
    ax.set_xticklabels([colname(d) for d in all_datasets], rotation=45, fontsize=14)

    # Add group labels
    ax.text(np.mean(x_wmt14), ax.get_ylim()[0] - 0.11, 'WMT14', 
            ha='center', va='top', fontsize=16)
    ax.text(np.mean(x_wmt19), ax.get_ylim()[0] - 0.11, 'WMT19', 
            ha='center', va='top', fontsize=16)

    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=14)

    return rects1_wmt14[0], rects2_wmt14[0]  # Return first pair for legend

def main(args):
    for metric, metric_name in METRICS.items():
        datasets = DATASETS[metric]
        str_select = "Best" if args.select == 'best' else methods_dict[args.select]

        caption = f"{str_select} PRR scores before and after detrending procedure. Metric is {metric_name}."
        header = get_header(caption)
        header += "&" + "&".join([colname(dataset).replace('_', '\\_') for dataset in datasets]) + "\\\\\n"
        latex = header

        # Create a figure for all models
        fig, axs = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'{str_select} PRR Scores Comparison - {metric_name}', fontsize=24)  # Larger title
        axs = axs.flatten()
        
        subplot_idx = 0
        first_rects = None  # To store the first pair of rectangles for legend

        for model_type in ['base', 'instruct']:
            for model, model_name in MODELS.items():
                all_metrics = [metric]
                ue_methods = list(methods_dict.values())
                latex += "\midrule\n"
                model_title = model_name if model_type == 'base' else f"{model_name} Instruct"
                latex += "& \\multicolumn{7}{c}{" + model_title + "}\\\\\n"
                latex += "\midrule\n"

                ue_scores, ue_coefs, ave_test_metric_values = detrend_ue(datasets, model, model_type, all_metrics, ue_methods, methods_dict, return_unprocessed=True)

                raw_scores_float = []
                detr_scores_float = []
                if args.select == 'best':
                    best_raw_scores = []
                    best_detr_scores = []

                    for i in range(len(datasets)):
                        best_raw_score = np.max([method_scores[i] for method, method_scores in ue_scores.items() if 'raw' in method])
                        best_detr_score = np.max([method_scores[i] for method, method_scores in ue_scores.items() if 'detr' in method])

                        raw_scores_float.append(best_raw_score)
                        detr_scores_float.append(best_detr_score)

                        if best_raw_score > best_detr_score:
                            best_raw_scores.append(f"\\textbf{{{best_raw_score:.2f}}}")
                            best_detr_scores.append(f"{best_detr_score:.2f}")
                        else:
                            best_raw_scores.append(f"{best_raw_score:.2f}")
                            best_detr_scores.append(f"\\textbf{{{best_detr_score:.2f}}}")

                    latex += f"{str_select} Raw PRR & " + " & ".join(best_raw_scores) + "\\\\\n"
                    latex += f"{str_select} Detrended PRR & " + " & ".join(best_detr_scores) + "\\\\\n"
                else:
                    method = args.select
                    method_raw_scores = []
                    method_detr_scores = []

                    for i in range(len(datasets)):
                        raw_score = ue_scores[f"{methods_dict[method]}_raw"][i]
                        detr_score = ue_scores[f"{methods_dict[method]}_detr"][i]

                        raw_scores_float.append(raw_score)
                        detr_scores_float.append(detr_score)

                        if raw_score > detr_score:
                            method_raw_scores.append(f"\\textbf{{{raw_score:.2f}}}")
                            method_detr_scores.append(f"{detr_score:.2f}")
                        else:
                            method_raw_scores.append(f"{raw_score:.2f}")
                            method_detr_scores.append(f"\\textbf{{{detr_score:.2f}}}")

                    latex += f"{str_select} Raw PRR & " + " & ".join(method_raw_scores) + "\\\\\n"
                    latex += f"{str_select} Detrended PRR & " + " & ".join(method_detr_scores) + "\\\\\n"

                # Create subplot without adding a legend
                rects1, rects2 = create_subplot_chart(
                    axs[subplot_idx],
                    datasets, 
                    raw_scores_float, 
                    detr_scores_float, 
                    model_title,
                    subplot_idx,
                )

                # Store first pair of rectangles for legend
                if subplot_idx == 0:
                    first_rects = (rects1, rects2)

                subplot_idx += 1

        # Add a single legend for the entire figure using the first pair of rectangles
        legend_text = [f"{str_select} Raw PRR", f"{str_select} Detrended PRR"]

        fig.legend(
            first_rects,
            legend_text,
            loc='center right',
            bbox_to_anchor=(0.18, 0.48),
            fontsize=18  # Larger legend font
        )

        # Adjust layout to prevent overlapping
        plt.tight_layout()
        # Adjust for the main title and legend
        plt.subplots_adjust(top=0.9, hspace=0.6)

        # Save the figure
        fig.savefig(f'charts/{metric}_{args.select}_all_models_comparison.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

        latex += footer()

        name = f'tables/best_tables/{metric}_{args.select}_ue_scores_norm.tex'
        with open(name, 'w') as f:
            f.write(latex)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate NMT tables.")
    parser.add_argument('--select', type=str, default='best', choices=list(methods_dict.keys()) + ['best'], help="Select the method to use for comparison.")
    args = parser.parse_args()

    main(args)
