import numpy as np
import matplotlib.pyplot as plt
from lm_polygraph.ue_metrics.pred_rej_area import PredictionRejectionArea
from lm_polygraph.ue_metrics.ue_metric import (
    get_random_scores,
    normalize_metric,
)
import sklearn
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from collections import defaultdict
from utils import extract_and_prepare_data
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

#MODELS = ['llama', 'mistral7b', 'stablelm12b']
MODELS = ['llama8b']
LLAMA_DATASETS = [
    'wmt14_csen',
   'wmt14_deen',
    'wmt14_ruen',
   'wmt14_fren',
   'wmt19_deen',
   'wmt19_fien',
   'wmt19_lten',
   'wmt19_ruen',
]

DATASETS = []
#    'wmt14',
#    'wmt19',
#]

#METRICS = ['Comet', 'bleu_proper', 'comet_qe', 'comet_metric']
METRICS = ['Comet']

ue_metric = PredictionRejectionArea(max_rejection=0.5)

def build_rejection_curve(ues, metrics):
    order = np.argsort(ues)
    sorted_metrics = metrics[order]
    sum_rej_metrics = np.cumsum(sorted_metrics)
    num_points_left = np.arange(1, len(sum_rej_metrics) + 1)

    rej_metrics = sum_rej_metrics / num_points_left
    rej_rates = 1 - num_points_left / len(sum_rej_metrics)

    return rej_metrics[::-1], rej_rates[::-1]

def plot_rejection_curve(raw_ues, detr_ues, metrics, model, dataset, metric):
    path_to_charts = f'charts/{model}{prefix}/{dataset}/{metric}'
    Path(path_to_charts).mkdir(parents=True, exist_ok=True)

    oracle_rejection, rates = build_rejection_curve(-metrics, metrics)
    raw_rejection, rates = build_rejection_curve(raw_ues, metrics)
    detr_rejection, rates = build_rejection_curve(detr_ues, metrics)

    plt.plot(rates, oracle_rejection, label='Oracle')
    plt.plot(rates, raw_rejection, label='Raw')
    plt.plot(rates, detr_rejection, label='Detrended')
    plt.legend()
    plt.xlabel('Rejection Rate')
    plt.ylabel(metric)
    plt.title(f'{model}{prefix} {dataset} {metric}')
    plt.savefig(f'{path_to_charts}/{dataset}_{method.lower()}.png')
    plt.close()

    diff_at_30 = difference_at_rejection_rate(0.3, rates, raw_rejection, detr_rejection)
    diff_at_50 = difference_at_rejection_rate(0.5, rates, raw_rejection, detr_rejection)
    diff_at_70 = difference_at_rejection_rate(0.7, rates, raw_rejection, detr_rejection)

    return diff_at_30, diff_at_50, diff_at_70

def difference_at_rejection_rate(rate, rejection_rates, raw_rejection, detr_rejection):
    closest_rate_id = np.argmin(np.abs(rejection_rates - rate))
    diff = detr_rejection[closest_rate_id] - raw_rejection[closest_rate_id]

    return diff

def score_ues(ues, metric):
    ues_nans = np.isnan(ues)
    metric_nans = np.isnan(metric)
    total_nans = ues_nans | metric_nans

    filtered_ues = ues[~total_nans]
    filtered_metric = metric[~total_nans]

    oracle_score = ue_metric(-filtered_metric, filtered_metric)
    random_score = get_random_scores(ue_metric, filtered_metric)

    raw_ue_metric_val = ue_metric(filtered_ues, filtered_metric)

    raw_score = normalize_metric(raw_ue_metric_val, oracle_score, random_score)

    return raw_score

pathlib.Path('tables').mkdir(parents=True, exist_ok=True)
pathlib.Path('charts').mkdir(parents=True, exist_ok=True)

for model in MODELS:
    for model_type in ['base']:
        prefix = '' if model_type == 'base' else '_instruct'

        for metric in METRICS:
            all_metrics = [metric]
            ue_methods = list(methods_dict.values())
            ue_scores = defaultdict(list)
            coefs = defaultdict(list)
            ue_coefs = defaultdict(list)
            quality_coefs = defaultdict(list)
            diffs_at_30 = defaultdict(list)
            diffs_at_50 = defaultdict(list)
            diffs_at_70 = defaultdict(list)
            ue_scores_raw = defaultdict(list)
            ue_scores_detrend = defaultdict(list)
            ue_scores_diff = defaultdict(list)
            if 'llama' in model:
                datasets = LLAMA_DATASETS
            else:
                datasets = DATASETS

            for dataset in datasets:
                train_ue_values, \
                test_ue_values, \
                train_metric_values, \
                test_metric_values, \
                train_gen_lengths, \
                gen_lengths = extract_and_prepare_data(dataset, methods_dict, all_metrics, model=model, model_type=model_type)
                
                upper_q = np.quantile(train_gen_lengths, 0.95)
                lower_q = np.quantile(train_gen_lengths, 0.05)
                below_q_ids = (train_gen_lengths < upper_q) & (train_gen_lengths > lower_q)
                train_gen_lengths = train_gen_lengths[below_q_ids]
                #for metric in all_metrics:
                #    train_metric_values[metric] = train_metric_values[metric][below_q_ids]
                for method in ue_methods:
                    train_ue_values[method] = train_ue_values[method][below_q_ids]

                train_normalized_metric_values = {}
                test_normalized_metric_values = {}
                train_normalized_ue_values = {}
                test_normalized_ue_values = {}

                ue_residuals = {}
                gen_length_scaler = MinMaxScaler()
                train_gen_lengths_normalized = gen_length_scaler.fit_transform(train_gen_lengths[:, np.newaxis]).squeeze()
                # test_gen_lengths_normalized = gen_length_scaler.transform(gen_lengths[:, np.newaxis]).squeeze()

                scaler = MinMaxScaler()
                train_normalized_metric_values[f"{dataset}_{metric}"] = scaler.fit_transform(train_metric_values[metric][below_q_ids][:, np.newaxis]).squeeze()
                # test_normalized_ue_values[method] = scaler.transform(test_ue_values[method][:, np.newaxis]).squeeze()
                
                linreg = sklearn.linear_model.LinearRegression()
                linreg.fit(train_gen_lengths_normalized[:, np.newaxis], train_normalized_metric_values[f"{dataset}_{metric}"])
                quality_coefs[metric].append(linreg.coef_[0])

                for method in ue_methods:
                    if normalize:
                        gen_length_scaler = MinMaxScaler()
                        train_gen_lengths_normalized = gen_length_scaler.fit_transform(train_gen_lengths[:, np.newaxis]).squeeze()
                        test_gen_lengths_normalized = gen_length_scaler.transform(gen_lengths[:, np.newaxis]).squeeze()

                        scaler = MinMaxScaler()
                        train_normalized_ue_values[method] = scaler.fit_transform(train_ue_values[method][:, np.newaxis]).squeeze()
                        test_normalized_ue_values[method] = scaler.transform(test_ue_values[method][:, np.newaxis]).squeeze()

                        linreg = sklearn.linear_model.LinearRegression()
                        linreg.fit(train_gen_lengths_normalized[:, np.newaxis], train_normalized_ue_values[method])
                        ue_coefs[method].append(linreg.coef_[0])
                        
                        ue_residuals[method] = test_normalized_ue_values[method] - linreg.predict(test_gen_lengths_normalized[:, np.newaxis])
                        scaler = MinMaxScaler()
                        norm_residuals = scaler.fit_transform(ue_residuals[method][:, np.newaxis]).squeeze()
                        linreg = sklearn.linear_model.LinearRegression()
                        linreg.fit(test_gen_lengths_normalized[:, np.newaxis], norm_residuals)
                        # ue_coefs[method].append(linreg.coef_[0])
                        for metric in all_metrics:
                            met_vals = test_metric_values[metric]
                            raw_score = score_ues(test_ue_values[method], met_vals)
                            raw_norm_score = score_ues(test_normalized_ue_values[method], met_vals)
                            detrended_score = score_ues(ue_residuals[method], met_vals)

                            ue_scores[f'{method}_raw'].append(raw_score)
                            ue_scores[f'{method}_detr'].append(detrended_score)
                            ue_scores_raw[method].append(raw_score)
                            ue_scores_detrend[method].append(detrended_score)
                            ue_scores_diff[method].append(detrended_score-raw_score)
                            diff_at_30, diff_at_50, diff_at_70 = plot_rejection_curve(test_normalized_ue_values[method], ue_residuals[method], met_vals, model, dataset, metric)
                            diffs_at_30[method].append(diff_at_30)
                            diffs_at_50[method].append(diff_at_50)
                            diffs_at_70[method].append(diff_at_70)
                    else:
                        linreg = sklearn.linear_model.LinearRegression()
                        linreg.fit(train_gen_lengths[:, np.newaxis], train_ue_values[method])
                        ue_coefs[method].append(linreg.coef_[0])

                        ue_residuals[method] = test_ue_values[method] - linreg.predict(gen_lengths[:, np.newaxis])
                        linreg = sklearn.linear_model.LinearRegression()
                        linreg.fit(gen_lengths[:, np.newaxis], ue_residuals[method])
                        ue_coefs[method].append(linreg.coef_[0])
                        for metric in all_metrics:
                            met_vals = test_metric_values[metric]
                            raw_score = score_ues(test_ue_values[method], met_vals)
                            detrended_score = score_ues(ue_residuals[method], met_vals)

                            ue_scores[f'{method}_raw'].append(raw_score)
                            ue_scores[f'{method}_detr'].append(detrended_score)

                            diff_at_30, diff_at_50, diff_at_70 = plot_rejection_curve(test_ue_values[method], ue_residuals[method], met_vals, model, dataset, metric)
                            diffs_at_30[method].append(diff_at_30)
                            diffs_at_50[method].append(diff_at_50)
                            diffs_at_70[method].append(diff_at_70)

            raw_column_values = []
            detr_column_values = []
            for j, _ in enumerate(datasets):
                #n = len(all_metrics)
                #for i, _ in enumerate(all_metrics):
                #    _id = i + j*n

                _id = j

                raw_column_values.append([ue_scores[f'{method}_raw'][_id] for method in ue_methods])
                detr_column_values.append([ue_scores[f'{method}_detr'][_id] for method in ue_methods])

                metric_raw_scores = np.array([ue_scores[f'{method}_raw'][_id] for method in ue_methods])
                metric_detr_scores = np.array([ue_scores[f'{method}_detr'][_id] for method in ue_methods])

                top_raw_id = np.argmax(metric_raw_scores)
                top_detr_id = np.argmax(metric_detr_scores)

                for method in ue_methods:
                    ue_scores[f'{method}_raw'][_id] = f'{ue_scores[f"{method}_raw"][_id]:.2f}'
                    ue_scores[f'{method}_detr'][_id] = f'{ue_scores[f"{method}_detr"][_id]:.2f}'

                # wrap best detr method in bold
                ue_scores[f'{ue_methods[top_detr_id]}_detr'][_id] = f'\\textbf{{{ue_scores[f"{ue_methods[top_detr_id]}_detr"][_id]}}}'
                # wrap best raw method in underline
                ue_scores[f'{ue_methods[top_raw_id]}_raw'][_id] = f'\\underline{{{ue_scores[f"{ue_methods[top_raw_id]}_raw"][_id]}}}'

            total_column_values = []
            for raw_column, detr_column in zip(raw_column_values, detr_column_values):
                total_column_values.append([val for pair in zip(raw_column, detr_column) for val in pair])

            columns = [f'{dataset}_{metric}' for dataset in datasets for metric in all_metrics] 
            df = pd.DataFrame.from_dict(ue_scores, orient='index', columns=columns)
            # print(ue_scores)
            print(columns)
            print(ue_coefs)
            print(ue_scores_raw)
            print(ue_scores_detrend)
            print(ue_scores_diff)
            print(quality_coefs)


            name = f'tables/{model}{prefix}_{metric}_ue_scores.tex'
            if normalize:
                name = f'tables/{model}{prefix}_{metric}_ue_scores_norm_difference.tex'
            
            with open(name, 'w') as f:
                latex_table = """
                \\begin{table}[h]
                    \\centering
                    \\caption{Performance of UE Metrics across Dataset Groups}
                    \\label{tab:ue_metrics}
                    \\renewcommand{\\arraystretch}{1.2} % Adjust row spacing
                    \\setlength{\\tabcolsep}{4pt}       % Adjust column spacing
    \\begin{tabular}{lcccccccccccccccc}
        \\toprule
        \\multirow{3}{*}{} & \\multicolumn{8}{c}{WMT 14} & \\multicolumn{8}{c}{WMT 19} \\\\
        & \\multicolumn{2}{c}{cs-en} & \\multicolumn{2}{c}{de-en} & \\multicolumn{2}{c}{ru-en}  & \\multicolumn{2}{c}{fr-en} & \\multicolumn{2}{c}{de-en}& \\multicolumn{2}{c}{fi-en} & \\multicolumn{2}{c}{it-en} & \\multicolumn{2}{c}{ru-en} \\\\ """ + """
"""  + " & ".join([f"\\multicolumn{{2}}{{c}}{{{qc:.2f}}}" for qc in quality_coefs[metric]]) + """ \\\\
                        & """ + " & ".join(["Imp. & Reg." for _ in datasets]) + """ \\\\
                        \\midrule
                """
                for ue_metric in ue_coefs.keys():
                    row = f"        {ue_metric} & " + " & ".join(
                        [f"{ue_scores_diff[ue_metric][i]:.2f} & {ue_coefs[ue_metric][i]:.2f}" for i in range(len(ue_coefs[ue_metric]))]
                    ) + " \\\\\n"
                    latex_table += row
                latex_table += """\\bottomrule
                \\end{tabular}
                \\end{table}
                """
                f.write(latex_table)







