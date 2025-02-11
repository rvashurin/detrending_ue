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

normalize = True

methods_dict = {
    'MaximumSequenceProbability': 'MSP',
    'Perplexity': 'PPL',
    'MeanTokenEntropy': 'MTE',
    #'MeanPointwiseMutualInformation': 'MPMI',
    #'MeanConditionalPointwiseMutualInformation': 'MCPMI',
    #'CCP': 'CCP',
    #'PTrue': 'PTrue',
    #'PTrueSampling': 'PTrueS',
    'MonteCarloSequenceEntropy': 'MCSE',
    'MonteCarloNormalizedSequenceEntropy': 'MCNSE',
    #'LexicalSimilarity_rouge1': 'LSR1',
    #'LexicalSimilarity_rouge2': 'LSR2',
    'LexicalSimilarity_rougeL': 'LSRL',
    #'LexicalSimilarity_BLEU': 'LSB',
    #'NumSemSets': 'NSS',
    #'EigValLaplacian_NLI_score_entail': 'ELE',
    #'EigValLaplacian_NLI_score_contra': 'ELC',
    #'EigValLaplacian_Jaccard_score': 'ELJ',
    #'DegMat_NLI_score_entail': 'DME',
    #'DegMat_NLI_score_contra': 'DMC',
    #'DegMat_Jaccard_score': 'DMJ',
    #'Eccentricity_NLI_score_entail': 'EcE',
    #'Eccentricity_NLI_score_contra': 'EcC',
    #'Eccentricity_Jaccard_score': 'EcJ',
    #'SemanticEntropy': 'SE',
    #'SAR': 'SAR',
    #'TokenSAR': 'TSAR',
    #'SentenceSAR': 'SSAR',
    #'RenyiNeg': 'RN',
    #'FisherRao': 'FR',
}

#MODELS = ['llama', 'mistral7b', 'stablelm12b']
MODELS = ['llama1b']
LLAMA_DATASETS = [
    'wmt14_csen',
#    'wmt14_deen',
    'wmt14_ruen',
#    'wmt14_fren',
#    'wmt19_deen',
#    'wmt19_fien',
#    'wmt19_lten',
#    'wmt19_ruen',
]
DATASETS = []
#    'wmt14',
#    'wmt19',
#]

#METRICS = ['Comet', 'bleu_proper', 'comet_qe', 'comet_metric']
METRICS = ['Comet', 'bleu_proper']

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
    path_to_charts = f'charts/{model}/{dataset}/{metric}'
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
    plt.title(f'{model} {dataset} {metric}')
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
    for metric in METRICS:
        all_metrics = [metric]
        ue_methods = list(methods_dict.values())
        ue_scores = defaultdict(list)
        coefs = defaultdict(list)
        ue_coefs = defaultdict(list)
        diffs_at_30 = defaultdict(list)
        diffs_at_50 = defaultdict(list)
        diffs_at_70 = defaultdict(list)

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
            gen_lengths = extract_and_prepare_data(dataset, methods_dict, all_metrics, model=model)
            
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
            #for metric in all_metrics:
            #    if normalize:
            #        gen_length_scaler = MinMaxScaler()
            #        train_gen_lengths_normalized = gen_length_scaler.fit_transform(train_gen_lengths[:, np.newaxis]).squeeze()
            #        test_gen_lengths_normalized = gen_length_scaler.transform(gen_lengths[:, np.newaxis]).squeeze()

            #        scaler = MinMaxScaler()
            #        train_normalized_metric_values[metric] = scaler.fit_transform(train_metric_values[metric][:, np.newaxis]).squeeze()
            #        linreg = sklearn.linear_model.LinearRegression()
            #        linreg.fit(train_gen_lengths_normalized[:, np.newaxis], train_normalized_metric_values[metric])
            #        coefs[metric].append(linreg.coef_[0])

            #        test_normalized_metric_values[metric] = scaler.transform(test_metric_values[metric][:, np.newaxis]).squeeze()

            #        linreg = sklearn.linear_model.LinearRegression()
            #        linreg.fit(test_gen_lengths_normalized[:, np.newaxis], test_normalized_metric_values[metric])
            #        coefs[metric].append(linreg.coef_[0])
            #    else:
            #        linreg = sklearn.linear_model.LinearRegression()
            #        linreg.fit(train_gen_lengths[:, np.newaxis], train_metric_values[metric])
            #        coefs[metric].append(linreg.coef_[0])

            #        linreg = sklearn.linear_model.LinearRegression()
            #        linreg.fit(gen_lengths[:, np.newaxis], test_metric_values[metric])
            #        coefs[metric].append(linreg.coef_[0])

            train_normalized_ue_values = {}
            test_normalized_ue_values = {}

            ue_residuals = {}

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
                    ue_coefs[method].append(linreg.coef_[0])
                    for metric in all_metrics:
                        met_vals = test_metric_values[metric]
                        raw_score = score_ues(test_ue_values[method], met_vals)
                        raw_norm_score = score_ues(test_normalized_ue_values[method], met_vals)
                        detrended_score = score_ues(ue_residuals[method], met_vals)

                        ue_scores[f'{method}_raw'].append(raw_score)
                        ue_scores[f'{method}_detr'].append(detrended_score)

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

        raw_method_id_ranks = np.flip(np.argsort(raw_column_values, axis=-1), axis=-1)
        raw_mean_ranks = [np.nonzero(raw_method_id_ranks == method_i)[1].mean() for method_i, _ in enumerate(ue_methods)]

        detr_method_id_ranks = np.flip(np.argsort(detr_column_values, axis=-1), axis=-1)
        detr_mean_ranks = [np.nonzero(detr_method_id_ranks == method_i)[1].mean() for method_i, _ in enumerate(ue_methods)]

        total_method_id_ranks = np.flip(np.argsort(total_column_values, axis=-1), axis=-1)
        total_mean_ranks = [np.nonzero(total_method_id_ranks == method_i)[1].mean() for method_i, _ in enumerate(ue_methods * 2)]

        for method_i, method in enumerate(ue_methods):
            ue_scores[f'{method}_raw'].extend((str(raw_mean_ranks[method_i]), '-', total_mean_ranks[method_i * 2]))
            ue_scores[f'{method}_detr'].extend(('-', str(detr_mean_ranks[method_i]), total_mean_ranks[method_i * 2 + 1]))

        columns = [f'{dataset}_{metric}' for dataset in datasets for metric in all_metrics] + ['raw_rank', 'detr_rank', 'rank']
        df = pd.DataFrame.from_dict(ue_scores, orient='index', columns=columns)
        name = f'tables/{model}_{metric}_ue_scores.tex'
        if normalize:
            name = f'tables/{model}_{metric}_ue_scores_norm.tex'
        with open(name, 'w') as f:
            latex = df.to_latex(float_format="%.2f", escape=False)
            latex = latex.replace('_', '\_')
            # find first line where \midrule is present
            start_id = latex.split('\n').index('\\midrule') + 2
            # add \midrule every third line starting from start_id
            latex = '\n'.join([line if i % 2 != 0 else line + '\n\\midrule' for i, line in enumerate(latex.split('\n'), start=start_id)])
            f.write(latex)

        columns = [f'{dataset}_{metric}' for dataset in datasets for metric in all_metrics]

        df = pd.DataFrame.from_dict(diffs_at_30, orient='index', columns=columns)
        name = f'tables/{model}_{metric}_ue_rej_diffs_at_30.tex'
        with open(name, 'w') as f:
            latex = df.to_latex(float_format="%.2f", escape=False)
            latex = latex.replace('_', '\_')
            f.write(latex)
        df = pd.DataFrame.from_dict(diffs_at_50, orient='index', columns=columns)
        name = f'tables/{model}_{metric}_ue_rej_diffs_at_50.tex'
        with open(name, 'w') as f:
            latex = df.to_latex(float_format="%.2f", escape=False)
            latex = latex.replace('_', '\_')
            f.write(latex)
        df = pd.DataFrame.from_dict(diffs_at_70, orient='index', columns=columns)
        name = f'tables/{model}_{metric}_ue_rej_diffs_at_70.tex'
        with open(name, 'w') as f:
            latex = df.to_latex(float_format="%.2f", escape=False)
            latex = latex.replace('_', '\_')
            f.write(latex)



        columns = []
        for dataset in datasets:
            for coef_type in ['train_c', 'test_c']:
                columns.append(f'{dataset}_{coef_type}')

        #df = pd.DataFrame.from_dict(coefs, orient='index', columns=columns)
        #name = f'tables/{model}_{metric}_metric_trends.tex'
        #if normalize:
        #    name = f'tables/{model}_{metric}_metric_trends_norm.tex'
        #with open(name, 'w') as f:
        #    latex = df.to_latex(float_format="%.3f", escape=False)
        #    latex = latex.replace('_', '\_')
        #    f.write(latex)

        df = pd.DataFrame.from_dict(ue_coefs, orient='index', columns=columns)
        name = f'tables/{model}_{metric}_ue_trends.tex'
        if normalize:
            name = f'tables/{model}_{metric}_ue_trends_norm.tex'

        with open(name, 'w') as f:
            latex = df.to_latex(float_format="%.3f", escape=False)
            latex = latex.replace('_', '\_')
            f.write(latex)
