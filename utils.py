import numpy as np
import matplotlib.pyplot as plt
from lm_polygraph.utils.manager import UEManager
import sklearn
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
import logging
from lm_polygraph.ue_metrics.pred_rej_area import PredictionRejectionArea
from lm_polygraph.ue_metrics.ue_metric import (
    get_random_scores,
    normalize_metric,
)

ue_metric = PredictionRejectionArea(max_rejection=0.5)

log = logging.getLogger("lm_polygraph")
log.setLevel(logging.ERROR)


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


def load_managers(dataset, model='llama', model_type='base', task='nmt'):
    prefix = '' if model_type == 'base' else '_instruct'
    if task == 'nmt':
        manager = UEManager.load(f'processed_mans/{model}{prefix}_{dataset}_test_qe_enriched_processed.man')
        train_manager = UEManager.load(f'mans/{model}{prefix}_{dataset}_train_qe_enriched.man')
    else:
        manager = UEManager.load(f'mans/{model}{prefix}_{dataset}_test.man')
        train_manager = UEManager.load(f'mans/{model}{prefix}_{dataset}_train.man')

    return manager, train_manager


def extract_and_prepare_data(dataset, methods_dict, all_metrics, model='llama', model_type='base', task='nmt'):
    manager, train_manager = load_managers(dataset, model, model_type, task)

    full_ue_methods = list(methods_dict.keys())
    ue_methods = list(methods_dict.values())

    sequences = manager.stats['greedy_tokens']
    texts = manager.stats['greedy_texts']
    targets = manager.stats['target_texts']

    train_sequences = train_manager.stats['greedy_tokens']
    train_texts = train_manager.stats['greedy_texts']
    train_targets = train_manager.stats['target_texts']

    train_gen_lengths = np.array([len(seq) for seq in train_sequences])
    gen_lengths = np.array([len(seq) for seq in sequences])

    # Get train and test values for metrics and UE, remove union of nans
    test_nans = []
    train_nans = []

    train_metric_values = {}
    test_metric_values = {}
    for metric in all_metrics:
        values = np.array(manager.gen_metrics[('sequence', metric)])
        test_metric_values[metric] = np.array(values)
        test_nans.extend(np.argwhere(np.isnan(values)).flatten())

        train_values = np.array(train_manager.gen_metrics[('sequence', metric)])
        train_metric_values[metric] = np.array(train_values)
        train_nans.extend(np.argwhere(np.isnan(train_values)).flatten())

    train_ue_values = {}
    test_ue_values = {}
    for i, method in enumerate(full_ue_methods):
        train_values = np.array(train_manager.estimations[('sequence', method)])
        train_ue_values[ue_methods[i]] = train_values
        train_nans.extend(np.argwhere(np.isnan(train_values)).flatten())

        values = np.array(manager.estimations[('sequence', method)])
        test_ue_values[ue_methods[i]] = values
        test_nans.extend(np.argwhere(np.isnan(values)).flatten())

    train_nans = np.unique(train_nans).astype(int)
    test_nans = np.unique(test_nans).astype(int)

    # Remove nans
    for metric in all_metrics:
        test_metric_values[metric] = np.delete(test_metric_values[metric], test_nans)
        train_metric_values[metric] = np.delete(train_metric_values[metric], train_nans)

    for method in ue_methods:
        test_ue_values[method] = np.delete(test_ue_values[method], test_nans)
        train_ue_values[method] = np.delete(train_ue_values[method], train_nans)

    train_gen_lengths = np.delete(train_gen_lengths, train_nans)
    gen_lengths = np.delete(gen_lengths, test_nans)

    return train_ue_values, test_ue_values, train_metric_values, test_metric_values, train_gen_lengths, gen_lengths


def detrend_ue(datasets, model, model_type, all_metrics, ue_methods, methods_dict, task='nmt'):
    ue_scores = defaultdict(list)
    ue_coefs = defaultdict(list)
    ave_test_metric_values = {}

    if len(all_metrics) == 1 and len(datasets) > 1:
        all_metrics = all_metrics * len(datasets)
    elif len(all_metrics) != len(datasets):
        raise ValueError('Number of metrics and datasets must be the same')

    for metric, dataset in zip(all_metrics, datasets):
        train_ue_values, \
        test_ue_values, \
        train_metric_values, \
        test_metric_values, \
        train_gen_lengths, \
        gen_lengths = extract_and_prepare_data(dataset, methods_dict, [metric], model=model, model_type=model_type, task=task)

        ave_test_metric_values[dataset] = np.mean(test_metric_values[metric])

        upper_q = np.quantile(train_gen_lengths, 0.95)
        lower_q = np.quantile(train_gen_lengths, 0.05)
        below_q_ids = (train_gen_lengths < upper_q) & (train_gen_lengths > lower_q)
        print(f'{model} {dataset} Below q ids: {below_q_ids.sum()}')
        train_gen_lengths = train_gen_lengths[below_q_ids]

        for method in ue_methods:
            train_ue_values[method] = train_ue_values[method][below_q_ids]

        train_normalized_ue_values = {}
        test_normalized_ue_values = {}

        ue_residuals = {}

        for method in ue_methods:
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

            met_vals = test_metric_values[metric]
            raw_score = score_ues(test_ue_values[method], met_vals)
            raw_norm_score = score_ues(test_normalized_ue_values[method], met_vals)
            detrended_score = score_ues(ue_residuals[method], met_vals)

            ue_scores[f'{method}_raw'].append(raw_score)
            ue_scores[f'{method}_detr'].append(detrended_score)


    raw_column_values = []
    detr_column_values = []
    for _id, _ in enumerate(datasets):
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

    return ue_scores, ue_coefs, ave_test_metric_values
