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
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score

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
        manager = UEManager.load(f'processed_mans/{model}{prefix}_{dataset}_test_full_enriched.man')
        train_manager = UEManager.load(f'processed_mans/{model}{prefix}_{dataset}_train_full_enriched.man')
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


# def detrend_ue(datasets, model, model_type, all_metrics, ue_methods, methods_dict, task='nmt', return_unprocessed=False):
#     ue_scores = defaultdict(list)
#     ue_coefs = defaultdict(list)
#     ave_test_metric_values = {}

#     if len(all_metrics) == 1 and len(datasets) > 1:
#         all_metrics = all_metrics * len(datasets)
#     elif len(all_metrics) != len(datasets):
#         raise ValueError('Number of metrics and datasets must be the same')

#     for metric, dataset in zip(all_metrics, datasets):
#         train_ue_values, \
#         test_ue_values, \
#         train_metric_values, \
#         test_metric_values, \
#         train_gen_lengths, \
#         gen_lengths = extract_and_prepare_data(dataset, methods_dict, [metric], model=model, model_type=model_type, task=task)

#         ave_test_metric_values[dataset] = np.mean(test_metric_values[metric])

#         upper_q = np.quantile(train_gen_lengths, 0.95)
#         lower_q = np.quantile(train_gen_lengths, 0.05)
#         below_q_ids = (train_gen_lengths < upper_q) & (train_gen_lengths > lower_q)
#         print(f'{model} {dataset} Below q ids: {below_q_ids.sum()}')
#         train_gen_lengths = train_gen_lengths[below_q_ids]

#         for method in ue_methods:
#             train_ue_values[method] = train_ue_values[method][below_q_ids]

#         train_normalized_ue_values = {}
#         test_normalized_ue_values = {}

#         ue_residuals = {}

#         for method in ue_methods:
#             gen_length_scaler = MinMaxScaler()
#             train_gen_lengths_normalized = gen_length_scaler.fit_transform(train_gen_lengths[:, np.newaxis]).squeeze()
#             test_gen_lengths_normalized = gen_length_scaler.transform(gen_lengths[:, np.newaxis]).squeeze()

#             scaler = MinMaxScaler()
#             train_normalized_ue_values[method] = scaler.fit_transform(train_ue_values[method][:, np.newaxis]).squeeze()
#             test_normalized_ue_values[method] = scaler.transform(test_ue_values[method][:, np.newaxis]).squeeze()

#             linreg = sklearn.linear_model.LinearRegression()
#             linreg.fit(train_gen_lengths_normalized[:, np.newaxis], train_normalized_ue_values[method])
#             ue_coefs[method].append(linreg.coef_[0])

#             ue_residuals[method] = test_normalized_ue_values[method] - linreg.predict(test_gen_lengths_normalized[:, np.newaxis])
#             scaler = MinMaxScaler()
#             norm_residuals = scaler.fit_transform(ue_residuals[method][:, np.newaxis]).squeeze()
#             linreg = sklearn.linear_model.LinearRegression()
#             linreg.fit(test_gen_lengths_normalized[:, np.newaxis], norm_residuals)
#             ue_coefs[method].append(linreg.coef_[0])

#             met_vals = test_metric_values[metric]
#             raw_score = score_ues(test_ue_values[method], met_vals)
#             raw_norm_score = score_ues(test_normalized_ue_values[method], met_vals)
#             detrended_score = score_ues(ue_residuals[method], met_vals)

#             ue_scores[f'{method}_raw'].append(raw_score)
#             ue_scores[f'{method}_detr'].append(detrended_score)
#             ue_scores[f'{method}_raw_full'].append(test_normalized_ue_values[method])
#             ue_scores[f'{method}_detr_full'].append(ue_residuals[method])


#     if return_unprocessed:
#         return ue_scores, ue_coefs, ave_test_metric_values,test_gen_lengths_normalized[:, np.newaxis]

#     raw_column_values = []
#     detr_column_values = []
#     for _id, _ in enumerate(datasets):
#         raw_column_values.append([ue_scores[f'{method}_raw'][_id] for method in ue_methods])
#         detr_column_values.append([ue_scores[f'{method}_detr'][_id] for method in ue_methods])

#         metric_raw_scores = np.array([ue_scores[f'{method}_raw'][_id] for method in ue_methods])
#         metric_detr_scores = np.array([ue_scores[f'{method}_detr'][_id] for method in ue_methods])

#         top_raw_id = np.argmax(metric_raw_scores)
#         top_detr_id = np.argmax(metric_detr_scores)

#         for method in ue_methods:
#             ue_scores[f'{method}_raw'][_id] = f'{ue_scores[f"{method}_raw"][_id]:.2f}'
#             ue_scores[f'{method}_detr'][_id] = f'{ue_scores[f"{method}_detr"][_id]:.2f}'

#         # wrap best detr method in bold
#         ue_scores[f'{ue_methods[top_detr_id]}_detr'][_id] = f'\\textbf{{{ue_scores[f"{ue_methods[top_detr_id]}_detr"][_id]}}}'
#         # wrap best raw method in underline
#         ue_scores[f'{ue_methods[top_raw_id]}_raw'][_id] = f'\\underline{{{ue_scores[f"{ue_methods[top_raw_id]}_raw"][_id]}}}'

#     total_column_values = []
#     for raw_column, detr_column in zip(raw_column_values, detr_column_values):
#         total_column_values.append([val for pair in zip(raw_column, detr_column) for val in pair])

#     raw_method_id_ranks = np.flip(np.argsort(raw_column_values, axis=-1), axis=-1)
#     raw_mean_ranks = [np.nonzero(raw_method_id_ranks == method_i)[1].mean() for method_i, _ in enumerate(ue_methods)]

#     detr_method_id_ranks = np.flip(np.argsort(detr_column_values, axis=-1), axis=-1)
#     detr_mean_ranks = [np.nonzero(detr_method_id_ranks == method_i)[1].mean() for method_i, _ in enumerate(ue_methods)]

#     total_method_id_ranks = np.flip(np.argsort(total_column_values, axis=-1), axis=-1)
#     total_mean_ranks = [np.nonzero(total_method_id_ranks == method_i)[1].mean() for method_i, _ in enumerate(ue_methods * 2)]

#     for method_i, method in enumerate(ue_methods):
#         ue_scores[f'{method}_raw'].extend((str(raw_mean_ranks[method_i]), '-', total_mean_ranks[method_i * 2]))
#         ue_scores[f'{method}_detr'].extend(('-', str(detr_mean_ranks[method_i]), total_mean_ranks[method_i * 2 + 1]))
    
#     return ue_scores, ue_coefs, ave_test_metric_values



def detrend_ue_w_quality(datasets, model, model_type, all_metrics, ue_methods, methods_dict, task='nmt', return_unprocessed=False, quality_fit_sample_size=None):
    ue_scores = defaultdict(list)
    ue_scores_full = {}
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

        train_normalized_metric_values = {}
        test_normalized_metric_values = {}
        ue_residuals = {}


        for method in ue_methods:
            gen_length_scaler = MinMaxScaler()
            train_gen_lengths_normalized = gen_length_scaler.fit_transform(train_gen_lengths[:, np.newaxis]).squeeze()
            test_gen_lengths_normalized = gen_length_scaler.transform(gen_lengths[:, np.newaxis]).squeeze()

            scaler = MinMaxScaler()
            train_normalized_ue_values[method] = scaler.fit_transform(train_ue_values[method][:, np.newaxis]).squeeze()
            test_normalized_ue_values[method] = scaler.transform(test_ue_values[method][:, np.newaxis]).squeeze()

            scaler = MinMaxScaler()
            train_normalized_metric_values[method] = scaler.fit_transform(train_metric_values[metric][:, np.newaxis]).squeeze()
            test_normalized_metric_values[method] = scaler.transform(test_metric_values[metric][:, np.newaxis]).squeeze()

            # quality_reg = sklearn.linear_model.LinearRegression()
            # quality_reg.fit(train_gen_lengths_normalized[:, np.newaxis], train_normalized_metric_values[method][below_q_ids])
            # quality_slope = quality_reg.coef_[0]

            # if quality_fit_sample_size is not None and quality_fit_sample_size < len(train_gen_lengths):
            #     sample_indices = np.random.choice(len(train_gen_lengths), size=quality_fit_sample_size, replace=False)
            #     quality_reg = sklearn.linear_model.LinearRegression()
            #     quality_reg.fit(
            #         train_gen_lengths_normalized[sample_indices, np.newaxis],
            #         train_normalized_metric_values[method][sample_indices][below_q_ids]
            #     )
            if quality_fit_sample_size is not None and quality_fit_sample_size < len(train_gen_lengths):
                # Filter to only non-outliers
                # filtered_lengths = train_gen_lengths_normalized
                filtered_metrics = train_normalized_metric_values[method][below_q_ids]
                filtered_gen_lengths_normalized = train_gen_lengths_normalized

                # Adaptive binning using KMeans
                n_bins = 10
                est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='kmeans')
                bin_ids = est.fit_transform(filtered_gen_lengths_normalized.reshape(-1, 1)).astype(int).squeeze()

                sample_per_bin = quality_fit_sample_size // n_bins
                stratified_indices = []

                for bin_id in np.unique(bin_ids):
                    bin_indices = np.where(bin_ids == bin_id)[0]
                    n = min(sample_per_bin, len(bin_indices))
                    if n > 0:
                        stratified_indices.extend(np.random.choice(bin_indices, size=n, replace=False))

                stratified_indices = np.array(stratified_indices)

                # Fit quality regression on stratified sample from non-outliers
                quality_reg = sklearn.linear_model.LinearRegression()
                quality_reg.fit(
                    filtered_gen_lengths_normalized[stratified_indices, np.newaxis],
                    filtered_metrics[stratified_indices]
                )


            else:
                quality_reg = sklearn.linear_model.LinearRegression()
                quality_reg.fit(
                    train_gen_lengths_normalized[:, np.newaxis],
                    train_normalized_metric_values[method][below_q_ids]
                )

            # Fit UE ~ length
            linreg = sklearn.linear_model.LinearRegression()
            linreg.fit(train_gen_lengths_normalized[:, np.newaxis], train_normalized_ue_values[method])
            ue_slope = linreg.coef_[0]
            ue_coefs[method].append(ue_slope)

            predicted_quality_trend = quality_reg.predict(test_gen_lengths_normalized[:, np.newaxis])
            # Predict UE trend on test
            predicted_ue_trend = linreg.predict(test_gen_lengths_normalized[:, np.newaxis])


            # joint_reg = sklearn.linear_model.LinearRegression()
            # joint_reg.fit(np.stack([train_gen_lengths_normalized, train_normalized_metric_values[method]], axis=1),
                        # train_normalized_ue_values[method])
            # predicted_trend = joint_reg.predict(np.stack([test_gen_lengths_normalized, test_normalized_metric_values[method]], axis=1))
            # adjusted_ue = test_normalized_ue_values[method] - predicted_trend

            # Residual-based adjustment
            adjusted_ue = test_normalized_ue_values[method] - predicted_ue_trend - predicted_quality_trend

            residual_reg = sklearn.linear_model.LinearRegression()
            residual_reg.fit(test_gen_lengths_normalized[:, np.newaxis], adjusted_ue)
            ue_coefs[method].append(residual_reg.coef_[0])

            met_vals = test_metric_values[metric]
            raw_score = score_ues(test_ue_values[method], met_vals)
            raw_norm_score = score_ues(test_normalized_ue_values[method], met_vals)
            detrended_score = score_ues(adjusted_ue, met_vals)
            ue_scores_full[f'{method}_raw'] = test_normalized_ue_values[method]
            ue_scores_full[f'{method}_detr'] = adjusted_ue
            ue_scores[f'{method}_raw'].append(raw_score)
            ue_scores[f'{method}_detr'].append(detrended_score)
            ue_scores[f'{method}_raw_full'].append(test_normalized_ue_values[method])
            ue_scores[f'{method}_detr_full'].append(adjusted_ue)



    normalized_lengths= test_gen_lengths_normalized.tolist()

    if return_unprocessed:
        return ue_scores, ue_coefs, ave_test_metric_values,  test_gen_lengths_normalized[:, np.newaxis] , test_normalized_metric_values[method]

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
    
    return ue_scores, ue_coefs, ave_test_metric_values, normalized_lengths




def detrend_ue(datasets, model, model_type, all_metrics, ue_methods, methods_dict, task='nmt', return_unprocessed=False):
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

    if return_unprocessed:
        return ue_scores, ue_coefs, ave_test_metric_values

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


def detrend_ue_w_quality_only(datasets, model, model_type, all_metrics, ue_methods, methods_dict, task='nmt', return_unprocessed=False):
    ue_scores = defaultdict(list)
    ue_scores_full = {}
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

        train_normalized_metric_values = {}
        test_normalized_metric_values = {}
        ue_residuals = {}


        for method in ue_methods:
            gen_length_scaler = MinMaxScaler()
            train_gen_lengths_normalized = gen_length_scaler.fit_transform(train_gen_lengths[:, np.newaxis]).squeeze()
            test_gen_lengths_normalized = gen_length_scaler.transform(gen_lengths[:, np.newaxis]).squeeze()

            scaler = MinMaxScaler()
            train_normalized_ue_values[method] = scaler.fit_transform(train_ue_values[method][:, np.newaxis]).squeeze()
            test_normalized_ue_values[method] = scaler.transform(test_ue_values[method][:, np.newaxis]).squeeze()

            scaler = MinMaxScaler()
            train_normalized_metric_values[method] = scaler.fit_transform(train_metric_values[metric][:, np.newaxis]).squeeze()
            test_normalized_metric_values[method] = scaler.transform(test_metric_values[metric][:, np.newaxis]).squeeze()

            quality_reg = sklearn.linear_model.LinearRegression()
            quality_reg.fit(train_gen_lengths_normalized[:, np.newaxis], train_normalized_metric_values[method][below_q_ids])
            quality_slope = quality_reg.coef_[0]

            # Fit UE ~ length
            linreg = sklearn.linear_model.LinearRegression()
            linreg.fit(train_gen_lengths_normalized[:, np.newaxis], train_normalized_ue_values[method])
            ue_slope = linreg.coef_[0]
            ue_coefs[method].append(ue_slope)

            predicted_quality_trend = quality_reg.predict(test_gen_lengths_normalized[:, np.newaxis])
            # Predict UE trend on test
            predicted_ue_trend = linreg.predict(test_gen_lengths_normalized[:, np.newaxis])


            # joint_reg = sklearn.linear_model.LinearRegression()
            # joint_reg.fit(np.stack([train_gen_lengths_normalized, train_normalized_metric_values[method]], axis=1),
                        # train_normalized_ue_values[method])
            # predicted_trend = joint_reg.predict(np.stack([test_gen_lengths_normalized, test_normalized_metric_values[method]], axis=1))
            # adjusted_ue = test_normalized_ue_values[method] - predicted_trend

            # Residual-based adjustment
            adjusted_ue = - predicted_quality_trend

            residual_reg = sklearn.linear_model.LinearRegression()
            residual_reg.fit(test_gen_lengths_normalized[:, np.newaxis], adjusted_ue)
            ue_coefs[method].append(residual_reg.coef_[0])

            met_vals = test_metric_values[metric]
            raw_score = score_ues(test_ue_values[method], met_vals)
            raw_norm_score = score_ues(test_normalized_ue_values[method], met_vals)
            detrended_score = score_ues(adjusted_ue, met_vals)
            ue_scores_full[f'{method}_raw'] = test_normalized_ue_values[method]
            ue_scores_full[f'{method}_detr'] = adjusted_ue
            ue_scores[f'{method}_raw'].append(raw_score)
            ue_scores[f'{method}_detr'].append(detrended_score)
            ue_scores[f'{method}_raw_full'].append(test_normalized_ue_values[method])
            ue_scores[f'{method}_detr_full'].append(adjusted_ue)



    normalized_lengths= test_gen_lengths_normalized.tolist()

    if return_unprocessed:
        return ue_scores, ue_coefs, ave_test_metric_values,  test_gen_lengths_normalized[:, np.newaxis] , test_normalized_metric_values[method]

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
    
    return ue_scores, ue_coefs, ave_test_metric_values, normalized_lengths


def summarize_quality_fit(datasets, model, model_type, all_metrics, ue_methods, methods_dict, task='nmt', quality_fit_sample_size=None):
    summary_stats = defaultdict(lambda: defaultdict(dict))

    if len(all_metrics) == 1 and len(datasets) > 1:
        all_metrics = all_metrics * len(datasets)
    elif len(all_metrics) != len(datasets):
        raise ValueError('Number of metrics and datasets must be the same')

    for metric, dataset in zip(all_metrics, datasets):
        train_ue_values, _, train_metric_values, _, train_gen_lengths, _ = extract_and_prepare_data(
            dataset, methods_dict, [metric], model=model, model_type=model_type, task=task
        )

        upper_q = np.quantile(train_gen_lengths, 0.95)
        lower_q = np.quantile(train_gen_lengths, 0.05)
        below_q_ids = (train_gen_lengths < upper_q) & (train_gen_lengths > lower_q)
        train_gen_lengths = train_gen_lengths[below_q_ids]

        for method in ue_methods:
            train_ue = train_ue_values[method][below_q_ids]
            train_quality = train_metric_values[metric][below_q_ids]

            length_scaler = MinMaxScaler()
            train_gen_lengths_norm = length_scaler.fit_transform(train_gen_lengths[:, np.newaxis]).squeeze()

            metric_scaler = MinMaxScaler()
            train_quality_norm = metric_scaler.fit_transform(train_quality[:, np.newaxis]).squeeze()

            ue_scaler = MinMaxScaler()
            train_ue_norm = ue_scaler.fit_transform(train_ue[:, np.newaxis]).squeeze()

            # Sample (optional)
            if quality_fit_sample_size is not None and quality_fit_sample_size < len(train_gen_lengths):
                # Stratified sampling based on generation lengths
                n_bins = 10
                est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='kmeans')
                bin_ids = est.fit_transform(train_gen_lengths_norm.reshape(-1, 1)).astype(int).squeeze()
                sample_per_bin = quality_fit_sample_size // n_bins
                stratified_indices = []

                for bin_id in np.unique(bin_ids):
                    bin_indices = np.where(bin_ids == bin_id)[0]
                    n = min(sample_per_bin, len(bin_indices))
                    if n > 0:
                        stratified_indices.extend(np.random.choice(bin_indices, size=n, replace=False))
                indices = np.array(stratified_indices)
            else:
                indices = np.arange(len(train_gen_lengths_norm))

            X = train_gen_lengths_norm[indices].reshape(-1, 1)
            y_quality = train_quality_norm[indices]
            y_ue = train_ue_norm[indices]

            # Fit quality regression
            reg_quality = sklearn.linear_model.LinearRegression().fit(X, y_quality)
            preds_quality = reg_quality.predict(X)
            r2_quality = r2_score(y_quality, preds_quality)
            mse_quality = mean_squared_error(y_quality, preds_quality)

            # Fit UE regression
            reg_ue = sklearn.linear_model.LinearRegression().fit(X, y_ue)
            preds_ue = reg_ue.predict(X)
            r2_ue = r2_score(y_ue, preds_ue)
            mse_ue = mean_squared_error(y_ue, preds_ue)

            # Store summary
            summary_stats[dataset][method] = {
                'n_train_points': len(indices),
                'quality_r2': round(r2_quality, 4),
                'quality_mse': round(mse_quality, 4),
                'quality_coef': round(reg_quality.coef_[0], 4),
                'quality_intercept': round(reg_quality.intercept_, 4),
                'ue_r2': round(r2_ue, 4),
                'ue_mse': round(mse_ue, 4),
                'ue_coef': round(reg_ue.coef_[0], 4),
                'ue_intercept': round(reg_ue.intercept_, 4),
            }

    return summary_stats


def summarize_ue_fit(datasets, model, model_type, all_metrics, ue_methods, methods_dict, task='nmt'):
    summary_stats = defaultdict(lambda: defaultdict(dict))

    if len(all_metrics) == 1 and len(datasets) > 1:
        all_metrics = all_metrics * len(datasets)
    elif len(all_metrics) != len(datasets):
        raise ValueError('Number of metrics and datasets must be the same')

    for metric, dataset in zip(all_metrics, datasets):
        train_ue_values, _, _, _, train_gen_lengths, _ = extract_and_prepare_data(
            dataset, methods_dict, [metric], model=model, model_type=model_type, task=task
        )

        upper_q = np.quantile(train_gen_lengths, 0.95)
        lower_q = np.quantile(train_gen_lengths, 0.05)
        below_q_ids = (train_gen_lengths < upper_q) & (train_gen_lengths > lower_q)
        train_gen_lengths = train_gen_lengths[below_q_ids]

        for method in ue_methods:
            train_ue = train_ue_values[method][below_q_ids]

            length_scaler = MinMaxScaler()
            train_gen_lengths_norm = length_scaler.fit_transform(train_gen_lengths[:, np.newaxis]).squeeze()

            ue_scaler = MinMaxScaler()
            train_ue_norm = ue_scaler.fit_transform(train_ue[:, np.newaxis]).squeeze()

            X = train_gen_lengths_norm.reshape(-1, 1)
            y = train_ue_norm

            linreg = sklearn.linear_model.LinearRegression().fit(X, y)
            y_pred = linreg.predict(X)

            summary_stats[dataset][method] = {
                'n_train_points': len(y),
                'ue_r2': round(r2_score(y, y_pred), 4),
                'ue_mse': round(mean_squared_error(y, y_pred), 4),
                'ue_coef': round(linreg.coef_[0], 4),
                'ue_intercept': round(linreg.intercept_, 4)
            }

    return summary_stats


def temp_scale_ue(datasets, model, model_type, all_metrics, ue_methods, methods_dict, task='nmt'):
    from scipy.optimize import minimize_scalar

    ue_scores = defaultdict(list)
    best_temperatures = defaultdict(list)
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
        _, _ = extract_and_prepare_data(dataset, methods_dict, [metric], model=model, model_type=model_type, task=task)

        ave_test_metric_values[dataset] = np.mean(test_metric_values[metric])

        for method in ue_methods:
            train_ue = train_ue_values[method]
            train_metric = train_metric_values[metric]
            test_ue = test_ue_values[method]
            test_metric = test_metric_values[metric]

            # Optional: normalize train/test UE
            train_ue = MinMaxScaler().fit_transform(train_ue[:, None]).squeeze()
            test_ue = MinMaxScaler().fit_transform(test_ue[:, None]).squeeze()

            # Objective: minimize negative PRR (or whatever score_ues returns)
            def objective(temp):
                scaled = train_ue / temp
                return -score_ues(scaled, train_metric)

            res = minimize_scalar(objective, bounds=(0.1, 5.0), method='bounded')
            best_T = res.x
            best_temperatures[method].append(best_T)

            # Apply temperature scaling to test
            scaled_test_ue = test_ue / best_T
            scaled_score = score_ues(scaled_test_ue, test_metric)

            ue_scores[f'{method}_raw'].append(score_ues(test_ue, test_metric))
            ue_scores[f'{method}_temp'].append(scaled_score)

    # Format table like in detrend_ue
    raw_column_values = []
    temp_column_values = []

    for i in range(len(datasets)):
        raw_column_values.append([ue_scores[f'{method}_raw'][i] for method in ue_methods])
        temp_column_values.append([ue_scores[f'{method}_temp'][i] for method in ue_methods])

        metric_raw_scores = np.array(raw_column_values[-1])
        metric_temp_scores = np.array(temp_column_values[-1])

        top_raw = np.argmax(metric_raw_scores)
        top_temp = np.argmax(metric_temp_scores)

        for method in ue_methods:
            ue_scores[f'{method}_raw'][i] = f'{ue_scores[f"{method}_raw"][i]:.2f}'
            ue_scores[f'{method}_temp'][i] = f'{ue_scores[f"{method}_temp"][i]:.2f}'

        ue_scores[f'{ue_methods[top_raw]}_raw'][i] = f'\\underline{{{ue_scores[f"{ue_methods[top_raw]}_raw"][i]}}}'
        ue_scores[f'{ue_methods[top_temp]}_temp'][i] = f'\\textbf{{{ue_scores[f"{ue_methods[top_temp]}_temp"][i]}}}'

    # Rankings (optional)
    total_column_values = []
    for raw, temp in zip(raw_column_values, temp_column_values):
        total_column_values.append([val for pair in zip(raw, temp) for val in pair])

    raw_ranks = np.flip(np.argsort(raw_column_values, axis=-1), axis=-1)
    raw_mean_ranks = [np.nonzero(raw_ranks == i)[1].mean() for i in range(len(ue_methods))]

    temp_ranks = np.flip(np.argsort(temp_column_values, axis=-1), axis=-1)
    temp_mean_ranks = [np.nonzero(temp_ranks == i)[1].mean() for i in range(len(ue_methods))]

    total_ranks = np.flip(np.argsort(total_column_values, axis=-1), axis=-1)
    total_mean_ranks = [np.nonzero(total_ranks == i)[1].mean() for i in range(len(ue_methods) * 2)]

    for i, method in enumerate(ue_methods):
        ue_scores[f'{method}_raw'].extend((str(raw_mean_ranks[i]), '-', total_mean_ranks[i * 2]))
        ue_scores[f'{method}_temp'].extend(('-', str(temp_mean_ranks[i]), total_mean_ranks[i * 2 + 1]))

    return ue_scores, best_temperatures, ave_test_metric_values



from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
import numpy as np

def isotonic_scale_ue(datasets, model, model_type, all_metrics, ue_methods, methods_dict, task='nmt'):
    ue_scores = defaultdict(list)
    learned_models = defaultdict(list)
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
        _, _ = extract_and_prepare_data(dataset, methods_dict, [metric], model=model, model_type=model_type, task=task)

        ave_test_metric_values[dataset] = np.mean(test_metric_values[metric])

        for method in ue_methods:
            train_ue = train_ue_values[method]
            test_ue = test_ue_values[method]
            train_metric = train_metric_values[metric]
            test_metric = test_metric_values[metric]

            # Normalize both
            ue_scaler = MinMaxScaler()
            train_ue = ue_scaler.fit_transform(train_ue[:, None]).squeeze()
            test_ue = ue_scaler.transform(test_ue[:, None]).squeeze()

            # Fit isotonic regressor: maps UE → quality (e.g., score estimate)
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(train_ue, train_metric)
            learned_models[method].append(iso)

            # Apply learned calibration on test
            test_ue_scaled = iso.transform(test_ue)
            prr_scaled = score_ues(test_ue_scaled, test_metric)
            prr_raw = score_ues(test_ue, test_metric)

            ue_scores[f'{method}_raw'].append(prr_raw)
            ue_scores[f'{method}_iso'].append(prr_scaled)

    # Format PRR scores into LaTeX-style strings
    raw_column_values = []
    iso_column_values = []

    for i in range(len(datasets)):
        raw_column_values.append([ue_scores[f'{method}_raw'][i] for method in ue_methods])
        iso_column_values.append([ue_scores[f'{method}_iso'][i] for method in ue_methods])

        metric_raw_scores = np.array(raw_column_values[-1])
        metric_iso_scores = np.array(iso_column_values[-1])

        top_raw = np.argmax(metric_raw_scores)
        top_iso = np.argmax(metric_iso_scores)

        for method in ue_methods:
            ue_scores[f'{method}_raw'][i] = f'{ue_scores[f"{method}_raw"][i]:.2f}'
            ue_scores[f'{method}_iso'][i] = f'{ue_scores[f"{method}_iso"][i]:.2f}'

        ue_scores[f'{ue_methods[top_raw]}_raw'][i] = f'\\underline{{{ue_scores[f"{ue_methods[top_raw]}_raw"][i]}}}'
        ue_scores[f'{ue_methods[top_iso]}_iso'][i] = f'\\textbf{{{ue_scores[f"{ue_methods[top_iso]}_iso"][i]}}}'

    # Optional: ranking summary
    total_column_values = []
    for raw, iso in zip(raw_column_values, iso_column_values):
        total_column_values.append([val for pair in zip(raw, iso) for val in pair])

    raw_ranks = np.flip(np.argsort(raw_column_values, axis=-1), axis=-1)
    raw_mean_ranks = [np.nonzero(raw_ranks == i)[1].mean() for i in range(len(ue_methods))]

    iso_ranks = np.flip(np.argsort(iso_column_values, axis=-1), axis=-1)
    iso_mean_ranks = [np.nonzero(iso_ranks == i)[1].mean() for i in range(len(ue_methods))]

    total_ranks = np.flip(np.argsort(total_column_values, axis=-1), axis=-1)
    total_mean_ranks = [np.nonzero(total_ranks == i)[1].mean() for i in range(len(ue_methods) * 2)]

    for i, method in enumerate(ue_methods):
        ue_scores[f'{method}_raw'].extend((str(raw_mean_ranks[i]), '-', total_mean_ranks[i * 2]))
        ue_scores[f'{method}_iso'].extend(('-', str(iso_mean_ranks[i]), total_mean_ranks[i * 2 + 1]))

    return ue_scores, learned_models, ave_test_metric_values


from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
from scipy.optimize import minimize_scalar
import numpy as np

def sigmoid_scale_ue(datasets, model, model_type, all_metrics, ue_methods, methods_dict, task='nmt'):
    def sigmoid_scaling(x, temp):
        x_centered = x - np.mean(x)
        return 1 / (1 + np.exp(-x_centered / temp))

    ue_scores = defaultdict(list)
    best_temperatures = defaultdict(list)
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
        _, _ = extract_and_prepare_data(dataset, methods_dict, [metric], model=model, model_type=model_type, task=task)

        ave_test_metric_values[dataset] = np.mean(test_metric_values[metric])

        for method in ue_methods:
            train_ue = train_ue_values[method]
            train_metric = train_metric_values[metric]
            test_ue = test_ue_values[method]
            test_metric = test_metric_values[metric]

            # Normalize UE to [0, 1]
            scaler = MinMaxScaler()
            train_ue = scaler.fit_transform(train_ue[:, None]).squeeze()
            test_ue = scaler.transform(test_ue[:, None]).squeeze()

            # Optimize temperature on training set
            def objective(temp):
                scaled = sigmoid_scaling(train_ue, temp)
                return -score_ues(scaled, train_metric)

            res = minimize_scalar(objective, bounds=(0.05, 5.0), method='bounded')
            best_T = res.x
            best_temperatures[method].append(best_T)

            # Apply on test set
            test_scaled = sigmoid_scaling(test_ue, best_T)
            prr_scaled = score_ues(test_scaled, test_metric)
            prr_raw = score_ues(test_ue, test_metric)

            ue_scores[f'{method}_raw'].append(prr_raw)
            ue_scores[f'{method}_sigmoid'].append(prr_scaled)

    # Format output as in detrend_ue()
    raw_column_values = []
    sigmoid_column_values = []

    for i in range(len(datasets)):
        raw_column_values.append([ue_scores[f'{method}_raw'][i] for method in ue_methods])
        sigmoid_column_values.append([ue_scores[f'{method}_sigmoid'][i] for method in ue_methods])

        top_raw = np.argmax(raw_column_values[-1])
        top_sigmoid = np.argmax(sigmoid_column_values[-1])

        for method in ue_methods:
            ue_scores[f'{method}_raw'][i] = f'{ue_scores[f"{method}_raw"][i]:.2f}'
            ue_scores[f'{method}_sigmoid'][i] = f'{ue_scores[f"{method}_sigmoid"][i]:.2f}'

        ue_scores[f'{ue_methods[top_raw]}_raw'][i] = f'\\underline{{{ue_scores[f"{ue_methods[top_raw]}_raw"][i]}}}'
        ue_scores[f'{ue_methods[top_sigmoid]}_sigmoid'][i] = f'\\textbf{{{ue_scores[f"{ue_methods[top_sigmoid]}_sigmoid"][i]}}}'

    # Optional ranking block
    total_column_values = []
    for raw, scaled in zip(raw_column_values, sigmoid_column_values):
        total_column_values.append([val for pair in zip(raw, scaled) for val in pair])

    raw_ranks = np.flip(np.argsort(raw_column_values, axis=-1), axis=-1)
    raw_mean_ranks = [np.nonzero(raw_ranks == i)[1].mean() for i in range(len(ue_methods))]

    sigmoid_ranks = np.flip(np.argsort(sigmoid_column_values, axis=-1), axis=-1)
    sigmoid_mean_ranks = [np.nonzero(sigmoid_ranks == i)[1].mean() for i in range(len(ue_methods))]

    total_ranks = np.flip(np.argsort(total_column_values, axis=-1), axis=-1)
    total_mean_ranks = [np.nonzero(total_ranks == i)[1].mean() for i in range(len(ue_methods) * 2)]

    for i, method in enumerate(ue_methods):
        ue_scores[f'{method}_raw'].extend((str(raw_mean_ranks[i]), '-', total_mean_ranks[i * 2]))
        ue_scores[f'{method}_sigmoid'].extend(('-', str(sigmoid_mean_ranks[i]), total_mean_ranks[i * 2 + 1]))

    return ue_scores, best_temperatures, ave_test_metric_values

from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize
import numpy as np
from collections import defaultdict


def quadratic_scale_ue(datasets, model, model_type, all_metrics, ue_methods, methods_dict, task='nmt'):
    def quad_fn(x, a, b, c):
        return a * x**2 + b * x + c

    ue_scores = defaultdict(list)
    best_params = defaultdict(list)
    ave_test_metric_values = {}

    if len(all_metrics) == 1 and len(datasets) > 1:
        all_metrics = all_metrics * len(datasets)
    elif len(all_metrics) != len(datasets):
        raise ValueError('Number of metrics and datasets must be the same')

    for metric, dataset in zip(all_metrics, datasets):
        train_ue_values, test_ue_values, train_metric_values, test_metric_values, _, _ = extract_and_prepare_data(
            dataset, methods_dict, [metric], model=model, model_type=model_type, task=task)

        ave_test_metric_values[dataset] = np.mean(test_metric_values[metric])

        for method in ue_methods:
            train_ue = train_ue_values[method]
            test_ue = test_ue_values[method]
            train_metric = train_metric_values[metric]
            test_metric = test_metric_values[metric]

            # Normalize to [0, 1]
            scaler = MinMaxScaler()
            train_ue = scaler.fit_transform(train_ue[:, None]).squeeze()
            test_ue = scaler.transform(test_ue[:, None]).squeeze()

            def objective(params):
                a, b, c = params
                scaled = quad_fn(train_ue, a, b, c)
                return -score_ues(scaled, train_metric)

            res = minimize(objective, x0=[0, 1, 0], method='Powell')  # or 'Nelder-Mead'
            a_opt, b_opt, c_opt = res.x
            best_params[method].append((a_opt, b_opt, c_opt))

            test_scaled = quad_fn(test_ue, a_opt, b_opt, c_opt)
            prr_scaled = score_ues(test_scaled, test_metric)
            prr_raw = score_ues(test_ue, test_metric)

            ue_scores[f'{method}_raw'].append(prr_raw)
            ue_scores[f'{method}_quad'].append(prr_scaled)

    # Format output LaTeX-style
    raw_column_values = []
    quad_column_values = []

    for i in range(len(datasets)):
        raw_column_values.append([ue_scores[f'{method}_raw'][i] for method in ue_methods])
        quad_column_values.append([ue_scores[f'{method}_quad'][i] for method in ue_methods])

        top_raw = np.argmax(raw_column_values[-1])
        top_quad = np.argmax(quad_column_values[-1])

        for method in ue_methods:
            ue_scores[f'{method}_raw'][i] = f'{ue_scores[f"{method}_raw"][i]:.2f}'
            ue_scores[f'{method}_quad'][i] = f'{ue_scores[f"{method}_quad"][i]:.2f}'

        ue_scores[f'{ue_methods[top_raw]}_raw'][i] = f'\\underline{{{ue_scores[f"{ue_methods[top_raw]}_raw"][i]}}}'
        ue_scores[f'{ue_methods[top_quad]}_quad'][i] = f'\\textbf{{{ue_scores[f"{ue_methods[top_quad]}_quad"][i]}}}'

    return ue_scores, best_params, ave_test_metric_values
