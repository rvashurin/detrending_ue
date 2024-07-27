import numpy as np
import matplotlib.pyplot as plt
from lm_polygraph.utils.manager import UEManager
from lm_polygraph.ue_metrics.pred_rej_area import PredictionRejectionArea
from lm_polygraph.ue_metrics.ue_metric import (
    get_random_scores,
    normalize_metric,
)
import sklearn
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from collections import defaultdict
from sacrebleu import CHRF, BLEU

methods_dict = {
    'MaximumSequenceProbability': 'MSP',
    'Perplexity': 'PPL',
    'MeanTokenEntropy': 'MTE',
    'MeanPointwiseMutualInformation': 'MPMI',
    'MeanConditionalPointwiseMutualInformation': 'MCPMI',
    'CCP': 'CCP',
    'PTrue': 'PTrue',
    'PTrueSampling': 'PTrueS',
    'MonteCarloSequenceEntropy': 'MCSE',
    'MonteCarloNormalizedSequenceEntropy': 'MCNSE',
    'LexicalSimilarity_rouge1': 'LSR1',
    'LexicalSimilarity_rouge2': 'LSR2',
    'LexicalSimilarity_rougeL': 'LSRL',
    'LexicalSimilarity_BLEU': 'LSB',
    'NumSemSets': 'NSS',
    'EigValLaplacian_NLI_score_entail': 'ELE',
    'EigValLaplacian_NLI_score_contra': 'ELC',
    'EigValLaplacian_Jaccard_score': 'ELJ',
    'DegMat_NLI_score_entail': 'DME',
    'DegMat_NLI_score_contra': 'DMC',
    'DegMat_Jaccard_score': 'DMJ',
    'Eccentricity_NLI_score_entail': 'EcE',
    'Eccentricity_NLI_score_contra': 'EcC',
    'Eccentricity_Jaccard_score': 'EcJ',
    'SemanticEntropy': 'SE',
    'SAR': 'SAR',
    'TokenSAR': 'TSAR',
    'SentenceSAR': 'SSAR',
    'RenyiNeg': 'RN',
    'FisherRao': 'FR',
}

MODELS = ['stablelm12b', 'mistral7b']
DATASETS = ['wmt14', 'wmt19']

gen_metrics = ['Comet', 'Rouge_rougeL', 'BLEU', 'CHRF']
#additional_metrics = ['BLEU', 'CHRF']
additional_metrics = []
all_metrics = gen_metrics + additional_metrics

ue_metric = PredictionRejectionArea()

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

for model in MODELS:
    ue_scores = defaultdict(list)
    for dataset in DATASETS:
        manager = UEManager.load(f'mans/polygraph_tacl_{model}_{dataset}_nmt_aug.man')
        train_manager = UEManager.load(f'mans/polygraph_tacl_{model}_{dataset}_train_nmt_aug.man')

        full_ue_methods = [key[1] for key in train_manager.estimations.keys()]
        ue_methods = [methods_dict[method] for method in full_ue_methods]

        sequences = manager.stats['greedy_tokens']
        texts = manager.stats['greedy_texts']
        targets = manager.stats['target_texts']

        train_sequences = train_manager.stats['greedy_tokens']
        train_texts = train_manager.stats['greedy_texts']
        train_targets = train_manager.stats['target_texts']

        train_gen_lengths = np.array([len(seq) for seq in train_sequences])
        gen_lengths = np.array([len(seq) for seq in sequences])
        #uniq_lengths = np.sort(np.unique(gen_lengths))
        #len_ids = [np.argwhere(gen_lengths == length).squeeze() for length in uniq_lengths]


        # Get train and test values for metrics and UE, remove union of nans
        test_nans = []
        train_nans = []

        train_metric_values = {}
        test_metric_values = {}
        for metric in gen_metrics:
            values = np.array(manager.gen_metrics[('sequence', metric)])
            test_metric_values[metric] = np.array(values)
            test_nans.extend(np.argwhere(np.isnan(values)).flatten())

            train_values = np.array(train_manager.gen_metrics[('sequence', metric)])
            train_metric_values[metric] = np.array(train_values)
            train_nans.extend(np.argwhere(np.isnan(train_values)).flatten())

        for metric in additional_metrics:
            try:
                scorer = globals()[metric](effective_order=True)
            except:
                scorer = globals()[metric]()
            values = np.array([scorer.sentence_score(texts[i], [targets[i]]).score for i in range(len(texts))])
            test_metric_values[metric] = values
            test_nans.extend(np.argwhere(np.isnan(values)).flatten())
            manager.gen_metrics[('sequence', metric)] = values
            manager.save(f'mans/polygraph_tacl_{model}_{dataset}_nmt_aug.man')

            train_values = np.array([scorer.sentence_score(train_texts[i], [train_targets[i]]).score for i in range(len(train_texts))])
            train_metric_values[metric] = train_values
            train_nans.extend(np.argwhere(np.isnan(train_values)).flatten())
            train_manager.gen_metrics[('sequence', metric)] = train_values
            train_manager.save(f'mans/polygraph_tacl_{model}_{dataset}_train_nmt_aug.man')

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


        train_normalized_metric_values = {}
        test_normalized_metric_values = {}
        coefs = defaultdict(list)
        for metric in all_metrics:
            scaler = MinMaxScaler()
            train_normalized_metric_values[metric] = scaler.fit_transform(train_metric_values[metric][:, np.newaxis]).squeeze()
            linreg = sklearn.linear_model.LinearRegression()
            linreg.fit(train_gen_lengths[:, np.newaxis], train_normalized_metric_values[metric])
            coefs[metric].append(linreg.coef_[0])

            test_normalized_metric_values[metric] = scaler.transform(test_metric_values[metric][:, np.newaxis]).squeeze()

            linreg = sklearn.linear_model.LinearRegression()
            linreg.fit(gen_lengths[:, np.newaxis], test_normalized_metric_values[metric])
            coefs[metric].append(linreg.coef_[0])

        train_normalized_ue_values = {}
        test_normalized_ue_values = {}

        train_coefs = {}
        ue_residuals = {}

        for method in ue_methods:
            scaler = MinMaxScaler()
            train_normalized_ue_values[method] = scaler.fit_transform(train_ue_values[method][:, np.newaxis]).squeeze()
            test_normalized_ue_values[method] = scaler.transform(test_ue_values[method][:, np.newaxis]).squeeze()

            linreg = sklearn.linear_model.LinearRegression()
            linreg.fit(train_gen_lengths[:, np.newaxis], train_normalized_ue_values[method])
            train_coefs[method] = linreg.coef_[0]

            ue_residuals[method] = test_normalized_ue_values[method] - linreg.predict(gen_lengths[:, np.newaxis])
            for metric in all_metrics:
                met_vals = test_metric_values[metric]
                raw_score = score_ues(test_ue_values[method], met_vals)
                raw_norm_score = score_ues(test_normalized_ue_values[method], met_vals)
                detrended_score = score_ues(ue_residuals[method], met_vals)
                diff = raw_score - detrended_score
                ue_scores[f'{method}_raw'].append(raw_score)
                #ue_scores[f'{method}_raw_norm'].append(raw_norm_score)
                ue_scores[f'{method}_detr'].append(detrended_score)
                #ue_scores[f'{method}_diff'].append(diff)

    raw_column_values = []
    detr_column_values = []
    for j, _ in enumerate(DATASETS):
        n = len(all_metrics)
        for i, _ in enumerate(all_metrics):
            _id = i + j*n
            
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

        #fig, axs = plt.subplots(1, len(all_metrics), figsize=(8*len(all_metrics), 7))

        #for i, metric in enumerate(all_metrics):
        #    ax = axs[i]
        #    ax.plot(uniq_lengths, metric_values[metric], label=metric)
        #    for method in ue_methods:
        #        pass
        #        #ax.plot(uniq_lengths, ue_values[method], label=method, linestyle='--', alpha=0.5)
        #        #ax.plot(uniq_lengths, sorted_ue_residuals[method], label=f'{method}_detr', linestyle='--', alpha=0.5)
        #    ax.legend()
        #    ax.set_title(metric)
        #    ax.set_xlabel('Generated sequence length')
        #    ax.set_ylabel('Metric value')
        #    if metric == 'Comet':
        #        ax.set_ylim(0, 1)
        #    else:
        #        ax.set_ylim(0, 100)

        #fig.suptitle(f'Polygraph TACL StableLM12b {dataset}')

        #plt.tight_layout()
        #plt.savefig(f'charts/polygraph_tacl_stablelm12b_{dataset}.png')

    columns = [f'{dataset}_{metric}' for dataset in DATASETS for metric in all_metrics] + ['raw_rank', 'detr_rank', 'rank']
    df = pd.DataFrame.from_dict(ue_scores, orient='index', columns=columns)
    with open(f'{model}_ue_scores.tex', 'w') as f:
        latex = df.to_latex(float_format="%.2f", escape=False)
        latex = latex.replace('_', '\_')
        # find first line where \midrule is present
        start_id = latex.split('\n').index('\\midrule') + 2
        # add \midrule every third line starting from start_id
        latex = '\n'.join([line if i % 2 != 0 else line + '\n\\midrule' for i, line in enumerate(latex.split('\n'), start=start_id)])
        f.write(latex)

    columns = [f'{dataset}_{coef_type}' for dataset in DATASETS for coef_type in ['train_c, test_c']]
    df = pd.DataFrame.from_dict(coefs, orient='index', columns=columns)
    with open(f'{model}_metric_trends.tex', 'w') as f:
        latex = df.to_latex(float_format="%.2f", escape=False)
        latex = latex.replace('_', '\_')
        f.write(latex)
