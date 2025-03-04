import numpy as np
import matplotlib.pyplot as plt
from lm_polygraph.utils.manager import UEManager
import sklearn
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
import logging

log = logging.getLogger("lm_polygraph")
log.setLevel(logging.ERROR)

def load_managers(dataset, model='llama', model_type='base'):
    prefix = '' if model_type == 'base' else '_instruct'
    manager = UEManager.load(f'processed_mans/{model}{prefix}_{dataset}_test_processed.man')

    train_manager = UEManager.load(f'mans/{model}{prefix}_{dataset}_train.man')

    return manager, train_manager

def extract_and_prepare_data(dataset, methods_dict, all_metrics, model='llama', model_type='base'):
    manager, train_manager = load_managers(dataset, model, model_type)

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
