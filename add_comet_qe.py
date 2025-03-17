import gc
import torch
from comet import download_model, load_from_checkpoint
from sacrebleu import BLEU
from utils import load_managers
from typing import List

MODELS = ['llama', 'gemma']
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


def get_comet_qe_scores(
    translated_sentences: List[str],
    original_sentences: List[str],
    ):
    model_path = download_model('Unbabel/wmt23-cometkiwi-da')
    model = load_from_checkpoint(model_path)

    data = []
    for original, translation in zip(original_sentences, translated_sentences):
        data.append({'src': original, 'mt': translation})

    scores = model.predict(data, batch_size=1, gpus=1)

    return scores


def get_comet_metric_scores(
        translated_sentences: List[str],
        reference_sentences: List[str],
        original_sentences: List[str],
    ):
    model_path = download_model('Unbabel/wmt22-comet-da')
    model = load_from_checkpoint(model_path)

    data = []
    for original, translation, reference in zip(original_sentences, translated_sentences, reference_sentences):
        data.append({'src': original, 'mt': translation, 'ref': reference})

    scores = model.predict(data, batch_size=1, gpus=1)

    return scores


for split in ['test', 'train']:
    for model in MODELS:
        for dataset in DATASETS:
            manager = UEManager.load(f'mans/{model}_{dataset}_{split}.man')

            original_sentences = manager.stats['input_texts']
            translated_sentences = manager.stats['greedy_texts']
            reference_sentences = manager.stats['target_texts']

            manager.estimations[('sequence', 'comet_qe')] = get_comet_qe_scores(translated_sentences, original_sentences)[0]

            if model == 'llama':
                manager.gen_metrics[('sequence', 'Comet')] = get_comet_metric_scores(translated_sentences, reference_sentences, original_sentences)[0]

            manager.save(f'processed_mans/{model}_{dataset}_{split}_qe_enriched.man')

            gc.collect()
            torch.cuda.empty_cache()
