import gc
import torch
from comet import download_model, load_from_checkpoint
from sacrebleu import BLEU
from utils import load_managers
from typing import List
from lm_polygraph.utils.manager import UEManager
import pathlib

MODELS = ['llama', 'gemma', 'eurollm']
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
LLAMA_DATASETS = DATASETS

def get_bleu_scores(
    translated_sentences: List[str],
    reference_sentences: List[str],
):

    bleu = BLEU(effective_order=True)
    scores = [bleu.sentence_score(translated_sentences[i], [reference_sentences[i]]).score for i in range(len(translated_sentences))]
    signature = bleu.get_signature()

    return scores, signature

for model in MODELS:
    #for model_type in ['base', 'instruct', 'instruct_zeroshot']:
    for model_type in ['base']:
        prefix = '' if model_type == 'base' else '_instruct'
        if model_type == 'instruct_zeroshot':
            prefix = '_instruct_zeroshot'

        pathlib.Path(f'processed_mans').mkdir(parents=True, exist_ok=True)

        if 'llama' in model:
            datasets = LLAMA_DATASETS
        else:
            datasets = DATASETS

        for dataset in datasets:
            manager = UEManager.load(f'mans/{model}{prefix}_{dataset}_test_qe_enriched.man')

            original_sentences = manager.stats['input_texts']
            translated_sentences = manager.stats['greedy_texts']
            reference_sentences = manager.stats['target_texts']

            manager.gen_metrics[('sequence', 'bleu_proper')] = get_bleu_scores(translated_sentences, reference_sentences)[0]

            manager.save(f'processed_mans/{model}{prefix}_{dataset}_test_qe_enriched_processed.man')

            gc.collect()
            torch.cuda.empty_cache()
