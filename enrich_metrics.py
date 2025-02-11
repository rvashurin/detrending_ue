import gc
import torch
from comet import download_model, load_from_checkpoint
from sacrebleu import BLEU
from utils import load_managers
from typing import List
from lm_polygraph.utils.manager import UEManager
import pathlib

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
#
#def get_comet_metric_scores(
#        translated_sentences: List[str],
#        reference_sentences: List[str],
#        original_sentences: List[str],
#    ):
#    model_path = download_model('Unbabel/wmt22-comet-da')
#    model = load_from_checkpoint(model_path)
#
#    data = []
#    for original, translation, reference in zip(original_sentences, translated_sentences, reference_sentences):
#        data.append({'src': original, 'mt': translation, 'ref': reference})
#
#    scores = model.predict(data, batch_size=1, gpus=1)
#
#    return scores
#
#def get_comet_qe_scores(
#    translated_sentences: List[str],
#    original_sentences: List[str],
#    ):
#    model_path = download_model('Unbabel/wmt23-cometkiwi-da-xxl')
#    model = load_from_checkpoint(model_path)
#
#    data = []
#    for original, translation in zip(original_sentences, translated_sentences):
#        data.append({'src': original, 'mt': translation})
#
#    scores = model.predict(data, batch_size=1, gpus=1)
#
#    return scores

def get_bleu_scores(
    translated_sentences: List[str],
    reference_sentences: List[str],
    ):

    bleu = BLEU(effective_order=True)
    scores = [bleu.sentence_score(translated_sentences[i], [reference_sentences[i]]).score for i in range(len(translated_sentences))]
    signature = bleu.get_signature()

    return scores, signature

for model in MODELS:
    pathlib.Path(f'processed_mans').mkdir(parents=True, exist_ok=True)

    if 'llama' in model:
        datasets = LLAMA_DATASETS
    else:
        datasets = DATASETS

    for dataset in datasets:
        manager = UEManager.load(f'mans/{model}_{dataset}_test.man')

        original_sentences = manager.stats['input_texts']
        translated_sentences = manager.stats['greedy_texts']
        reference_sentences = manager.stats['target_texts']

        #manager.gen_metrics[('sequence', 'comet_metric')] = get_comet_metric_scores(translated_sentences, reference_sentences, original_sentences)[0]
        #manager.gen_metrics[('sequence', 'comet_qe')] = get_comet_qe_scores(translated_sentences, original_sentences)[0]
        manager.gen_metrics[('sequence', 'bleu_proper')] = get_bleu_scores(translated_sentences, reference_sentences)[0]

        manager.save(f'processed_mans/{model}_{dataset}_test_processed.man')

        gc.collect()
        torch.cuda.empty_cache()
