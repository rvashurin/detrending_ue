import gc
import torch
from lm_polygraph.utils.deberta import Deberta, MultilingualDeberta
from utils import load_managers
from typing import List
from lm_polygraph.utils.manager import UEManager
from lm_polygraph.generation_metrics import *
from lm_polygraph.estimators import *
from lm_polygraph.stat_calculators import *
import pathlib
import os
from tqdm import tqdm


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

nli_model = Deberta(batch_size=10, device='cuda:0')

estimators = [TokenSAR(),]

ue_metrics = [
    PredictionRejectionArea(max_rejection=0.5),
]

stat_calculators = [
    CrossEncoderSimilarityMatrixCalculator(nli_model=nli_model)
]

#managers = {}
for model in tqdm(MODELS):
    #for model_type in ['base', 'instruct']:
    for model_type in ['base']:
        for split in ['train', 'test']:
            prefix = '' if model_type == 'base' else '_instruct'

            pathlib.Path(f'processed_mans').mkdir(parents=True, exist_ok=True)

            for dataset in tqdm(DATASETS):
                man = UEManager.load(f'/workspace/processed_mans/{model}{prefix}_{dataset}_{split}.man')

                stats = man.stats

                for calculator in stat_calculators:
                    texts = stats["greedy_texts"]
                    values = calculator(dependencies=stats, texts=texts, model=None)
                    stats.update(values)

                for estimator in estimators:
                    values = estimator(stats)
                    man.estimations[('sequence', str(estimator))] = values

                man.stats = stats

                man.ue_metrics = ue_metrics

                man.eval_ue()
                pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
                man.save_path = os.path.join(out_dir, f"{model}_{dataset}.man")
                man.save()

if __name__ == '__main__':
    main()
