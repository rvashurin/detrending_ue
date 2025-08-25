# **Uncertainty-LINE**: Length-Invariant Estimation of Uncertainty for Large Language Models.

This repository contains the code and processed data for reproducing the Uncertainty-LINE method.

---

## Repository Structure

```
├── processed_mans/
│ └── ... # Full experimental managers, including all model outputs, UE and Quality metrics
├── 01_plots.ipynb # Notebook for generating visualizations and plots
├── 02_results.ipynb # Notebook for assembling and analyzing overall experimental results
├── 03_ablations.ipynb # Notebook for running ablation studies to assess method components
├── utils.py # Main utility functions: data loading, regression, evaluation, plotting, etc.
├── README.md # This document
└── requirements.txt # Python dependencies
```

---

## Getting Started

1. **Install Dependencies**

   ```bash
   pip install -r requirements.txt  
   ```

2. **Run lm-polygraph**

   ```bash
HYDRA_CONFIG=`pwd`/examples/configs/polygraph_eval_wmt14_csen.yaml \
  polygraph_eval \
  batch_size=1 \
  cache_path=/path/to/cache \
  model=gemma  \
  subsample_eval_dataset=2000 \
  deberta_batch_size=1 \
  +deberta_device=cuda:0 \
  model.load_model_args.device_map=auto 


# Run evaluation on the train split
HYDRA_CONFIG=`pwd`/examples/configs/polygraph_eval_wmt14_csen.yaml \
  polygraph_eval \
  batch_size=1 \
  cache_path=/path/to/cache/train \
  model=gemma \
  subsample_eval_dataset=2000 \
  deberta_batch_size=1 \
  eval_split=train \
  +deberta_device=cuda:0 \
  model.load_model_args.device_map=auto 
   ```