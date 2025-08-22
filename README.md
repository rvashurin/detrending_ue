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
   pip install -r requirements.txt```