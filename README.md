# Predicting Metaphase II (MII) Oocyte Yield from Baseline Pre-treatment Clinical Predictors

Machine-learning pipeline for predicting metaphase-II (MII) oocyte yield from
baseline clinical predictors in IVF patients on a short antagonist protocol
 - 5 model families are evalauted: Ridge regression, ElasticNet, Support Vector Regression, RandomForest, and XGBoost

There are 2 main pipelines in this project:
- Repeated k-fold cross-validation screening pipeline
   - Iteratively evaluates all possible predictor combinations
- Nested cross-validation pipeline
   - Used to separate model selection from evaluation and provide a less optimistic estimate of performance 

> **Data availability:** the patient-level dataset is *not* included in this

## Repository structure

​```
├─r_markdown/           # R data cleaning, preprocessing, and  statistical analysis notebooks (.Rmd)
├─processing_scripts/   # R helper functions required for data cleaning (sourced by the .Rmd file
├─modelling_scripts/    # Python screening, nested CV, final interpretation of SVR model
├─hpc/                  # SLURM submission script
├─results/              # Pipeline results / .csv outputs
├─figures/              # Figures
├─data/                 
├── environment.yml       # Conda environment for the Python/HPC modelling
└── renv.lock             # R package versions
​```

## Pipeline / run order

Stages 1–2 are local ; stage 3 runs on the HPC.

**1. Clean** — run the cleaning notebook in `r_markdown/` (these source helpers
from `processing_scripts/`, so keep relative paths intact) to generate clean dataset

**2. Pre-processing, Feature Selection and Exploratory Analysis** — missingness (`naniar`), correlation (`ggcorrplot`), exploratory modelling (`MASS`)

**3. ML Modelling (HPC)** — submit as batch job via bash script, this will run the screening and nested cv pipelines

​```bash
sbatch submit_modelling.sh
​```

CSV metrics land in `results/`; figures in `figures/`.

## Environment

### Python (HPC modelling)
​```bash
conda env create -f environment.yml
conda activate ml_env
​```

### R (Data cleaning + exploratory analysis)
​```r
renv::restore()
​```

