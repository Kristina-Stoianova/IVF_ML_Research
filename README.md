# Using Machine Learning to predict outcomes of Controlled Ovarian Stimulation (COS) in IVF/ICSI cycles
##  Predicting Metaphase II (MII) Oocyte Yield from Baseline Pre-treatment Clinical Predictors 

This project develops an extensive data cleaning pipeline to clean messy clinical data
This project develops a custom workflow to predict the number of MII oocytes and identify top predictors

5 model families are evalauted: Ridge regression, ElasticNet, Support Vector Regression, RandomForest, and XGBoost

There are 2 main ML pipelines built in Python 3.11 using scikit-learn and xgboost:
- Repeated k-fold cross-validation screening pipeline
   - Iteratively evaluates all possible predictor combinations
- Nested cross-validation pipeline
   - Used to separate model selection from evaluation and provide a less optimistic estimate of performance 

> **Data availability:** the patient-level dataset is *not* included in this

## Repository structure

```
├─r_markdown/           # R data cleaning, preprocessing, and  statistical analysis notebooks (.Rmd)
├─processing_scripts/   # R helper functions required for data cleaning (sourced by the .Rmd file
├─modelling_scripts/    # Python screening, nested CV, final interpretation of SVR model
├─hpc/                  # SLURM submission script
├─results/              # Pipeline results / .csv outputs
├─figures/              # Figures
├─data/                 
├── environment.yml       # Conda environment for the Python/HPC modelling
└── renv.lock             # R package versions
```


**1. Clean** — run the cleaning notebook in `r_markdown/` ( source helper functions
from `processing_scripts/`) to generate clean dataset

**2. Pre-processing, Feature Selection and Exploratory Analysis** — missingness (`naniar`), correlation (`ggcorrplot`), exploratory modelling (`MASS`)

**3. ML Modelling (HPC)** — submit as batch job via bash script, this will run the screening and nested cv pipelines

​```bash
sbatch submit_modelling.sh
​```

.csv results saved in `results/`; figures in `figures/`.

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

