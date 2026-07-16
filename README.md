# Using Machine Learning to predict outcomes of Controlled Ovarian Stimulation (COS) in IVF/ICSI cycles
 -  Predicting the number of Metaphase II (MII) oocytes from Baseline Pre-treatment Clinical Variables

# Project Overview:

## 1. Development of a data cleaning pipeline to standardise and prepare clinical data for modelling

## 2. Development of a custom modelling workflow to predict the number of MII oocytes and identify top predictors
  - 5 model families are evaluated: Ridge regression, ElasticNet, Support Vector Regression, RandomForest, and XGBoost
  
## There are 2 main ML pipelines built in Python 3.12.13 using scikit-learn and xgboost:
- Repeated k-fold cross-validation screening pipeline
   - Iteratively evaluates all possible predictor combinations
- Nested cross-validation pipeline
   - Used to separate model selection from evaluation
   - Provides a less optimistic estimate of performance 
   - Allows predictor stability to be assessed
- The workflow is optimised for a small dataset and addresses missing data with KNN imputation

> **Data availability:** the patient dataset is *not* included in this repository

## Repository structure

```
IVF_ML_research/
├── data/
│   ├── raw/                            # original patient data (pre-cleaning)
│   └── processed/                      # cleaned, model-ready CSVs
├── r_markdown/
│   └── Cleaning_pipe.Rmd               # main R cleaning & preprocessing pipeline
├── processing_scripts/
│   ├── cleaning_helpers.R              # value standardisation helpers
│   └── cleaning_functions_parsers.R    # stimulation/trigger string parsers
├── modelling_scripts/
│   ├── model_looping1.py               # Repeated k-fold cross-validation screening pipeline
│   ├── nested_model_looping.py         # Nested cross-validation
│   ├── nested_single_mode.py           # Nested cross-validation - fixed predictors only tunes hyperparameters
│   ├── run_script.sh                   # Example submission script
│   └── SVR_interpretation.ipynb        # SVR interpretation (SHAP, perm importance) - fixed predictors and hyperparameters (can be found in this notebook)
├── results/                            # pipeline outputs / .csv metrics
├── figures/                            # generated figures (PNG/PDF)
├── environment.yml                     # Python env (conda) ml_env
└── renv.lock                           # R package versions
```

| Workflow step | Description |
|---------------|-------------|
| **1. Clean** | Run the cleaning notebook in `r_markdown/` (sources helper functions from `processing_scripts/`) to generate the clean dataset |
| **2. Pre-processing, feature selection and exploratory analysis** | Missingness (`naniar`), Correlation (`ggcorrplot`), Statistical analysis, Exploratory modelling (`MASS`) |
| **3. ML modelling** | scikit-learn pipeline (iterative predictor screening --> nested CV --> SHAP interpretation) |


## Environment

### Python ML modelling is performed on the HPC, SVR interpretation can be run locally  
 - In both cases conda enviornment captures package dependencies

​```bash
conda env create -f environment.yml
conda activate ml_env
​```

- Run modelling pipeline by submitted as batch job using bash script:
​```bash
sbatch -p run_script.sh
​```

### Output of modelling 
.CSV results saved in `results/`
;figures in `figures/`.

### Python (3.12.13)
 - Recreate with the conda environment file:

​```bash
conda env create -f environment.yml
conda activate ml_env
​```

| Package | Version |
|--------------|---------|
| scikit-learn | 1.6.1 |
| xgboost | 3.2.0 |
| shap | 0.52.0 |
| scipy | 1.16.3 |
| pandas | 2.2.2 |
| numpy | 2.0.2 |
| matplotlib | 3.10.0 |
| tqdm | 4.67.3 |

### R (4.5.1)

​
```r
install.packages(c(
  "here", "janitor", "tidyverse", "readr", "dplyr", "stringr",
  "knitr", "ggplot2", "ggcorrplot", "purrr", "naniar", "Amelia",
  "UpSetR", "VIM", "ggpubr", "patchwork", "cowplot", "FSA",
  "MASS", "broom", "car"
))

```
​

| Package | Version | | Package | Version |
|------------|---------|---|-----------|---------|
| here | 1.0.2 | | naniar | 1.1.0 |
| janitor | 2.2.1 | | Amelia | 1.8.3 |
| tidyverse | 2.0.0 | | UpSetR | 1.4.0 |
| readr | 2.1.6 | | VIM | 6.2.6 |
| dplyr | 1.2.0 | | ggpubr | 0.6.3 |
| stringr | 1.6.0 | | patchwork | 1.3.2 |
| knitr | 1.5.0 | | cowplot | 1.2.0 |
| ggplot2 | 4.0.1 | | FSA | 0.10.1 |
| ggcorrplot | 0.1.4.1 | | MASS | 7.3.65 |
| purrr | 1.2.1 | | broom | 1.0.10 |
| car | 3.1.3 | | | |
