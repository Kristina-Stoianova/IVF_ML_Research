## Screening Pipeline

## Iteratively tests every combination of candidate predictors for a single model family at a time
## Output = .csv of feature combinations and their performance metrics
## Screening stage --> absolute metrics may be slightly optimistic as hyperparameters are tuned on the full dataset before CV evaluation

## How to use:
## set MODEL_TO_RUN to one model family per run
## Rename SAVE_PATH before each run or results for new model will be overwritten
## Submit via HPC bash script with n_jobs set to available CPUs

## Scalability
## Hyperparameter search space can be modified
## Number of iterations, CV number of splits, and repeats can be modified as sample size grows
## User can choose features/predictor variables and select the minimum number in each model
## Random state set to 42 for tree-based models - reproducibility
## n_jobs = number of CPUs, can specify in bash running script


import os
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import t, randint, loguniform, uniform

from sklearn.pipeline           import Pipeline
from sklearn.impute             import KNNImputer, SimpleImputer
from sklearn.preprocessing      import StandardScaler, OneHotEncoder
from sklearn.compose            import ColumnTransformer
from sklearn.linear_model       import Ridge, ElasticNet
from sklearn.ensemble           import RandomForestRegressor
from sklearn.svm                import SVR
from sklearn.model_selection    import (RepeatedKFold, KFold,
                                        RandomizedSearchCV,
                                        cross_val_score, cross_validate,
                                        cross_val_predict)
from sklearn.metrics            import mean_absolute_error, r2_score

from xgboost  import XGBRegressor
from tqdm     import tqdm

def cv_confidence_interval(scores, confidence=0.95):
    n = len(scores)
    mean = scores.mean()
    se = scores.std() / np.sqrt(n)
    t_crit = t.ppf((1 + confidence) / 2, df=n - 1)
    margin = t_crit * se
    return round(mean - margin, 4), round(mean + margin, 4)

## CONFIGURATION

MODEL_TO_RUN  = "XGBoost"        # "Ridge" | "ElasticNet" | "XGBoost" | "RandomForest" | "SVR"
LOG_TRANSFORM = True           # log1p-transform the target before modelling: OOF predictions are back trasnformed to original oocyte units for interpretation
N_ITER_SEARCH = { # RandomizedSearchCV iterations per combination
    "Ridge"        : 15,
    "ElasticNet"   : 20,
    "SVR"          : 30,
    "RandomForest" : 35,
    "XGBoost"      : 40,
}

MIN_FEATURES  = 2            # minimum numeric features per combo
# MAX_FEATURES is set automatically to len(CANDIDATE_NUMERIC) below

DATA_PATH  = '/data/processed/shortprotocol_firstconsentedcycle.csv'
SAVE_DIR   = '/results'
TARGET     = 'No_mature_eggs'

AETIOLOGY_CATEGORIES = [
    'Female_factor', 'No_female_factor', 'Unexplained'
]

## SCALING CONFIGURATION
## StandardScaler is applied inside every pipeline:
## KNNImputer requires scaled input for correct distance calculations
## Scaling applied before imputation
## Scaler and imputation applied on training fold to prevent leakage
## Tree-based models do not require scaling
## However tree-based models are still imputed with KNN which requires scaling
## Simplifies pipeline without affecting tree-based models

## Scaling block is retained to make scaling intent explicit per model and to support future pipeline variants where scaling may be conditionally applied

REQUIRES_SCALING = {
    "Ridge"         : True, # linear model - sensitive to feature scale
    "ElasticNet"    : True, # linear model - sensitive to feature scale
    "RandomForest"  : False, # tree based
    "XGBoost"       : False, # tree based
    "SVR"           : True, # kernel based - sensitive to feature scale
}


## MODEL CONFIGURATION - HYPERPARAMETER SEARCH SPACES WITH RandomizedSearchCV


param_distributions = {
 
    "Ridge": {
        'model__alpha': loguniform(1e-3, 1e5),
    },
 
    "ElasticNet": {
	'model__alpha': loguniform(1e-3, 1e5),
	'model__l1_ratio': uniform(0.05, 0.90),
    },
 
    "RandomForest": {
        'model__n_estimators'     : randint(100, 400),
        'model__max_depth'        : randint(2, 7),
        'model__min_samples_split': randint(5, 20),
        'model__min_samples_leaf' : randint(3, 12),
        'model__max_features'     : ['sqrt', 'log2', 0.7, 1.0],
    },
 
    "XGBoost": {
        'model__learning_rate'   : loguniform(0.01, 0.3),
        'model__n_estimators'    : randint(50, 400),
        'model__max_depth'       : randint(2, 5),
        'model__min_child_weight': randint(2, 15),
        'model__subsample'       : uniform(0.6, 0.4),
        'model__colsample_bytree': uniform(0.6, 0.4),
        'model__reg_lambda'      : loguniform(0.1, 1000),
        'model__reg_alpha'       : loguniform(1e-3, 1e4),
        'model__gamma'           : uniform(0, 3),
    },
 
    "SVR": {
        'model__C'      : loguniform(0.01, 1e5),
        'model__epsilon': loguniform(0.01, 1000),
        'model__kernel' : ['rbf', 'linear'],
        'model__gamma'  : ['scale', 'auto'],
    },
}

## MODELS

model_instances = {
    "Ridge"         : Ridge(),
    "ElasticNet"    : ElasticNet(max_iter=50000, tol=1e-4, random_state=42),
    "RandomForest"  : RandomForestRegressor(random_state=42),
    "XGBoost"       : XGBRegressor(random_state=42, verbosity=0),
    "SVR"           : SVR(),

}


## LOAD DATA

df = pd.read_csv(DATA_PATH)
print("Loaded:", df.shape)

df['Aetiology_group'] = pd.Categorical(
    df['Aetiology_group'],
    categories = AETIOLOGY_CATEGORIES,
)


# Log-transform skewed continuous predictors (keeps original cols intact)
df['Baseline_AMH_log']        = np.log1p(df['Baseline_AMH'])
df['Baseline_follicles_log']  = np.log1p(df['Baseline_total_follicles'])
#df['BMI_log']                 = np.log1p(df['BMI'])
df['Baseline_endometrium_log'] = np.log1p(df['Baseline_endometrium'])

# Candidate feature pools
# Add or remove features here — the loop tries ALL subsets automatically
CANDIDATE_NUMERIC = [
    'Baseline_AMH_log',
    'Baseline_follicles_log',
    'Age',
    'BMI',
    'Baseline_endometrium_log',
]

# Aetiology_group is tested alongside every numeric combo (with AND without)

CANDIDATE_CATEGORICAL = ['Aetiology_group']   # set to [] to exclude entirely

MAX_FEATURES = len(CANDIDATE_NUMERIC)         # try all numeric subset sizes

# Target
y_raw = df[TARGET].copy()
y     = np.log1p(y_raw) if LOG_TRANSFORM else y_raw
print(f"Target {'log-transformed' if LOG_TRANSFORM else 'untransformed'}")
print(f"Missing in target: {y_raw.isnull().sum()}")


## PIPELINE BUILDER
## StandardScaler applied as the first step
## KNNImputer is fitted on each training fold and applies to the corresponding test fold - preventing leakage of test set info into imputed values

## Categorical features receives mode imputation and are one hot encoded with the first category dropped 

def build_pipeline(estimator, numeric_features, categorical_features, model_name):

    numeric_transformer = Pipeline([
        ('scaler',  StandardScaler()),
        ('imputer', KNNImputer(n_neighbors=5, weights='distance', metric='nan_euclidean')),
    ])

    if len(categorical_features) == 0:
        preprocessor = numeric_transformer
    else:
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot',  OneHotEncoder(
                categories    = [AETIOLOGY_CATEGORIES],
                drop          = 'first',
                sparse_output = False,
                handle_unknown= 'ignore',
            )),
        ])
        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer,     numeric_features),
            ('cat', categorical_transformer, categorical_features),
        ])

    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model',         estimator),
    ])

## CV STRATEGY
## Three separate CV objects are used with independent random seeds to ensure
## fold splits are not correlated across roles:
##
##   kf_search  — inner loop for RandomizedSearchCV hyperparameter tuning
##
##   kf_eval    — outer evaluation loop for stable metric estimates.
##                5 repeats is sufficient for screening
##                NOTE: hyperparameters are tuned on the full dataset before
##                this evaluation, so metrics are consistent for ranking but
##                slightly optimistic
##
##   kf_predict — single-pass KFold for cross_val_predict. Repeated folds are
##                incompatible with cross_val_predict because each sample must
##                receive exactly one prediction. Single-pass OOF predictions
##                have higher variance than kf_eval metrics; use kf_eval metrics
##                as the primary ranking criterion.


kf_search  = RepeatedKFold(n_splits=5, n_repeats=3,  random_state=42)
kf_eval    = RepeatedKFold(n_splits=5, n_repeats=5, random_state=99)
kf_predict = KFold(n_splits=5, shuffle=True,          random_state=7) ##can make n_split=10

## OUTPUT SETUP

os.makedirs(SAVE_DIR, exist_ok=True)
SAVE_PATH = os.path.join(SAVE_DIR, f'{MODEL_TO_RUN}_THESIS_SCREEN.csv')
# Rename SAVE_PATH before each model run — results are appended, not overwritten
 
print(f"Model     : {MODEL_TO_RUN}")
print(f"Saving to : {SAVE_PATH}\n")

## BUILD FEATURE COMBINATION LIST - every numeric subset paired with and without categorical feature
all_combos = [
    (MODEL_TO_RUN, list(num_feats), cat_feats)
    for r in range(MIN_FEATURES, MAX_FEATURES + 1)
    for num_feats in combinations(CANDIDATE_NUMERIC, r)
    for cat_feats in (
        [[], CANDIDATE_CATEGORICAL] if CANDIDATE_CATEGORICAL else [[]]
    )
]

print(f"\nTotal runs : {len(all_combos)}")

## MAIN LOOP

for model_name, num_feats, cat_feats in tqdm(all_combos, desc="All model screening"):

    all_feats = num_feats + cat_feats
    X_sub     = df[all_feats].copy()

    estimator = model_instances[model_name]
    param_grid = param_distributions[model_name]

    pipe = build_pipeline(estimator, num_feats, cat_feats, model_name)

    # Hyperparameter search with RandomizedSearch CV
    search = RandomizedSearchCV(
        estimator          = pipe,
        param_distributions= param_grid,
        n_iter             = N_ITER_SEARCH[model_name],
        cv                 = kf_search,
        scoring            = 'neg_mean_absolute_error',
        n_jobs             = 6,
        refit              = True,
        random_state       = 42,
        verbose            = 0,
    )
    search.fit(X_sub, y)
    best_pipeline = search.best_estimator_

    best_params_dict = {
        k.replace('model__', ''): v
        for k, v in search.best_params_.items()
    }

    # Cross validation evaluation metrics
    eval_scores = {}
    for metric, label in [
        ('neg_mean_absolute_error',     'MAE'),
        ('neg_root_mean_squared_error', 'RMSE'),
        ('r2',                          'R2'),
    ]:
        scores = cross_val_score(
            best_pipeline, X_sub, y,
            cv=kf_eval, scoring=metric, n_jobs=6
        )
        if metric != 'r2':
            scores = -scores

        ci_lower, ci_upper = cv_confidence_interval(scores)
        suffix = '_log' if metric != 'r2' else ''

        eval_scores[f'CV_{label}_mean'] = round(scores.mean(), 4)
        eval_scores[f'CV_{label}_std']  = round(scores.std(),  4)
        eval_scores[f'CV_{label}_CI95_lower{suffix}'] = ci_lower
        eval_scores[f'CV_{label}_CI95_upper{suffix}'] = ci_upper

    # Check overfitting
    cv_results = cross_validate(
        best_pipeline, X_sub, y,
        cv                = kf_eval,
        scoring           = 'r2',
        return_train_score= True,
        n_jobs            = 6,
    )
    train_r2    = round(cv_results['train_score'].mean(), 4)
    val_r2      = round(cv_results['test_score'].mean(),  4)
    overfit_gap = round(train_r2 - val_r2, 4)
    overfit_verdict = (
        'Minimal'     if overfit_gap < 0.1 else
        'Moderate'    if overfit_gap < 0.2 else
        'Substantial'
    )

    # Out of fold predictions and back-transform target (back to oocyte units if log transformed during configuration)
    # Single K-Fold - each sample receives 1 prediction, not possible to use Repeated K-Fold
    # Back transformation allows MAE/RMSE to be interpreted clinically 
    y_pred_cv = cross_val_predict(best_pipeline, X_sub, y, cv=kf_predict)
    if LOG_TRANSFORM:
        y_eval      = np.expm1(y)
        y_pred_eval = np.expm1(y_pred_cv)
    else:
        y_eval      = y
        y_pred_eval = y_pred_cv

    oof_mae  = round(mean_absolute_error(y_eval, y_pred_eval), 4)
    oof_rmse = round(np.sqrt(np.mean((y_eval - y_pred_eval) ** 2)), 4)
    oof_r2   = round(r2_score(y_eval, y_pred_eval), 4)


## ASSEMBLE RESULTS .csv
    
    result_row = {
        # Identity
        'model'               : model_name,
        'numeric_features'    : str(num_feats),
        'categorical_features': str(cat_feats),
        'n_features'          : len(all_feats),
        'includes_aetiology'  : len(cat_feats) > 0,
        'log_transform'       : LOG_TRANSFORM,
        'scaling_applied'     : REQUIRES_SCALING[model_name],
 
        # Hyperparameters (NaN if not used by this model)
        # Ridge
        'param_alpha'             : best_params_dict.get('alpha',              np.nan),
	# ElasticNet
	'param_l1_ratio'          : best_params_dict.get('l1_ratio',           np.nan),
        # RandomForest
        'param_n_estimators'      : best_params_dict.get('n_estimators',       np.nan),
        'param_max_depth'         : best_params_dict.get('max_depth',          np.nan),
        'param_min_samples_split' : best_params_dict.get('min_samples_split',  np.nan),
        'param_min_samples_leaf'  : best_params_dict.get('min_samples_leaf',   np.nan),
        'param_max_features'      : best_params_dict.get('max_features',       np.nan),
        # XGBoost
        'param_learning_rate'     : best_params_dict.get('learning_rate',      np.nan),
        'param_subsample'         : best_params_dict.get('subsample',          np.nan),
        'param_colsample_bytree'  : best_params_dict.get('colsample_bytree',   np.nan),
        'param_reg_lambda'        : best_params_dict.get('reg_lambda',         np.nan),
        'param_reg_alpha'         : best_params_dict.get('reg_alpha',          np.nan),
        'param_min_child_weight'  : best_params_dict.get('min_child_weight',   np.nan),
        # XGBoost gamma - stored separately to SVR gamma
        'param_gamma_xgb'             : best_params_dict.get('gamma',              np.nan) if model_name == 'XGBoost' else np.nan,
        # SVR
        'param_C'                 : best_params_dict.get('C',                  np.nan),
        'param_epsilon'           : best_params_dict.get('epsilon',            np.nan),
        'param_kernel'            : best_params_dict.get('kernel',             np.nan),
        # SVR gamma - stored separately to XGBoost gamma
        'param_gamma_svr'         : best_params_dict.get('gamma',              np.nan) if model_name == 'SVR' else np.nan,
 
        # Search result
        'search_best_MAE_log'         : round(-search.best_score_, 4),
 
        # CV metrics (log-scale target)
        'CV_MAE_mean_log'             : eval_scores['CV_MAE_mean'],
        'CV_MAE_std_log'              : eval_scores['CV_MAE_std'],
        'CV_RMSE_mean_log'            : eval_scores['CV_RMSE_mean'],
        'CV_RMSE_std_log'             : eval_scores['CV_RMSE_std'],
        'CV_R2_mean'              : eval_scores['CV_R2_mean'],
        'CV_R2_std'               : eval_scores['CV_R2_std'],
        'CV_MAE_CI95_lower_log'  : eval_scores['CV_MAE_CI95_lower_log'],
        'CV_MAE_CI95_upper_log'  : eval_scores['CV_MAE_CI95_upper_log'],
        'CV_RMSE_CI95_lower_log' : eval_scores['CV_RMSE_CI95_lower_log'],
        'CV_RMSE_CI95_upper_log' : eval_scores['CV_RMSE_CI95_upper_log'],
        'CV_R2_CI95_lower'       : eval_scores['CV_R2_CI95_lower'],
        'CV_R2_CI95_upper'       : eval_scores['CV_R2_CI95_upper'],

        # OOF metrics (original oocyte units)
        'OOF_MAE_oocytes'         : oof_mae,
        'OOF_RMSE_oocytes'        : oof_rmse,
        'OOF_R2_oocytes'          : oof_r2,
 
        # Overfitting diagnostics
        'train_R2'                : train_r2,
        'val_R2'                  : val_r2,
        'overfit_gap_R2'          : overfit_gap,
        'overfit_verdict'         : overfit_verdict,
    }
 
    pd.DataFrame([result_row]).to_csv(
        SAVE_PATH,
        mode  = 'a',
        header= not os.path.exists(SAVE_PATH),
        index = False,
    )
 
print(f"\nDone. Results saved to {SAVE_PATH}")
