import os
import glob
from scipy.stats import randint, loguniform
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import loguniform

from sklearn.pipeline           import Pipeline
from sklearn.impute             import KNNImputer, SimpleImputer
from sklearn.preprocessing      import StandardScaler, OneHotEncoder
from sklearn.compose            import ColumnTransformer
from sklearn.linear_model       import Ridge
from sklearn.ensemble           import RandomForestRegressor
from sklearn.model_selection    import (RepeatedKFold, KFold,
                                        RandomizedSearchCV,
                                        cross_val_score, cross_validate,
                                        cross_val_predict)
from sklearn.metrics            import mean_absolute_error, r2_score

from xgboost  import XGBRegressor
from lightgbm import LGBMRegressor
from tqdm     import tqdm

MODEL_TO_RUN  = "Ridge"        # "Ridge" | "XGBoost" | "RandomForest" | "LightGBM"
LOG_TRANSFORM = True           # log1p-transform the target
N_ITER_SEARCH = 100             # RandomizedSearchCV iterations per combo
MIN_FEATURES  = 3            # minimum numeric features per combo
# MAX_FEATURES is set automatically to len(CANDIDATE_NUMERIC) — see below

DATA_PATH  = '/IVF_research/data/processed/shortprotocol_firstconsentedcycle.csv'
SAVE_DIR   = '/IVF_research/results'
TARGET     = 'No_mature_eggs'

AETIOLOGY_CATEGORIES = [
    'Egg_donor', 'No_female_factor', 'Female_factor', 'Unexplained', 'Other'
]

df = pd.read_csv(DATA_PATH)
print("Loaded:", df.shape)

# Set reference category order for aetiology (drop='first' drops 'Egg_donor')
df['Aetiology_group'] = pd.Categorical(
    df['Aetiology_group'],
    categories=AETIOLOGY_CATEGORIES
)

# Log-transform continuous candidates (avoids skew, keeps original cols intact)
df['Baseline_AMH_log']        = np.log1p(df['Baseline_AMH'])
df['Baseline_follicles_log']  = np.log1p(df['Baseline_total_follicles'])
df['BMI_log']                 = np.log1p(df['BMI'])
df['Baseline_endometrium_log'] = np.log1p(df['Baseline_endometrium'])
df['Final_follicles_lessthan_11.9_log'] = np.log1p(df['Final_follicles_lessthan_11.9'])

# Candidate feature pools
# Add or remove features here — the loop tries ALL subsets automatically
CANDIDATE_NUMERIC = [
    'Baseline_AMH_log',
    'Baseline_follicles_log',
    'Age',
    'BMI_log',
    'Baseline_endometrium_log',
    'Starting_gonadotropin_dose',
    'Total_gonadotropin_dose',
    'FSH_used',
    'hMG_used',
    'GnRH_antagonist_total_dose',
    'GnRH_antagonist_duration',
    'Final_follicles_lessthan_11.9_log'
]

# Aetiology_group is tested alongside every numeric combo (with AND without)
CANDIDATE_CATEGORICAL = ['Aetiology_group']   # set to [] to exclude entirely

MAX_FEATURES = len(CANDIDATE_NUMERIC)         # try all numeric subset sizes

# Target
y_raw = df[TARGET].copy()
y     = np.log1p(y_raw) if LOG_TRANSFORM else y_raw
print(f"Target {'log-transformed' if LOG_TRANSFORM else 'untransformed'}")
print(f"Missing in target: {y_raw.isnull().sum()}")

# Model configuration + hyperparameter grid settings
model_configs = {
   "Ridge": (
      Ridge(),
       {
          #loguniform samples on a log scale — right for a param
          #spanning several orders of magnitude (0.001 to 1000)
           'model__alpha': loguniform(0.01, 1000)
       }
    ),
#"RandomForest": (
 #    RandomForestRegressor(random_state=42),
 #    {
 #     'model__n_estimators':      randint(100, 600),
 #     'model__max_depth':         randint(2, 8),
 #     'model__min_samples_split': randint(5, 15),
 #     'model__min_samples_leaf':  randint(3, 9),
 #     'model__max_features':      ['sqrt', 'log2'],
 #     }
#),
#    "XGBoost": (
 #       XGBRegressor(random_state=42, verbosity=0),
 #       {
#            'model__n_estimators':     [100, 200, 300],
#            'model__max_depth':        [3, 5, 7],
#            'model__learning_rate':    loguniform(0.01, 0.3),
 #           'model__subsample':        [0.7, 0.8, 1.0],
 #           'model__colsample_bytree': [0.7, 0.8, 1.0],
 #       }
 #   ),
 #   "LightGBM": (
 #       LGBMRegressor(random_state=42, verbose=-1),
  #      {
 #           'model__n_estimators':  [100, 200, 300],
  #          'model__max_depth':     [-1, 5, 7],
 #           'model__learning_rate': loguniform(0.01, 0.3),
  #          'model__num_leaves':    [20, 31, 50],
 #           'model__subsample':     [0.7, 0.8, 1.0],
 #       }
 #   ),
}

## Pipeline builder
## Ridge gets StandardScaler --> required because penalty is scale-sensitive
## Tree models do not need scaler
## KNN imputer inside the pipeline to prevent data leakage
## Categorical gets its own imputer

def build_pipeline(estimator, numeric_features, categorical_features, model_name):

    use_scaler = (model_name == "Ridge")

    if use_scaler:
        numeric_transformer = Pipeline([
            ('imputer', KNNImputer(n_neighbors=5, weights='distance')),
            ('scaler',  StandardScaler()),
        ])
    else:
        numeric_transformer = Pipeline([
            ('imputer', KNNImputer(n_neighbors=5, weights='distance')),
        ])

    if len(categorical_features) == 0:
        # No categorical — numeric transformer acts as the full preprocessor
        preprocessor = numeric_transformer
    else:
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot',  OneHotEncoder(
                categories   = [AETIOLOGY_CATEGORIES],
                drop         = 'first',         # drops 'Egg_donor' as reference
                sparse_output= False,
                handle_unknown='ignore'
            ))
        ])
        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer,     numeric_features),
            ('cat', categorical_transformer, categorical_features),
        ])

    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model',         estimator),
    ])

## CV strategy
kf_search  = RepeatedKFold(n_splits=5, n_repeats=3,  random_state=42)
kf_eval    = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
kf_predict = KFold(n_splits=5, shuffle=True,          random_state=42)

##output
os.makedirs(SAVE_DIR, exist_ok=True)
SAVE_PATH = os.path.join(SAVE_DIR, f'results_Ridge1.csv')
print(f"Saving to     : {SAVE_PATH}\n")

# Build feature combo list --> every numeric subset paired with and without categorical
all_combos = [
    (model_name, list(num_feats), cat_feats)
    for model_name in model_configs.keys()
    for r in range(MIN_FEATURES, MAX_FEATURES + 1)
    for num_feats in combinations(CANDIDATE_NUMERIC, r)
    for cat_feats in (
        [[], CANDIDATE_CATEGORICAL] if CANDIDATE_CATEGORICAL else [[]]
    )
]

os.makedirs(SAVE_DIR, exist_ok=True)
print(f"\nTotal runs : {len(all_combos)}")
print(f"Saving to  : {SAVE_PATH}\n")

for model_name, num_feats, cat_feats in tqdm(all_combos, desc="All models"):

    all_feats = num_feats + cat_feats
    X_sub     = df[all_feats].copy()

    estimator, param_grid = model_configs[model_name]
    pipe = build_pipeline(estimator, num_feats, cat_feats, model_name)

    # Hyperparameter search with RandomizedSearch CV
    search = RandomizedSearchCV(
        estimator          = pipe,
        param_distributions= param_grid,
        n_iter             = N_ITER_SEARCH,
        cv                 = kf_search,
        scoring            = 'neg_mean_absolute_error',
        n_jobs             = -1,
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
            cv=kf_eval, scoring=metric, n_jobs=-1
        )
        if metric != 'r2':
            scores = -scores
        eval_scores[f'CV_{label}_mean'] = round(scores.mean(), 4)
        eval_scores[f'CV_{label}_std']  = round(scores.std(),  4)

    # Check overfitting
    cv_results = cross_validate(
        best_pipeline, X_sub, y,
        cv                = kf_eval,
        scoring           = 'r2',
        return_train_score= True,
        n_jobs            = -1,
    )
    train_r2    = round(cv_results['train_score'].mean(), 4)
    val_r2      = round(cv_results['test_score'].mean(),  4)
    overfit_gap = round(train_r2 - val_r2, 4)
    overfit_verdict = (
        'Minimal'     if overfit_gap < 0.1 else
        'Moderate'    if overfit_gap < 0.2 else
        'Substantial'
    )

    # Out of fold predictions --> back-transform to oocyte units non-log
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

    # ASSEMBLE
    result_row = {
        # Identity
        'model'               : model_name,
        'numeric_features'    : str(num_feats),
        'categorical_features': str(cat_feats),
        'n_features'          : len(all_feats),
        'includes_aetiology'  : len(cat_feats) > 0,
        'log_transform'       : LOG_TRANSFORM,
        # Hyperparameters
        'param_alpha'            : best_params_dict.get('alpha',             np.nan),
        'param_n_estimators'     : best_params_dict.get('n_estimators',      np.nan),
        'param_max_depth'        : best_params_dict.get('max_depth',         np.nan),
        'param_min_samples_split': best_params_dict.get('min_samples_split', np.nan),
        'param_max_features'     : best_params_dict.get('max_features',      np.nan),
        'param_learning_rate'    : best_params_dict.get('learning_rate',     np.nan),
        'param_subsample'        : best_params_dict.get('subsample',         np.nan),
        'param_colsample_bytree' : best_params_dict.get('colsample_bytree',  np.nan),
        'param_num_leaves'       : best_params_dict.get('num_leaves',        np.nan),
        # Search MAE
        'search_best_MAE'     : round(-search.best_score_, 4),
        # CV metrics (log scale)
        'CV_MAE_mean'         : eval_scores['CV_MAE_mean'],
        'CV_MAE_std'          : eval_scores['CV_MAE_std'],
        'CV_RMSE_mean'        : eval_scores['CV_RMSE_mean'],
        'CV_RMSE_std'         : eval_scores['CV_RMSE_std'],
        'CV_R2_mean'          : eval_scores['CV_R2_mean'],
        'CV_R2_std'           : eval_scores['CV_R2_std'],
        # OOF metrics (oocyte units)
        'OOF_MAE_oocytes'     : oof_mae,
        'OOF_RMSE_oocytes'    : oof_rmse,
        'OOF_R2_oocytes'      : oof_r2,
        # Overfitting
        'train_R2'            : train_r2,
        'val_R2'              : val_r2,
        'overfit_gap_R2'      : overfit_gap,
        'overfit_verdict'     : overfit_verdict,
    }

    pd.DataFrame([result_row]).to_csv(
        SAVE_PATH,
        mode  = 'a',
        header= not os.path.exists(SAVE_PATH),
        index = False,
    )

print(f"\nDone. Results saved to {SAVE_PATH}")