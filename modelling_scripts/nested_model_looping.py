## Nested Cross-Validation — Exhaustive Predictor Screening
## Corrected version of the original screening script
## Fixes the train-then-evaluate leakage by moving hyperparameter tuning inside each outer fold so the outer test fold is never touched during tuning
##
## Logic:
##   Outer loop (RepeatedKFold, 25 folds) — honest generalisation estimate
##   Inner loop (KFold, 5 folds) — predictor combo selection + hyperparameter tuning
##   Outer test fold touched exactly once, after selection is finalised
##
## Output:
##   CSV with one row per outer fold — selected config, honest test metrics,
##   overfitting diagnostics, and winning hyperparameters
##
## Usage:
##   Set MODEL_TO_RUN to one model family per run
##   Submit via HPC bash script with n_jobs set to available CPUs
##
## Version Requirements:
##   Python 3.11, scikit-learn, xgboost, numpy, pandas, scipy, tqdm

import os
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import randint, loguniform, uniform

from sklearn.base            import clone
from sklearn.pipeline        import Pipeline
from sklearn.impute          import KNNImputer, SimpleImputer
from sklearn.preprocessing   import StandardScaler, OneHotEncoder
from sklearn.compose         import ColumnTransformer
from sklearn.linear_model    import Ridge, ElasticNet
from sklearn.ensemble        import RandomForestRegressor
from sklearn.svm             import SVR
from sklearn.model_selection import RepeatedKFold, KFold, RandomizedSearchCV
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
from tqdm    import tqdm


## CONFIGURATION
MODEL_TO_RUN  = "Ridge"   # "Ridge" | "ElasticNet" | "XGBoost" | "RandomForest" | "SVR"
LOG_TRANSFORM = True             # log1p-transform target before modelling

N_ITER_SEARCH = {
    "Ridge"        : 15,
    "ElasticNet"   : 15,
    "SVR"          : 20,
    "RandomForest" : 20,
    "XGBoost"      : 25,
}

MIN_FEATURES = 2   # minimum numeric features per combo

DATA_PATH = '/users/k25023936/IVF_research/data/processed/shortprotocol_firstconsentedcycle.csv'
SAVE_DIR  = '/users/k25023936/IVF_research/results'
TARGET    = 'No_mature_eggs'

AETIOLOGY_CATEGORIES = ['Female_factor', 'No_female_factor', 'Unexplained']


## HYPERPARAMETERS

param_distributions = {

    "Ridge": {
        'model__alpha': loguniform(1e-3, 1e5),
    },

    "ElasticNet": {
        'model__alpha'   : loguniform(1e-3, 1e5),
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
        'model__reg_alpha'       : loguniform(1e-3, 100,
        'model__gamma'           : uniform(0, 3),
    },

    "SVR": {
        'model__C'      : loguniform(0.01, 1e3),
        'model__epsilon': loguniform(0.01, 1),
        'model__kernel' : ['rbf', 'linear'],
        'model__gamma'  : ['scale', 'auto'],
    },
}


## MODEL INSTANCES

model_instances = {
    "Ridge"        : Ridge(),
    "ElasticNet"   : ElasticNet(max_iter=50000, tol=1e-4, random_state=42),
    "RandomForest" : RandomForestRegressor(random_state=42),
    "XGBoost"      : XGBRegressor(random_state=42, verbosity=0),
    "SVR"          : SVR(),
}


## LOAD AND PREPARE DATA

df = pd.read_csv(DATA_PATH)
print("Loaded:", df.shape)

df['Aetiology_group'] = pd.Categorical(
    df['Aetiology_group'],
    categories=AETIOLOGY_CATEGORIES,
)

# Log-transform skewed predictors upfront — deterministic, no leakage risk
df['Baseline_AMH_log']       = np.log1p(df['Baseline_AMH'])
df['Baseline_follicles_log'] = np.log1p(df['Baseline_total_follicles'])

CANDIDATE_NUMERIC = [
    'Baseline_AMH_log',
    'Baseline_follicles_log',
    'Age',
    'BMI',
    'Baseline_endometrium',
]

CANDIDATE_CATEGORICAL = ['Aetiology_group']   # set to [] to exclude entirely

MAX_FEATURES = len(CANDIDATE_NUMERIC)

## PIPELINE
## Scaler and imputer are inside the pipeline so they are fitted on the training fold only and applied to the test fold — no leakage
## Tree-based models are scale-invariant but scaling is retained for consistency and to support future pipeline variants

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
                categories    =[AETIOLOGY_CATEGORIES],
                drop          ='first',
                sparse_output =False,
                handle_unknown='ignore',
            )),
        ])
        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer,     numeric_features),
            ('cat', categorical_transformer, categorical_features),
        ])

    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model',        estimator),
    ])


## CV STRATEGY
## outer_cv: 25 folds (5 splits x 5 repeats) generalisation estimate
## inner_cv: 5-fold inside each outer training fold: tuning + combo selection
## Independent random seeds so fold splits are not correlated

outer_cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=99)
inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)


## FEATURE COMBINATION LIST
## Every numeric subset (size MIN_FEATURES to MAX_FEATURES) paired
## with and without Aetiology_group — 52 combos per model family

all_combos = [
    (MODEL_TO_RUN, list(num_feats), cat_feats)
    for r in range(MIN_FEATURES, MAX_FEATURES + 1)
    for num_feats in combinations(CANDIDATE_NUMERIC, r)
    for cat_feats in (
        [[], CANDIDATE_CATEGORICAL] if CANDIDATE_CATEGORICAL else [[]]
    )
]

print(f"Model      : {MODEL_TO_RUN}")
print(f"Total combos per outer fold : {len(all_combos)}")
print(f"Total outer folds           : 25  (5 splits x 5 repeats)")
print(f"Saving to  : {SAVE_DIR}\n")

os.makedirs(SAVE_DIR, exist_ok=True)


## NESTED CV MAIN LOOP

nested_results = []

for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(df), start=1):

    print(f"\nOuter fold {outer_fold}/25")

    # Outer train / test split
    df_train_outer = df.iloc[train_idx].copy()
    df_test_outer  = df.iloc[test_idx].copy()

    y_train_outer_raw = df_train_outer[TARGET].copy()
    y_test_outer_raw  = df_test_outer[TARGET].copy()

    # Log-transform target — fitted on train only, applied separately to test
    if LOG_TRANSFORM:
        y_train_outer = np.log1p(y_train_outer_raw)
        y_test_outer  = np.log1p(y_test_outer_raw)
    else:
        y_train_outer = y_train_outer_raw
        y_test_outer  = y_test_outer_raw

    # Inner loop: exhaustive screening + hyperparameter tuning
    # For each predictor combo, RandomizedSearchCV tunes hyperparameters using
    # inner_cv (5-fold) on df_train_outer only. The outer test fold is never
    # seen here. The combo with the best inner MAE is selected.

    best_inner_mae = np.inf
    best_search    = None
    best_config    = None

    for model_name, num_feats, cat_feats in tqdm(
        all_combos,
        desc=f"  Inner screening (fold {outer_fold})",
        leave=False
    ):
        all_feats         = num_feats + cat_feats
        X_train_outer_sub = df_train_outer[all_feats].copy()

        estimator  = clone(model_instances[model_name])
        param_grid = param_distributions[model_name]

        pipe = build_pipeline(
            estimator           =estimator,
            numeric_features    =num_feats,
            categorical_features=cat_feats,
            model_name          =model_name,
        )

        search = RandomizedSearchCV(
            estimator          =pipe,
            param_distributions=param_grid,
            n_iter             =N_ITER_SEARCH[model_name],
            cv                 =inner_cv,
            scoring            ='neg_mean_absolute_error',
            n_jobs             =6,
            refit              =True,        # refits best config on full df_train_outer subset
            random_state       =42,
            verbose            =0,
        )

        search.fit(X_train_outer_sub, y_train_outer)

        inner_mae = -search.best_score_

        if inner_mae < best_inner_mae:
            best_inner_mae = inner_mae
            best_search    = search
            best_config    = {
                'model'              : model_name,
                'numeric_features'   : list(num_feats),
                'categorical_features': list(cat_feats),
                'all_features'       : list(all_feats),
                'best_params'        : search.best_params_,
            }

    # Evaluate on outer test fold
    # best_search.best_estimator_ is already fitted on df_train_outer
    # (the winning combo's feature subset) because refit=True.
    # We call predict once on the untouched outer test fold.

    selected_features  = best_config['all_features']
    X_test_outer_best  = df_test_outer[selected_features].copy()
    X_train_outer_best = df_train_outer[selected_features].copy()

    final_model  = best_search.best_estimator_
    y_pred_outer = final_model.predict(X_test_outer_best)

    # Back-transform to oocyte units for interpretable metrics
    if LOG_TRANSFORM:
        y_test_eval = np.expm1(y_test_outer)
        y_pred_eval = np.expm1(y_pred_outer)
    else:
        y_test_eval = y_test_outer
        y_pred_eval = y_pred_outer

    outer_mae  = mean_absolute_error(y_test_eval, y_pred_eval)
    outer_rmse = np.sqrt(mean_squared_error(y_test_eval, y_pred_eval))
    outer_r2   = r2_score(y_test_eval, y_pred_eval)

    # Overfitting diagnostic
    # Train predictions from the same final model — purely diagnostic,
    # does not influence selection or reported test metrics

    y_train_pred = final_model.predict(X_train_outer_best)

    if LOG_TRANSFORM:
        y_train_eval      = np.expm1(y_train_outer)
        y_train_pred_eval = np.expm1(y_train_pred)
    else:
        y_train_eval      = y_train_outer
        y_train_pred_eval = y_train_pred

    train_r2    = r2_score(y_train_eval, y_train_pred_eval)
    overfit_gap = round(train_r2 - outer_r2, 4)
    overfit_verdict = (
        'Minimal'     if overfit_gap < 0.1 else
        'Moderate'    if overfit_gap < 0.2 else
        'Substantial'
    )

    # Store results
    inner_mae_col = 'best_inner_MAE_log' if LOG_TRANSFORM else 'best_inner_MAE_raw'

    result_row = {
        'outer_fold'           : outer_fold,

        # Selected configuration
        'selected_model'       : best_config['model'],
        'numeric_features'     : str(best_config['numeric_features']),
        'categorical_features' : str(best_config['categorical_features']),
        'n_features'           : len(selected_features),
        'includes_aetiology'   : len(best_config['categorical_features']) > 0,

        # Inner-loop selection score (log scale if LOG_TRANSFORM)
        inner_mae_col          : round(best_inner_mae, 4),

        # Outer test performance (oocyte units)
        'outer_MAE_oocytes'    : round(outer_mae,  4),
        'outer_RMSE_oocytes'   : round(outer_rmse, 4),
        'outer_R2_oocytes'     : round(outer_r2,   4),

        # Overfitting diagnostics
        'train_R2_oocytes'     : round(train_r2,    4),
        'overfit_gap_R2'       : overfit_gap,
        'overfit_verdict'      : overfit_verdict,

        # Winning hyperparameters for this fold
        'best_params'          : str(best_config['best_params']),
    }

    nested_results.append(result_row)

    print(
        f"  Selected : {best_config['model']} | "
        f"features : {best_config['all_features']} | "
        f"outer R² : {round(outer_r2, 3)} | "
        f"outer MAE : {round(outer_mae, 3)} oocytes"
    )


## SAVE RESULTS .csv

nested_df = pd.DataFrame(nested_results)

SAVE_PATH_NESTED = os.path.join(
    SAVE_DIR,
    f"{MODEL_TO_RUN}_nested_exhaustive_screen.csv"
)

nested_df.to_csv(SAVE_PATH_NESTED, index=False)

print(f"\nNested CV complete — saved to: {SAVE_PATH_NESTED}")


## SUMMARY 

print("\nOuter-fold performance summary (25 folds):")
print(
    nested_df[[
        'outer_MAE_oocytes',
        'outer_RMSE_oocytes',
        'outer_R2_oocytes',
        'overfit_gap_R2',
    ]].agg(['mean', 'std']).round(4)
)

print("\nSelected model frequency across outer folds:")
print(nested_df['selected_model'].value_counts())

print("\nSelected feature combo frequency across outer folds:")
print(
    nested_df.groupby(['selected_model', 'numeric_features', 'categorical_features'])
    .size()
    .reset_index(name='fold_count')
    .sort_values('fold_count', ascending=False)
    .head(10)
    .to_string(index=False)
)