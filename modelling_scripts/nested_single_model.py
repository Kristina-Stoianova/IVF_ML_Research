## Nested Cross-Validation — FINAL MODEL (fixed predictors and tune hyperparameters only)

## Run once for the single final model after screening (step 1) and the full nested selection-stability analysis (step 2)
## Difference from the exhaustive nested script:
## No predictor combination search
## The predictor set is FIXED 
## The inner loop tunes hyperparameters only
## Every outer fold now estimates the SAME model, the outer MAE mean +/- SD
## Less optimistic estimate of final model performance and generalisation error

## Nested CV:
## Outer loop (RepeatedKFold, 25 folds) 
## Inner loop (KFold, 5 folds)- hyperparameter tuning only, on the outer-train fold
## Outer test fold untouched until evaluation

## Output:
##   1) CSV one row per outer fold
##   2) Summary = mean +/- SD across folds
##   3) Optional final-fit section — tunes on ALL data and reports the deployable model

import os
import numpy as np
import pandas as pd
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


##CONFIGURATION

MODEL_TO_RUN  = "SVR"    # "Ridge" | "ElasticNet" | "SVR" | "RandomForest" | "XGBoost"
LOG_TRANSFORM = True     # log1p-transform the target before modelling

## FIXED predictor set 

FIXED_NUMERIC     = [
    'Baseline_AMH_log',
    'Baseline_follicles_log',
    'BMI',
]
FIXED_CATEGORICAL = ['Aetiology_group']   # set to [] to exclude aetiology

N_ITER_SEARCH = {
    "Ridge"        : 15,
    "ElasticNet"   : 15,
    "SVR"          : 20,
    "RandomForest" : 20,
    "XGBoost"      : 25,
}

N_JOBS = 6   # match to available CPUs on the HPC node

DATA_PATH = '/data/processed/shortprotocol_firstconsentedcycle.csv'
SAVE_DIR  = 'results'
TARGET    = 'No_mature_eggs'

AETIOLOGY_CATEGORIES = ['Female_factor', 'No_female_factor', 'Unexplained']

FIT_FINAL_ON_ALL_DATA = True   # after nested CV, tune once on all data for the deployable model


##Hyperparameters

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
        'model__reg_alpha'       : loguniform(1e-3, 100),
        'model__gamma'           : uniform(0, 3),
    },

    "SVR": {
        'model__C'      : loguniform(0.01, 1e3),
        'model__epsilon': loguniform(0.01, 1),
        'model__kernel' : ['rbf', 'linear'],
        'model__gamma'  : ['scale', 'auto'],
    },
}

model_instances = {
    "Ridge"        : Ridge(),
    "ElasticNet"   : ElasticNet(max_iter=50000, tol=1e-4, random_state=42),
    "RandomForest" : RandomForestRegressor(random_state=42),
    "XGBoost"      : XGBRegressor(random_state=42, verbosity=0),
    "SVR"          : SVR(),
}


## Load data

df = pd.read_csv(DATA_PATH)
print("Loaded:", df.shape)

df['Aetiology_group'] = pd.Categorical(
    df['Aetiology_group'],
    categories=AETIOLOGY_CATEGORIES,
)


for src, dst in [
    ('Baseline_AMH',             'Baseline_AMH_log'),
    ('Baseline_total_follicles', 'Baseline_follicles_log'),
    ('Baseline_endometrium',     'Baseline_endometrium_log'),
]:
    if src in df.columns:
        df[dst] = np.log1p(df[src])

# Validate that the fixed predictors actually exist

missing = [c for c in FIXED_NUMERIC + FIXED_CATEGORICAL if c not in df.columns]
if missing:
    raise ValueError(f"Fixed predictors not found in data: {missing}")

ALL_FEATURES = FIXED_NUMERIC + FIXED_CATEGORICAL
print(f"Model            : {MODEL_TO_RUN}")
print(f"Fixed numeric    : {FIXED_NUMERIC}")
print(f"Fixed categorical: {FIXED_CATEGORICAL}")
print(f"Saving to        : {SAVE_DIR}\n")

os.makedirs(SAVE_DIR, exist_ok=True)


## Pipeline
## StandardScaler before KNNImputer 

def build_pipeline(estimator, numeric_features, categorical_features):

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


## Nested CV - same structure to the exhaustive nested script 

outer_cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=99)
inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)


## No feature search, fixed predictors, inner loop tunes hyperparameters only

nested_results = []

for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(df), start=1):

    df_train_outer = df.iloc[train_idx].copy()
    df_test_outer  = df.iloc[test_idx].copy()

    X_train_outer = df_train_outer[ALL_FEATURES].copy()
    X_test_outer  = df_test_outer[ALL_FEATURES].copy()

    y_train_outer_raw = df_train_outer[TARGET].copy()
    y_test_outer_raw  = df_test_outer[TARGET].copy()

    if LOG_TRANSFORM:
        y_train_outer = np.log1p(y_train_outer_raw)
        y_test_outer  = np.log1p(y_test_outer_raw)
    else:
        y_train_outer = y_train_outer_raw
        y_test_outer  = y_test_outer_raw

    # Inner loop: hyperparameter tuning on the outer-train fold only
    estimator = clone(model_instances[MODEL_TO_RUN])
    pipe = build_pipeline(estimator, FIXED_NUMERIC, FIXED_CATEGORICAL)

    search = RandomizedSearchCV(
        estimator          =pipe,
        param_distributions=param_distributions[MODEL_TO_RUN],
        n_iter             =N_ITER_SEARCH[MODEL_TO_RUN],
        cv                 =inner_cv,
        scoring            ='neg_mean_absolute_error',
        n_jobs             =N_JOBS,
        refit              =True,      # refits best model
        random_state       =42,
        verbose            =0,
    )
    search.fit(X_train_outer, y_train_outer)

    inner_mae   = -search.best_score_
    final_model = search.best_estimator_

    # Evaluate once on untouched outer test fold
    y_pred_outer = final_model.predict(X_test_outer)

    if LOG_TRANSFORM:
        y_test_eval = np.expm1(y_test_outer)
        y_pred_eval = np.expm1(y_pred_outer)
    else:
        y_test_eval = y_test_outer
        y_pred_eval = y_pred_outer

    outer_mae  = mean_absolute_error(y_test_eval, y_pred_eval)
    outer_rmse = np.sqrt(mean_squared_error(y_test_eval, y_pred_eval))
    outer_r2   = r2_score(y_test_eval, y_pred_eval)

    # Overfitting
    y_train_pred = final_model.predict(X_train_outer)
    if LOG_TRANSFORM:
        y_train_eval      = np.expm1(y_train_outer)
        y_train_pred_eval = np.expm1(y_train_pred)
    else:
        y_train_eval      = y_train_outer
        y_train_pred_eval = y_train_pred

    train_r2    = r2_score(y_train_eval, y_train_pred_eval)
    overfit_gap = round(train_r2 - outer_r2, 4)
    overfit_verdict = (
        'Minimal'  if overfit_gap < 0.1 else
        'Moderate' if overfit_gap < 0.2 else
        'Substantial'
    )

    inner_mae_col = 'inner_MAE_log' if LOG_TRANSFORM else 'inner_MAE_raw'

    nested_results.append({
        'outer_fold'         : outer_fold,
        inner_mae_col        : round(inner_mae, 4),
        'outer_MAE_oocytes'  : round(outer_mae,  4),
        'outer_RMSE_oocytes' : round(outer_rmse, 4),
        'outer_R2_oocytes'   : round(outer_r2,   4),
        'train_R2_oocytes'   : round(train_r2,   4),
        'overfit_gap_R2'     : overfit_gap,
        'overfit_verdict'    : overfit_verdict,
        'best_params'        : str(search.best_params_),
    })

    print(
        f"Outer fold {outer_fold:2d}/25 | "
        f"MAE {outer_mae:5.3f} | RMSE {outer_rmse:5.3f} | R2 {outer_r2:6.3f}"
    )


##save results

nested_df = pd.DataFrame(nested_results)

feat_tag = "_".join(
    [f.replace('Baseline_', '').replace('_log', '') for f in FIXED_NUMERIC]
) + ("_aet" if FIXED_CATEGORICAL else "")

SAVE_PATH = os.path.join(SAVE_DIR, f"{MODEL_TO_RUN}_FINAL_nested_{feat_tag}.csv")
nested_df.to_csv(SAVE_PATH, index=False)
print(f"\nSaved per-fold results to: {SAVE_PATH}")


## Report mean +/- SD across folds - the 25 folds are NOT independent
## Report the SD descriptively

print("\Final Model")
print(f"Model    : {MODEL_TO_RUN}")
print(f"Predictors: {ALL_FEATURES}")
summary = nested_df[[
    'outer_MAE_oocytes', 'outer_RMSE_oocytes', 'outer_R2_oocytes', 'overfit_gap_R2',
]].agg(['mean', 'std', 'min', 'max']).round(4)
print(summary)

mae_m, mae_s = nested_df.outer_MAE_oocytes.mean(), nested_df.outer_MAE_oocytes.std()
r2_m,  r2_s  = nested_df.outer_R2_oocytes.mean(),  nested_df.outer_R2_oocytes.std()
print(f"\nHeadline: MAE = {mae_m:.2f} +/- {mae_s:.2f} oocytes | "
      f"R2 = {r2_m:.2f} +/- {r2_s:.2f}  (n=25 outer folds, non-independent)")
print("Overfit verdicts:", dict(nested_df.overfit_verdict.value_counts()))


## Nested CV above estimates performance


if FIT_FINAL_ON_ALL_DATA:
    print("\nFINAL MODEL (fit on all data)")

    X_all = df[ALL_FEATURES].copy()
    y_all = np.log1p(df[TARGET]) if LOG_TRANSFORM else df[TARGET].copy()

    final_search = RandomizedSearchCV(
        estimator          =build_pipeline(clone(model_instances[MODEL_TO_RUN]),
                                            FIXED_NUMERIC, FIXED_CATEGORICAL),
        param_distributions=param_distributions[MODEL_TO_RUN],
        n_iter             =N_ITER_SEARCH[MODEL_TO_RUN],
        cv                 =inner_cv,
        scoring            ='neg_mean_absolute_error',
        n_jobs             =N_JOBS,
        refit              =True,
        random_state       =42,
    )
    final_search.fit(X_all, y_all)
    print("Chosen hyperparameters:", final_search.best_params_)

    # Linear-kernel coefficient
    try:
        fitted   = final_search.best_estimator_
        model    = fitted.named_steps['model']
        is_linear_svr = (MODEL_TO_RUN == 'SVR'
                         and getattr(model, 'kernel', None) == 'linear')
        if MODEL_TO_RUN in ('Ridge', 'ElasticNet') or is_linear_svr:
            pre = fitted.named_steps['preprocessor']
            try:
                feat_names = pre.get_feature_names_out()
            except Exception:
                feat_names = np.array(ALL_FEATURES)
            coefs = np.ravel(model.coef_)
            print("\nStandardised coefficients (log-oocyte scale):")
            for name, c in sorted(zip(feat_names, coefs), key=lambda t: -abs(t[1])):
                print(f"  {name:35s} {c:+.4f}")
        else:
            print("(Coefficients not reported: non-linear model — use SHAP/permutation importance.)")
    except Exception as e:
        print(f"(Coefficient extraction skipped: {e})")

print("\nDone.")
