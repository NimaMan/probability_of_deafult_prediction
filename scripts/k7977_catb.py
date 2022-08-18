import gc
import warnings
warnings.filterwarnings('ignore')
import scipy as sp
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import itertools
import os
import random
import joblib
import itertools
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
from itertools import combinations

import pd.metric as metric
from pd.utils import merge_with_pred_df
from pd.params import *


def train_and_evaluate(train, test, params, model_name, n_folds=5, seed=42):
    cat_features = []
    for cf in CATCOLS:
        cat_features.append(f"{cf}_first")
        cat_features.append(f"{cf}_last")

    for cat_col in cat_features:
        encoder = LabelEncoder()
        train[cat_col] = encoder.fit_transform(train[cat_col])
        test[cat_col] = encoder.transform(test[cat_col])
    
    # Round last float features to 2 decimal place
    num_cols = list(train.dtypes[(train.dtypes == 'float32') | (train.dtypes == 'float64')].index)
    num_cols = [col for col in num_cols if 'last' in col]
    for col in num_cols:
        train[col + '_round2'] = train[col].round(2)
        test[col + '_round2'] = test[col].round(2)
    
    # Get the difference between last and mean
    num_cols = [col for col in train.columns if 'last' in col]
    num_cols = [col[:-5] for col in num_cols if 'round' not in col]
    for col in num_cols:
        try:
            train[f'{col}_last_mean_diff'] = train[f'{col}_last'] - train[f'{col}_mean']
            test[f'{col}_last_mean_diff'] = test[f'{col}_last'] - test[f'{col}_mean']
        except:
            pass
    
    # Transform float64 and float32 to float16
    num_cols = list(train.dtypes[(train.dtypes == 'float32') | (train.dtypes == 'float64')].index)
    for col in tqdm(num_cols):
        train[col] = train[col].astype(np.float16)
        test[col] = test[col].astype(np.float16)
    # Get feature list
    features = [col for col in train.columns if col not in ['customer_ID', "target"]]
 
    # Create a numpy array to store test predictions
    test_predictions = np.zeros(len(test))
    # Create a numpy array to store out of folds predictions
    oof_predictions = np.zeros(len(train))
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (trn_ind, val_ind) in enumerate(kfold.split(train, train["target"])):
        print(' ')
        print('-'*50)
        print(f'Training fold {fold} with {len(features)} features...')
        x_train, x_val = train[features].iloc[trn_ind], train[features].iloc[val_ind]
        y_train, y_val = train["target"].iloc[trn_ind], train["target"].iloc[val_ind]
        
        model = CatBoostClassifier(iterations=105000, random_state=seed, task_type=params["device"])
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], cat_features=cat_features,  verbose=100)
        
        # Save best model
        joblib.dump(model, MODELDIR + f'lgbmk7977_fold{fold}_seed{seed}.pkl')
        # Predict validation
        val_pred = model.predict_proba(x_val)[:, 1] # Predict validation
        # Add to out of folds array
        oof_predictions[val_ind] = val_pred
        # Predict the test set
        test_pred = model.predict_proba(test[features])[:, 1]
        test_predictions += test_pred / n_folds
        # Compute fold metric
        score, gini, recall = metric.amex_metric(y_val.values.reshape(-1, ), val_pred, return_components=True)
        print(f'Our fold {fold} CV score is {score}')
        del x_train, x_val, y_train, y_val
        gc.collect()
    # Compute out of folds metric
    score, gini, recall = metric.amex_metric(train["target"].values.reshape(-1, ), oof_predictions, return_components=True)  # Compute out of folds metric
    print(f'Our out of folds CV score is {score}, Gini {gini}, recall {recall}')
    
    # Create a dataframe to store out of folds predictions
    oof_df = pd.DataFrame({'customer_ID': train['customer_ID'], model_name: oof_predictions})
    merge_with_pred_df(oof_df, type="train")
    # Create a dataframe to store test prediction
    test_df = pd.DataFrame({'customer_ID': test['customer_ID'], model_name: test_predictions})
    test_df.to_csv(OUTDIR + f'est_lgbm_baseline_{n_folds}fold_seed{seed}.csv', index=False)
    merge_with_pred_df(test_df, type="test")
 

if __name__ == "__main__":

    params = {
        "device": "GPU"
        }

    train = pd.read_parquet(OUTDIR + 'train_k7977.parquet')
    test = pd.read_parquet(OUTDIR + 'test_k7977.parquet')
    for seed in [42, 52, 62, 82]:
        model_name = f'K7977_catb_{seed}'
        train_and_evaluate(train, test, params, model_name=model_name,)