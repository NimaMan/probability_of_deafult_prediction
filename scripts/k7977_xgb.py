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
import xgboost as xgb
from itertools import combinations

import pd.metric as metric
from pd.utils import merge_with_pred_df
from pd.gmb_utils import xgb_amex
from pd.params import *



def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


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
        
        xgb_train = xgb.DMatrix(x_train, y_train, )
        xgb_valid = xgb.DMatrix(x_val, y_val, )

        model =  xgb.train(
            params = params,
            dtrain = xgb_train,
            maximize = True,
            num_boost_round = 10500,
            evals = [(xgb_train, 'train'), (xgb_valid, 'valid')],
            early_stopping_rounds = 100,
            verbose_eval = 100,
            feval = xgb_amex
            )
        
        # Save best model
        joblib.dump(model, MODELDIR + f'lgbmk7977_fold{fold}_seed{seed}.pkl')
        # Predict validation
        val_pred = model.predict(xgb.DMatrix(x_val)) # Predict validation
        # Add to out of folds array
        oof_predictions[val_ind] = val_pred
        # Predict the test set
        test_pred = model.predict(xgb.DMatrix(test[features]))
        test_predictions += test_pred / n_folds
        # Compute fold metric
        score, gini, recall = metric.amex_metric(y_val.values.reshape(-1, ), val_pred, return_components=True)
        print(f'Our fold {fold} CV score is {score}')
        del x_train, x_val, y_train, y_val, xgb_train, xgb_valid
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

   

    train = pd.read_parquet(OUTDIR + 'train_k7977.parquet')
    test = pd.read_parquet(OUTDIR + 'test_k7977.parquet')
    for seed in [52, 62, 82]:
        model_name = f'K7977_xgb_{seed}'
        params = {
        'objective': 'binary:logistic',
        #'metric': "binary_logloss",
        #'eval_metric':'logloss',
        'disable_default_eval_metric': 1,
        'seed': seed,
        'learning_rate': 0.05,
        'n_jobs': -1,
        'colsample_bytree': 0.6,
        'gamma':1.5,
        'min_child_weight':8,
        'lambda':70,
        'max_bin': 255,  # Deafult is 255
        'tree_method':'gpu_hist',
        'predictor':'gpu_predictor',
        'max_depth': 4, 
        'subsample': 0.8,
        }
        train_and_evaluate(train, test, params, model_name=model_name,)