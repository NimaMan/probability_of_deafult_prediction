import os
import gc
import tempfile
import json
import warnings
warnings.filterwarnings('ignore')
import scipy as sp
import numpy as np
import pandas as pd

import random
import joblib
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

from pd.params import *
from pd.gmb_utils import lgb_amex_metric


def get_agg_data(data_dir="train_agg_mean_q5_q95_q5_q95.npz"):
    d = np.load(OUTDIR+data_dir)
    #train_data = np.concatenate((d["d2"].astype(np.int32), d["d1"].reshape(d["d1"].shape[0], -1)), axis=1)
    train_labels = d["labels"]
    df2 = pd.DataFrame(d["d2"].astype(np.int32))
    df = pd.DataFrame(d["d1"].reshape(d["d1"].shape[0], -1))
    df = pd.concat((df2, df), axis=1,)
    df.columns = [f"c{i}" for i in range(df.shape[1])]
    cat_indices = list(np.arange(d["d2"].shape[1]))

    return df, train_labels, cat_indices


if __name__ == "__main__":
    params = {
        'objective': 'binary',
        'metric': "binary_logloss",
        'boosting': 'dart',
        'seed': 42,
        'num_leaves': 100,
        'learning_rate': 0.01,
        'feature_fraction': 0.20,
        'bagging_freq': 10,
        'bagging_fraction': 0.50,
        'n_jobs': -1,
        'lambda_l2': 4,
        'min_data_in_leaf': 40
        }
    
    run_info = params

    tempdir = tempfile.mkdtemp(prefix="pd_all_lgbm_", dir=OUTDIR)
    with open(os.path.join(tempdir, "run_info.json"), "w") as fh:
        json.dump(run_info, fh, indent=4)

    train_data, train_labels, cat_indices = get_agg_data(data_dir="train_agg_mean_q5_q95_q5_q95.npz")
    X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=1/9, random_state=0, shuffle=True)
    validation_data = (X_test, y_test)

    lgb_train = lgb.Dataset(X_train.reshape(X_train.shape[0], -1), y_train)
    lgb_valid = lgb.Dataset(X_test.reshape(X_test.shape[0], -1), y_test,)
    
    model = lgb.train(
            params = params,
            train_set = lgb_train,
            num_boost_round = 10500,
            valid_sets = [lgb_train, lgb_valid],
            early_stopping_rounds = 100,
            verbose_eval = 100,
            feval = lgb_amex_metric
            )
    
    joblib.dump(MODELDIR+f'train_logistic_raw_all_mean_q5_q95_q5_q95_data.pkl')
        
