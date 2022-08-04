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

    from pd.data.preprop import preprocess_data
    preprocess_data(data_type="train", time_dim=None, all_data=True, fillna="mean_q5_q95", borders=("q5", "q95"))
    train_data = np.load(OUTDIR+"train_logistic_raw_all_mean_q5_q95_q5_q95_data.npy")
    train_labels = np.load(OUTDIR+"train_logistic_raw_all_mean_q5_q95_q5_q95_labels.npy.npy")


    X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=1/9, random_state=0, shuffle=True)
    validation_data = (X_test, y_test)

    lgb_train = lgb.Dataset(X_train.reshape(X_train.shape[0], -1), y_train)
    lgb_valid = lgb.Dataset(X_test.reshape(X_test.shape[0], -1), y_test,)
    
    model = lgb.train(
            params = params,
            train_set = lgb_train,
            num_boost_round = 10000,
            valid_sets = [lgb_train, lgb_valid],
            early_stopping_rounds = 100,
            verbose_eval = 100,
            feval = lgb_amex_metric
            )
    
    model = joblib.load(OUTDIR+f'train_logistic_raw_all_mean_q5_q95_q5_q95_data.pkl')
        
