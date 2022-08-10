import os
import gc
import tempfile
import click
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

import pd.metric as metric
from pd.utils import merge_with_pred, get_customers_data_indices
from pd.params import *
from pd.gmb_utils import lgb_amex_metric, get_agg_data


def train_lgbm(train_data, train_labels, params, exp_name):

    train_indices = np.arange(train_labels.shape[0])
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(train_data, train_labels, train_indices, test_size=1/9, random_state=0, shuffle=True)
    validation_data = (X_test, y_test)
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid = lgb.Dataset(X_test, y_test,)
    
    print(f"Start training LGB {exp_name} with number of feature {X_train.shape[1]}", params)
    model = lgb.train(
            params = params,
            train_set = lgb_train,
            num_boost_round = 10500,
            valid_sets = [lgb_train, lgb_valid],
            early_stopping_rounds = 100,
            verbose_eval = 100,
            feval = lgb_amex_metric
            )
    
    joblib.dump(model, filename=MODELDIR+exp_name)
    y_pred = model.predict(X_test)
    merge_with_pred(y_pred, indices_test, model_name="lgbm13")
    del train_data, X_test, X_train, validation_data, lgb_train, lgb_valid
    gc.collect()

    return model 


def train_lgbm_cv(data, labels, indices, params, model_name, tempdir=None, n_folds=5, seed=42):
    """
    take the data in certain indices [for example c13 data]
    """
    used_indices, other_indices = indices
    print(f"training the {model_name}", params)
    oof_predictions = np.zeros(len(used_indices))
    best_model_name, best_model_score = "", 0
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (trn_ind, val_ind) in enumerate(kfold.split(data.iloc[used_indices], labels[used_indices], used_indices)):
        print('-'*50)
        print(f'Training fold {fold} ...')
        x_train, x_val = data.iloc[used_indices].iloc[trn_ind], data.iloc[used_indices].iloc[val_ind]
        y_train, y_val = labels[used_indices][trn_ind], labels[used_indices][val_ind]
        
        lgb_train = lgb.Dataset(x_train, y_train, )
        lgb_valid = lgb.Dataset(x_val, y_val, )

        model = lgb.train(
            params = params,
            train_set = lgb_train,
            num_boost_round = 10500,
            valid_sets = [lgb_train, lgb_valid],
            early_stopping_rounds = 100,
            verbose_eval = 100,
            feval = lgb_amex_metric
            )
        val_pred = model.predict(x_val) # Predict validation
        oof_predictions[val_ind] = val_pred  # Add to out of folds array
        
        # Compute fold metric
        score, gini, recall = metric.amex_metric(y_val.reshape(-1, ), val_pred, return_components=True)
        joblib.dump(model, os.path.join(MODELDIR, f'{model_name}_{int(score*10000)}'))
        if score > best_model_score:
            best_model_name = f'{model_name}_{int(score*10000)}'
            best_model_score = score

        pred_indices = used_indices[val_ind]
        merge_with_pred(val_pred, pred_indices, model_name=model_name)
    
        print(f'Our fold {fold} CV score is {score}')
        del x_train, x_val, y_train, y_val, lgb_train, lgb_valid
        gc.collect()
    score, gini, recall = metric.amex_metric(labels[used_indices].reshape(-1, ), oof_predictions, return_components=True)  # Compute out of folds metric
    print(f'Our out of folds CV score is {score}, Gini {gini}, recall {recall}')
    
    best_model = joblib.load(os.path.join(MODELDIR, best_model_name))
    if len(other_indices) > 0:
        other_pred = best_model.predict(data.iloc[other_indices])
        merge_with_pred(other_pred, other_indices, model_name=model_name)
    
    return best_model


def test_lgbm(model, model_name, test_data_name=f"test_agg1_mean_q5_q95_q5_q95"):

    test_data_dir = f"{test_data_name}.npz"
    test_data, labels, cat_indices = get_agg_data(data_dir=test_data_dir)
    test_pred = model.predict(test_data) # Predict the test set
    del test_data
    gc.collect()

    with open(OUTDIR+f'{test_data_name}_id.json', 'r') as f:
            test_id_dict = json.load(f)

    result = pd.DataFrame({"customer_ID":test_id_dict.values(), 
                        "prediction":test_pred.reshape(-1)
                        }
                        )

    sub_file_dir = os.path.join(OUTDIR, f"{model_name}.csv")
    result.set_index("customer_ID").to_csv(sub_file_dir)
    
    merge_with_pred(test_pred, np.arange(len(test_pred)),
                    model_name=model_name, type="test", id_dir=f'{test_data_name}_id.json')
    

@click.command()
@click.option("--agg", default=1)
@click.option("--n_workers", default=127)
def run_experiment(agg, n_workers):
    exp_name = f"train_agg{agg}_mean_q5_q95_q5_q95_data"
    params = {
        'objective': 'binary',
        'metric': "binary_logloss",
        'boosting': 'dart',
        'seed': 42,
        'num_leaves': 50,
        'learning_rate': 0.01,
        'feature_fraction': 0.20,
        'bagging_freq': 10,
        'bagging_fraction': 0.50,
        'n_jobs': -1,
        'lambda_l2': 4,
        'lambda_l1': 4,
        'min_data_in_leaf': 40, 
        'max_bin': 255,  # Deafult is 255

        }
    
    run_info = params
    tempdir = tempfile.mkdtemp(prefix=f"pd_lgbm_{exp_name}_", dir=OUTDIR)
    with open(os.path.join(tempdir, "run_info.json"), "w") as fh:
        json.dump(run_info, fh, indent=4)   

    train_data, train_labels, cat_indices = get_agg_data(data_dir=f"train_agg{agg}_mean_q5_q95_q5_q95.npz")
    
    model_name = f"lgbm13_agg{agg}"
    indices = get_customers_data_indices(num_data_points=[13], id_dir=f'train_agg{agg}_mean_q5_q95_q5_q95_id.json')
    model = train_lgbm_cv(train_data, train_labels, indices, params, model_name=model_name, tempdir=tempdir, n_folds=5, seed=42)

    model_name = f"lgbm_agg{agg}"
    indices = get_customers_data_indices(num_data_points=np.arange(14), id_dir=f'train_agg{agg}_mean_q5_q95_q5_q95_id.json')
    model = train_lgbm_cv(train_data, train_labels, indices, params, model_name=model_name, tempdir=tempdir, n_folds=5, seed=42)

    test_lgbm(model, model_name, test_data_name=f"test_agg{agg}_mean_q5_q95_q5_q95")


if __name__ == "__main__":
    run_experiment()