
import os
import pathlib
import numpy as np
import pandas as pd 
import json
import gc
import random
import joblib
from sklearn.model_selection import StratifiedKFold, train_test_split
import lightgbm as lgb
import pd.metric as metric
from pd.params import *


def amex_metric(y_true, y_pred, return_components=False):
    labels = np.transpose(np.array([y_true, y_pred]))
    labels = labels[labels[:, 1].argsort()[::-1]]
    weights = np.where(labels[:,0]==0, 20, 1)
    cut_vals = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four = np.sum(cut_vals[:,0]) / np.sum(labels[:,0])
    gini = [0,0]
    for i in [1,0]:
        labels = np.transpose(np.array([y_true, y_pred]))
        labels = labels[labels[:, i].argsort()[::-1]]
        weight = np.where(labels[:,0]==0, 20, 1)
        weight_random = np.cumsum(weight / np.sum(weight))
        total_pos = np.sum(labels[:, 0] *  weight)
        cum_pos_found = np.cumsum(labels[:, 0] * weight)
        lorentz = cum_pos_found / total_pos
        gini[i] = np.sum((lorentz - weight_random) * weight)
    if return_components:
        return 0.5 * (gini[1]/gini[0] + top_four), gini[1]/gini[0], top_four

    return 0.5 * (gini[1]/gini[0] + top_four)


def lgb_amex_metric(y_pred, y_true):
    y_true = y_true.get_label()
    score, gini, recall = amex_metric(y_true, y_pred, return_components=True)
    return f'amex_metric gini {gini:.3f} recall {recall:.3f}', score, True


def xgb_amex(y_pred, y_true):
    y_true = y_true.get_label()
    score, gini, recall = amex_metric(y_true, y_pred, return_components=True)
    
    #return f'gini_{int(gini*1e7)}_recall_{int(recall*1e7)}', score
    return f'amex', score

def train_lgbm(data, labels, params, feature=None, tempdir=None, n_folds=5, seed=42):
    
    oof_predictions = np.zeros(len(data))
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (trn_ind, val_ind) in enumerate(kfold.split(data, labels)):
        print(' ')
        print('-'*50)
        print(f'Training fold {fold} of {feature} feature...')
        x_train, x_val = data[trn_ind], data[val_ind]
        y_train, y_val = labels[trn_ind], labels[val_ind]
        lgb_train = lgb.Dataset(x_train.reshape(x_train.shape[0], -1), y_train, categorical_feature="auto")
        lgb_valid = lgb.Dataset(x_val.reshape(x_val.shape[0], -1), y_val, categorical_feature="auto")

        model = lgb.train(
            params = params,
            train_set = lgb_train,
            num_boost_round = 1000,
            valid_sets = [lgb_train, lgb_valid],
            early_stopping_rounds = 100,
            verbose_eval = 100,
            feval = lgb_amex_metric
            )
        # Save best model
        model_dir = os.path.join(OUTDIR, "Models")
        model_name =  f'{n_folds}fold_seed{seed}_{feature}.pkl'
        pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True) 
        joblib.dump(model, os.path.join(model_dir, model_name))
        val_pred = model.predict(x_val) # Predict validation
        oof_predictions[val_ind] = val_pred  # Add to out of folds array
        # Compute fold metric
        score, gini, recall = metric.amex_metric(y_val.reshape(-1, ), val_pred, return_components=True)
        print(f'Our fold {fold} CV score is {score}')
        del x_train, x_val, y_train, y_val, lgb_train, lgb_valid
        gc.collect()
    score, gini, recall = metric.amex_metric(labels.reshape(-1, ), oof_predictions, return_components=True)  # Compute out of folds metric
    print(f'Our out of folds CV score is {score}')
    # Create a dataframe to store out of folds predictions
    oof_dir = os.path.join(tempdir, f'{n_folds}fold_seed{seed}_{feature}.npy')
    np.save(oof_dir, oof_predictions)
    
    return (score, gini, recall)


def get_agg_data(data_dir="train_agg_mean_q5_q95_q5_q95.npz"):
    d = np.load(OUTDIR+data_dir)
    #train_data = np.concatenate((d["d2"].astype(np.int32), d["d1"].reshape(d["d1"].shape[0], -1)), axis=1)
    train_labels = d["labels"]
    df2 = pd.DataFrame(d["d2"].astype(np.int32))
    df = pd.DataFrame(d["d1"].reshape(d["d1"].shape[0], -1))
    df = pd.concat((df2, df), axis=1,)
    df.columns = [f"c{i}" for i in range(df.shape[1])]
    cat_indices = list(np.arange(33))

    return df, train_labels, cat_indices
