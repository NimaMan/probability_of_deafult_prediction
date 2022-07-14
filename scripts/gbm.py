import os
import click
import functools
import numpy as np
import pandas as pd 
import torch
import ray
import tempfile
import json
import gc
import random
import joblib
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

from pd.params import *
#from pd.metric import amex_metric


def amex_metric(y_true, y_pred):
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
    return 0.5 * (gini[1]/gini[0] + top_four)


def lgb_amex_metric(y_pred, y_true):
    y_true = y_true.get_label()
    return 'amex_metric', amex_metric(y_true, y_pred), True


def train_lgbm_single_feature(data, f, feature, params, tempdir=None, n_folds=5, seed=42):
    cat_feature = "auto"
    if f in CATCOLS:
        cat_feature = feature
    oof_predictions = np.zeros(len(data))
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (trn_ind, val_ind) in enumerate(kfold.split(data, data["target"])):
        print(' ')
        print('-'*50)
        print(f'Training fold {fold} of {feature} feature...')
        x_train, x_val = data[feature].iloc[trn_ind], data[feature].iloc[val_ind]
        y_train, y_val = data["target"].iloc[trn_ind], data["target"].iloc[val_ind]
        lgb_train = lgb.Dataset(x_train, y_train, categorical_feature=cat_feature)
        lgb_valid = lgb.Dataset(x_val, y_val, categorical_feature=cat_feature)
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
        #joblib.dump(model, tempdir+f'Models/lgbm_fold{fold}_seed{seed}.pkl')
        val_pred = model.predict(x_val) # Predict validation
        oof_predictions[val_ind] = val_pred  # Add to out of folds array
        # Compute fold metric
        score = amex_metric(y_val, val_pred)
        print(f'Our fold {fold} CV score is {score}')
        del x_train, x_val, y_train, y_val, lgb_train, lgb_valid
        gc.collect()
    score = amex_metric(data["target"], oof_predictions)  # Compute out of folds metric
    print(f'Our out of folds CV score is {score}')
    # Create a dataframe to store out of folds predictions
    oof_df = pd.DataFrame({'customer_ID': data['customer_ID'], 'target': data["target"], 'prediction': oof_predictions})
    oof_dir = os.path.join(tempdir, f'oof_lgbm_baseline_{n_folds}fold_seed{seed}_{feature}.csv')
    oof_df.to_csv(oof_dir, index=False)

    return score


@ray.remote
def worker_fn(data, f, feature, params):
    return train_lgbm_single_feature(data, f, feature, params)


def get_features_scores(data, features, params):
    candidate_rewards_tracker = {}
    candidate_rewards = {}
    remaining_ids = []
    for idx, f in enumerate(features.keys()):
        feat = features[f]
        df = data[feat + ["customer_ID", "target"]]
        indiv_remote_id = worker_fn.remote(df, f, feat, params)
        remaining_ids.append(indiv_remote_id)
        candidate_rewards_tracker[indiv_remote_id] = idx

    while remaining_ids:
        done_ids, remaining_ids = ray.wait(remaining_ids)
        result_id = done_ids[0]

        indiv_id = candidate_rewards_tracker[result_id]
        indiv_reward = ray.get(result_id)
        candidate_rewards[indiv_id] = indiv_reward

    rewards = {features[i]: candidate_rewards[i] for i in range(len(features))}

    return rewards


@click.command()
@click.option("--n-workers", default=32)
def run_experiment(n_workers):

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
        'lambda_l2': 2,
        'min_data_in_leaf': 40
        }

    run_info = params

    tempdir = tempfile.mkdtemp(prefix="pd_lgbm_", dir=OUTDIR)
    with open(os.path.join(tempdir, "run_info.json"), "w") as fh:
        json.dump(run_info, fh, indent=4)

    ray.init(num_cpus=n_workers, ignore_reinit_error=True)
    
    
    train = pd.read_parquet(OUTDIR+"train_fe.parquet")
    cat_features = [f"{cf}_last" for cf in CATCOLS]
    for cat_col in cat_features:
        encoder = LabelEncoder()
        train[cat_col] = encoder.fit_transform(train[cat_col])
    
    # Round last float features to 2 decimal place
    num_cols = list(train.dtypes[(train.dtypes == 'float32') | (train.dtypes == 'float64')].index)
    num_cols = [col for col in num_cols if 'last' in col]
    for col in num_cols:
        train[col + '_round2'] = train[col].round(2)
    
    features = {}
    for col in dataCols:
        if col not in ['customer_ID', "S_2", "target"]:
            features[col] = [c for c in train.columns if col+"_" in c]
            
    scores = get_features_scores(train, features, params)
    with open(os.path.join(tempdir, "scores.json"), "w") as fh:
        json.dump(scores, fh, indent=4)

    ray.shutdown()


if __name__ == "__main__":
    run_experiment()