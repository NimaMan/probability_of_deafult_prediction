import os
import click
import functools
import numpy as np
import pandas as pd 
import ray
import tempfile
import json
import gc
import random
import joblib
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
    
import lightgbm as lgb

from pd.gmb_utils import lgb_amex_metric, train_lgbm
import pd.metric as metric
from pd.params import *
#from pd.metric import amex_metric


@ray.remote
def worker_fn(data, labels, params, feature, tempdir):
    return train_lgbm(data, labels, params, feature, tempdir)



def gbm_hyper_parameter_opt(data, labels, features, params, tempdir):
    candidate_rewards_tracker = {}
    candidate_rewards = {}
    remaining_ids = []
    for idx, f in enumerate(features):
        d = data[:, :, idx]
        indiv_remote_id = worker_fn.remote(d, labels, params, f, tempdir)
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
        'lambda_l2': 4,
        'lambda_l1': 4,
        'min_data_in_leaf': 40
        }

    run_info = params

    tempdir = tempfile.mkdtemp(prefix="pd_lgbm_", dir=OUTDIR)
    with open(os.path.join(tempdir, "run_info.json"), "w") as fh:
        json.dump(run_info, fh, indent=4)

    ray.init(num_cpus=n_workers, ignore_reinit_error=True)
    
    train = np.load(OUTDIR+"train_raw_all_data.npy")
    labels = np.load(OUTDIR+"train_raw_all_labels.npy")            
    
    with open(os.path.join(tempdir, "scores.json"), "w") as fh:
        json.dump(scores, fh, indent=4)

    ray.shutdown()


if __name__ == "__main__":
    run_experiment()