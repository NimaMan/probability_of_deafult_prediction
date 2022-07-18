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
from sklearn.preprocessing import LabelEncoder

import torch 
import torch.nn 
from torch.utils.data import DataLoader

from pd.nn.model import Conv
from pd.data.loader import CustomerData
from pd.nn.train_utils import train_torch_model
from pd.metric import amex_metric
from pd.params import *
from pd.pred import pred_test_npy as predict


def train_torch(data, labels, feature, params, tempdir=None, n_folds=5, seed=42):
    
    oof_predictions = np.zeros(len(data))
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (trn_ind, val_ind) in enumerate(kfold.split(data, labels)):
        print(' ')
        print('-'*50)
        print(f'Training fold {fold} of {feature} feature...')
        x_train, x_val = data[trn_ind], data[val_ind]
        y_train, y_val = labels[trn_ind], labels[val_ind]
        validation_data = (x_val, y_val)

        train_dataset = CustomerData(x_train, train_labels=y_train)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
        model = Conv(input_dim=x_train.shape[-1], conv_channels=25)
        model = train_torch_model(model, train_loader, num_epochs=100, validation_data=validation_data, output_model_name=f)

        # Save best model
        #joblib.dump(model, tempdir+f'Models/lgbm_fold{fold}_seed{seed}.pkl')
        val_pred = model(x_val) # Predict validation
        oof_predictions[val_ind] = val_pred  # Add to out of folds array
        # Compute fold metric
        score = amex_metric(y_val, val_pred)
        print(f'Our fold {fold} CV score is {score}')
        del x_train, x_val, y_train, y_val, lgb_train, lgb_valid
        gc.collect()
    score = amex_metric(labels, oof_predictions)  # Compute out of folds metric
    print(f'Our out of folds CV score is {score}')
    # Create a dataframe to store out of folds predictions
    oof_dir = os.path.join(tempdir, f'{n_folds}fold_seed{seed}_{feature}.npy')
    np.save(oof_dir, oof_predictions)
    
    return score


@ray.remote
def worker_fn(data, labels, feature, params):
    return train_torch(data, labels, feature, params)


def get_features_scores(data, labels, features, params):
    candidate_rewards_tracker = {}
    candidate_rewards = {}
    remaining_ids = []
    for idx, f in enumerate(features):
        d = data[:, :, idx].reshape(data.shape[0], 13, -1)
        indiv_remote_id = worker_fn.remote(d, labels, f, params)
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
        }

    run_info = params

    tempdir = tempfile.mkdtemp(prefix="pd13_conv_", dir=OUTDIR)
    with open(os.path.join(tempdir, "run_info.json"), "w") as fh:
        json.dump(run_info, fh, indent=4)

    ray.init(num_cpus=n_workers, ignore_reinit_error=True)
    
    train = np.load(OUTDIR+"c13_data.npy")
    labels = np.load(OUTDIR+"c13_labels.npy")            
    scores = get_features_scores(train, labels, featureCols, params)

    with open(os.path.join(tempdir, "scores.json"), "w") as fh:
        json.dump(scores, fh, indent=4)

    ray.shutdown()


if __name__ == "__main__":
    run_experiment()