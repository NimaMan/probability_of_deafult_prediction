#%%
import os
import click
import functools
import torch
import numpy as np
import pandas as pd 
import torch.nn as nn
import torch.nn.functional as F
from bes.nn.es_module import ESModule
import ray
import tempfile
import json
from sklearn.model_selection import train_test_split

from bes.es import CMAES, BES
from pd.nn.mlp import Linear
from pd.params import *
#from pd.data.loader import load_npy_data as load_data
from pd.metric import amex_metric
from pd.nn.recall_models import MLP, MLPAtt
from torch.utils.data import Dataset, DataLoader


@ray.remote
def worker_with_batch(model, weights, feat, labels):
    model.eval()
    model.set_model_params(weights)
    with torch.no_grad():
        pred = model(feat)
        reward, gini, recall = amex_metric(labels, pred.detach().numpy(), return_components=True)

        return reward
        #return recall


def get_candidate_rewards_batch_data(candidates, model, feat, labels):
    candidate_rewards_tracker = {}
    candidate_rewards = {}
    remaining_ids = []
    for idx, weights in enumerate(candidates):
        
        indiv_remote_id = worker_with_batch.remote(model, weights, feat=feat, labels=labels)
        remaining_ids.append(indiv_remote_id)
        candidate_rewards_tracker[indiv_remote_id] = idx

    while remaining_ids:
        done_ids, remaining_ids = ray.wait(remaining_ids)
        result_id = done_ids[0]

        indiv_id = candidate_rewards_tracker[result_id]
        indiv_reward = ray.get(result_id)
        candidate_rewards[indiv_id] = indiv_reward

    rewards = [candidate_rewards[i] for i in range(candidates.shape[0])]

    return rewards


def add_time_feature_train(data):
    train_customers_count = pd.read_parquet(TRAINDATA).customer_ID.value_counts()
    data["T"] = 0
    data["T"].loc[train_customers_count.index] = train_customers_count.values.tolist()

    data["T"] = data["T"]/data["T"].max()

    return data


def add_time_feature_test(data):
    customers_count = pd.read_parquet(TESTDATA).customer_ID.value_counts()
    data["T"] = 0
    data["T"].loc[customers_count.index] = customers_count.values.tolist()
    data["T"] = data["T"]/data["T"].max()

    return data


def run_with_ray_send_data_to_worker(population_size, num_cma_iterations, tempdir, model_kwargs={}, model_name=None):
    train_method = "cma"
    
    training_history = []
    train_data = pd.read_csv(PREDDIR+"train_pred.csv", index_col=0)
    train_labels = train_data["target"].values
    cols = ["catb_agg1", "catb_agg2", "K7977_xgb_52", "K7977_xgb_62", "K7977_xgb_82","K7977_catb_52" ,"K7977_focal_52" ,
        "K7977_catb_62", "K7977_focal_62", "K7977_catb_82", "K7977_focal_82", "K7977_focal_xgb_42", "K7977_focal_xgb_52", "K7977_focal_xgb_62"]

    train_data[cols] = train_data[cols].rank(pct=True, axis=0)
    train_data["STD"] = train_data[cols].apply(np.std, axis=1)
    train_data["STD"] = train_data["STD"]/train_data["STD"].max()
    cols.append("STD")

    #train_data = add_time_feature_train(train_data)
    #cols.append("T")

    d = pd.read_parquet(TRAINDATA).groupby("customer_ID")[bestCols].agg(['mean', 'std', 'min', 'max', "first", 'last'])
    df = df.merge(d, how="left", on="customer_ID")
    cols = cols + d.columns.to_list()

    X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=1/9, random_state=0, shuffle=True)    
    input_dim = len(cols)
    if model_name == "Linear":
        model = Linear(input_dim)
    elif model_name == "Att":
        model = MLPAtt(input_dim)
    else:
        model = MLP(input_dim)

    if train_method == "cma":
        es = CMAES(model.num_params, popsize=population_size, sigma_init=5)
    else:
        es = BES(model, popsize=population_size, init_params=None, max_block_width=2000, sigma_init=5)

    for cma_iteration in range(1, num_cma_iterations+1):
        
        batch_data = X_train.sample(BATCH_SIZE_EN, weights="STD")
        labels = batch_data["target"].values
        batch_feat = torch.from_numpy(batch_data[cols].values).float()
        candidates = es.ask()
        rewards = get_candidate_rewards_batch_data(candidates, model, batch_feat, labels)
        training_history.append(np.array(rewards))
        h = f"episode: {int(cma_iteration)}, best reward {np.max(rewards)} median {np.median(rewards)} mean {np.mean(rewards)}," \
            f" std {np.std(rewards) },\n"
        with open(os.path.join(tempdir, "progress.txt"), "a") as fh:
            fh.write(h)
        best_idx = np.argmax(rewards)
        best_weights = candidates[best_idx]
        best_reward = rewards[best_idx]
        if best_reward > PerfThreshold:
                model.set_model_params(best_weights)
                val_features = torch.from_numpy(X_test[cols].values).float()
                val_pred = model(val_features)
                val_metrix = amex_metric(y_test, val_pred.detach().numpy(), return_components=True)
                print("The val ", val_metrix)
                np.save(os.path.join(tempdir, f"i{cma_iteration:04d}_{val_metrix[0]:.4f}.npy"), best_weights)
            
        es.tell(rewards)


#"""
@click.command()
@click.option("--population-size", default=254, type=int)
@click.option("--num_cma_iters", default=5000, type=int)
@click.option("--n_workers", default=127)
@click.option("--model_name", default="Att")
def run_experiment(population_size, num_cma_iters, n_workers, model_name):

    run_info = dict(
        population_size=population_size,
        num_cma_iterations=num_cma_iters,
        init_model_name=model_name
    )

    tempdir = tempfile.mkdtemp(prefix="pd_pred_rank_bestcols_", dir=OUTDIR)
    with open(os.path.join(tempdir, "run_info.json"), "w") as fh:
        json.dump(run_info, fh, indent=4)

    ray.init(num_cpus=n_workers, ignore_reinit_error=True)
    run_with_ray_send_data_to_worker(
        population_size,
        num_cma_iters,
        tempdir,
        model_name=model_name
    )
    ray.shutdown()
#"""


if __name__ == "__main__":
    run_experiment()
    """
    population_size = 64
    num_cma_iterations = 400

    run_info = dict(
        population_size=population_size,
        num_cma_iterations=num_cma_iterations,    )

    tempdir = tempfile.mkdtemp(prefix="pd_recall_", dir=OUTDIR)
    with open(os.path.join(tempdir, "run_info.json"), "w") as fh:
        json.dump(run_info, fh, indent=4)

    ray.init(num_cpus=3, ignore_reinit_error=True)
    run_with_ray_send_data_to_worker(
        population_size,
        num_cma_iterations,
        tempdir,
    )
    ray.shutdown()
    """
# %%
