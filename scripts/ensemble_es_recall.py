import os
import click
import functools
import numpy as np
import pandas as pd 
import torch
import torch.nn as nn
import torch.nn.functional as F
from bes.nn.es_module import ESModule
import ray
import tempfile
import json
from sklearn.model_selection import train_test_split

from bes.es import CMAES, BES
from pd.nn.model import Conv
from pd.params import *
#from pd.data.loader import load_npy_data as load_data
from pd.metric import amex_metric
from pd.pred import pred_test_npy
from pd.nn.recall_models import MLP, MLPAtt
from torch.utils.data import Dataset, DataLoader


class CustomerData(Dataset):
    def __init__(self, data:np.array, test_mode=False, train_labels=None):
        self.data = data
        self.test_mode = test_mode
        self.train_labels = train_labels
        
    def __len__(self):
        return len(self.train_labels)

    def __getitem__(self, index):        
        feat =  torch.from_numpy(self.data[index])

        if self.test_mode:
            return feat, index
        else:
            customer_label = torch.from_numpy(self.train_labels[index])
            return feat, customer_label


@ray.remote
def worker_with_batch(model, weights, feat, labels):
    model.eval()
    model.set_model_params(weights)
    with torch.no_grad():
        pred = model(feat)
        reward, gini, recall = amex_metric(labels, pred.detach().numpy(), return_components=True)

        #return reward
        return recall


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


def get_intial_model_params(model_cls, model_name):
    model = model_cls()
    model_param = torch.load(OUTDIR+model_name)
    model.load_state_dict(model_param)
    
    return model

def run_with_ray_send_data_to_worker(model_cls, population_size, num_cma_iterations, tempdir, model_kwargs={}, init_model_name=None ):
    train_method = "bcma"
    model = model_cls()
    init_params = None


    model = MLPAtt(128)
    if init_model_name is not None:
        conv_model = get_intial_model_params(model_cls, model_name=init_model_name)
        init_params = model.get_model_flat_params()
    if train_method == "cma":
        es = CMAES(model.num_params, popsize=population_size, sigma_init=5)
    else:
        es = BES(model, popsize=population_size, init_params=init_params, max_block_width=1000, sigma_init=5)

    training_history = []
    train_data = np.load(OUTDIR+"train_raw_all_data.npy")
    train_labels = np.load(OUTDIR+"train_raw_all_labels.npy")
    
    X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=1/9, random_state=0, shuffle=True)

    train_dataset = CustomerData(X_train, train_labels=y_train)
    train_loader = DataLoader(train_dataset, batch_size=15000)

    validation_data = (X_test, y_test)
    
    train_dataset = CustomerData(train_data, train_labels=train_labels)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

    for cma_iteration in range(1, num_cma_iterations+1):
        for feat, labels in train_loader:
            preds, h = conv_model(feat, return_featues=True)
            candidates = es.ask()
            rewards = get_candidate_rewards_batch_data(candidates, model, h, labels)
            training_history.append(np.array(rewards))
            h = f"episode: {int(cma_iteration)}, best reward {np.max(rewards)} median {np.median(rewards)} mean {np.mean(rewards)}," \
                f" std {np.std(rewards) },\n"
            with open(os.path.join(tempdir, "progress.txt"), "a") as fh:
                fh.write(h)
            if cma_iteration % logBestIndiv == 0:
                best_idx = np.argmax(rewards)
                best_weights = candidates[best_idx]
                np.save(os.path.join(tempdir, f"best_iteration_{cma_iteration:04d}.npy"), best_weights)
                best_reward = rewards[best_idx]
                if best_reward > PerfThreshold:
                    model.set_model_params(best_weights)
                    val_features = torch.from_numpy(X_test)
                    preds, val_features = conv_model(val_features, return_featues=True)
                    val_pred = model(val_features)
                    val_metrix = amex_metric(y_test, val_pred.detach().numpy())
                    if val_metrix > PerfThreshold:
                        output_name = f"{init_model_name}_{int(1000*best_reward)}"
                        pred_test_npy(model, model_name=output_name)

            es.tell(rewards)


@click.command()
@click.option("--population-size", default=64, type=int)
@click.option("--num-cma-iterations", default=400, type=int)
@click.option("--n-workers", default=32)
@click.option("--init_params", default="conv13_32_all")
def run_experiment(population_size, num_cma_iterations, n_workers, init_params):

    run_info = dict(
        model_class_name=str(Conv),
        population_size=population_size,
        num_cma_iterations=num_cma_iterations,
        init_model_name=init_params
    )

    tempdir = tempfile.mkdtemp(prefix="pd_recall_att_", dir=OUTDIR)
    with open(os.path.join(tempdir, "run_info.json"), "w") as fh:
        json.dump(run_info, fh, indent=4)

    ray.init(num_cpus=n_workers, ignore_reinit_error=True)
    run_with_ray_send_data_to_worker(
        Conv, 
        population_size,
        num_cma_iterations,
        tempdir,
        init_model_name=init_params
    )
    ray.shutdown()


if __name__ == "__main__":
    run_experiment()