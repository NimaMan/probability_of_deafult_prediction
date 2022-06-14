import os
import click
import functools
import numpy as np
import pandas as pd 
import torch
import ray
import tempfile
import json

from bes.es import CMAES, BES
from pd.nn.model import Conv
from pd.params import *
#from pd.data.loader import load_npy_data as load_data
from pd.metric import amex_metric
from pd.pred import pred_test_npy

from torch.utils.data import Dataset, DataLoader


class CustomerData(Dataset):
    def __init__(self, data:np.array, test_mode=False, train_labels=None):
        self.data = data
        self.test_mode = test_mode
        self.train_labels = train_labels
        
    def __len__(self):
        return len(self.train_labels)

    def __getitem__(self, index):        
        feat =  torch.as_tensor(self.data[index], dtype=torch.float32)

        if self.test_mode:
            return feat, index
        else:
            customer_label = torch.as_tensor(self.train_labels[index], dtype=torch.float32)
            return feat, customer_label



def rollout_with_batch(model, feat):
    feat, labels  = load_data()
    pred = model(feat)
    reward = amex_metric(labels, pred.detach().numpy())
    
    return reward


@ray.remote
def worker_with_batch(model, weights, feat, labels):
    model.eval()
    model.set_model_params(weights)
    with torch.no_grad():
        pred = model(feat)
        reward = amex_metric(labels, pred.detach().numpy())

        return reward


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


def run_with_ray_send_data_to_worker(model_cls, population_size, num_cma_iterations, tempdir, model_kwargs={} ):
    
    model = model_cls()
    es = BES(model, popsize=population_size)
    training_history = []
    train_data = np.load(OUTDIR+"train_data_all.npy")
    train_labels = np.load(OUTDIR+"train_labels_all.npy")
    train_dataset = CustomerData(train_data, train_labels=train_labels)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

    for cma_iteration in range(1, num_cma_iterations+1):
        for feat, labels in train_loader:
        
            candidates = es.ask()
            rewards = get_candidate_rewards_batch_data(candidates, model, feat, labels)
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
                    pred_test_npy(model, reward=best_reward)

            es.tell(rewards)


@click.command()
@click.option("--population-size", default=5, type=int)
@click.option("--num-cma-iterations", default=400, type=int)
@click.option("--n-workers", default=5)
def run_experiment(population_size, num_cma_iterations, n_workers):

    run_info = dict(
        model_class_name=str(Conv),
        population_size=population_size,
        num_cma_iterations=num_cma_iterations,
    )

    tempdir = tempfile.mkdtemp(prefix="pd_", dir=OUTDIR)
    with open(os.path.join(tempdir, "run_info.json"), "w") as fh:
        json.dump(run_info, fh, indent=4)

    ray.init(num_cpus=n_workers, ignore_reinit_error=True)
    run_with_ray_send_data_to_worker(
        Conv, 
        population_size,
        num_cma_iterations,
        tempdir,
    )
    ray.shutdown()


if __name__ == "__main__":
    run_experiment()