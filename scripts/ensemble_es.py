
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
from pd.nn.model import MLP
from pd.params import *
from pd.metric import amex_metric


    
    

@ray.remote
def worker_fn(model_factory, weights):
    pass 


def get_candidate_rewards(candidates, model_factory):
    candidate_rewards_tracker = {}
    candidate_rewards = {}
    remaining_ids = []
    for idx, weights in enumerate(candidates):
        
        indiv_remote_id = worker_fn.remote(model_factory, weights)
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


def run_with_ray(model_cls, population_size, num_cma_iterations, tempdir, model_kwargs={} ):
    
    model_factory = functools.partial(model_cls, **model_kwargs)
    es = BES(model_factory(), popsize=population_size)
    training_history = []
    for cma_iteration in range(1, num_cma_iterations+1):
        candidates = es.ask()
        rewards = get_candidate_rewards(candidates, model_factory)
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
                eval(model_factory, best_weights, reward=best_reward)

        es.tell(rewards)


@click.command()
@click.option("--population-size", default=50, type=int)
@click.option("--num-cma-iterations", default=400, type=int)
@click.option("--n-workers", default=32)
def run_experiment(population_size, num_cma_iterations, n_workers):

    model_kwargs = dict(input_dim=177, )

    run_info = dict(
        model_class_name=str(MLP),
        population_size=population_size,
        num_cma_iterations=num_cma_iterations,
        model_kwargs=model_kwargs,
    )

    tempdir = tempfile.mkdtemp(prefix="pd_", dir=OUTDIR)
    with open(os.path.join(tempdir, "run_info.json"), "w") as fh:
        json.dump(run_info, fh, indent=4)

    ray.init(num_cpus=n_workers, ignore_reinit_error=True)
    run_with_ray(
        MLP,
        population_size,
        num_cma_iterations,
        tempdir,
        model_kwargs=model_kwargs,
    )
    ray.shutdown()


if __name__ == "__main__":
    run_experiment()