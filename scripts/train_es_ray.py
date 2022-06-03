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
from pd.params import DATADIR, CATCOLS
from pd.data.scaler import get_scaler


def rollout(model):
    train_data = pd.read_parquet(DATADIR+"train_data.parquet")
    train_labels = pd.read_csv(DATADIR+"train_labels.csv")
    cont_cols = [col for col in train_data.columns.to_list() if col not in CATCOLS + ["customer_ID", "S_2", "target"]]

    # data prep 
    #train_data[["D_63", "D_64"]] = train_data[["D_63", "D_64"]].astype("category").apply(lambda x: x.cat.codes)
    #cat_cols = ["D_63", "D_64"]
    #train_data = train_data.dropna(how="any", axis=1) 
    ## transform the cont cols 
    scaler = get_scaler(train_data[cont_cols].values)
    train_data[cont_cols] = scaler.transform(train_data[cont_cols].values)
    

    return train_loader
    

@ray.remote
def worker_fn(model_factory, weights):
    model = model_factory()
    model.eval()
    model.set_model_params(weights)
    with torch.no_grad():
        return rollout(model)


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


def run_with_ray(model_cls, model_inpuit_dim, population_size, num_cma_iterations, tempdir, model_kwargs={} ):
    
    model_factory = functools.partial(model_cls, model_inpuit_dim, **model_kwargs)
    es = BES(model_factory(), popsize=population_size)
    
    training_history = []
    for cma_iteration in range(1, num_cma_iterations+1):
        candidates = es.ask()
        ind_rewards = get_candidate_rewards(candidates, model_factory)

        rewards = [np.mean([sum(r) for r in ind_r]) for ind_r in ind_rewards]
        num_steps = np.mean([np.mean([len(r) for r in ind_r]) for ind_r in ind_rewards])
        training_history.append(np.array(rewards))
        h = f"episode: {int(cma_iteration)}, best reward {np.max(rewards)} median {np.median(rewards)} mean {np.mean(rewards)}," \
            f" std {np.std(rewards) }, num_steps {num_steps}, avg_reward {np.max(rewards) / num_steps}\n"
        with open(os.path.join(tempdir, "progress.txt"), "a") as fh:
            fh.write(h)
        if cma_iteration % 100 == 0:
            best_idx = np.argmax(rewards)
            best_weights = candidates[best_idx]
            np.save(os.path.join(tempdir, f"best_iteration_{cma_iteration:04d}.npy"), best_weights)
        es.tell(rewards)


@click.command()
@click.option("--population-size", default=50, type=int)
@click.option("--num-cma-iterations", default=100, type=int)
@click.option("--n-workers", default=32)
def run_experiment(env_name, population_size, num_cma_iterations, n_workers, step_scaling, min_steps, max_steps,
                   scaling_exponent):

    model_kwargs = dict()

    run_info = dict(
        model_class_name=str(MLP),
        environment_name=env_name,
        population_size=population_size,
        num_cma_iterations=num_cma_iterations,
        model_kwargs=model_kwargs,
        step_scaling=step_scaling,
        min_steps=min_steps,
        max_steps=max_steps,
        scaling_exponent=scaling_exponent,
    )

    tempdir = tempfile.mkdtemp(prefix=env_name+"_", dir=".")
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