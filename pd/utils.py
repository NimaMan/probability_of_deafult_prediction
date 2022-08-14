#%%
import os 
import pickle 
import gzip 
import torch
import numpy as np
import pandas as pd 
from pd.params import *


def write_log(log, log_desc="log", out_dir=None):
    log_file_name = f"{log_desc}.txt"
    os.makedirs(out_dir, exist_ok=True)
    log_file_dir = os.path.join(out_dir, log_file_name)
    with open(log_file_dir, "a") as f:
        f.write(log)
        f.write("\n")


def wirte_data_pickle(data, name, outdir=None):

    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
    file_dir = os.path.join(outdir, name)
    with gzip.open(file_dir, "wb") as f:
        pickle.dump(data, f)


def load_pickle_data(path):
    with gzip.open(path, "rb") as f:
        data = pickle.load(f)

    return data


def get_torch_agg_data(data_dir="train_agg_mean_q5_q95_q5_q95.npz"):
    d = np.load(OUTDIR+data_dir)
    #train_data = np.concatenate((d["d2"].astype(np.int32), d["d1"].reshape(d["d1"].shape[0], -1)), axis=1)
    train_labels = d["labels"]
    cat_vars = d["d2"].astype(np.float32)
    cont_vars = d["d1"]
    
    return cont_vars, cat_vars, train_labels


def get_customers_data_indices(id_dir, num_data_points=[13]):
    import json 
    with open(OUTDIR+id_dir, 'r') as f:
            train_id_dict = json.load(f)
    train_id_dict = {val:key for key, val in train_id_dict.items()}
    train_customers_count = pd.read_parquet(TRAINDATA).customer_ID.value_counts().to_dict()
    wanted_indices = []
    other_indices = []
    for c, i in train_id_dict.items():
        if train_customers_count[c] in num_data_points:
            wanted_indices.append(int(i))
        else:
            other_indices.append(int(i))

    return np.array(wanted_indices), np.array(other_indices)


def get_customers_id_from_indices(indices, id_dir):
    import json 
    with open(OUTDIR+id_dir, 'r') as f:
            id_dict = json.load(f)
    customers = [id_dict[str(idx)] for idx in indices]
    
    return customers


def merge_with_pred(y_pred, y_indices, model_name, id_dir, type="train"):
    if type == "train":
        pred_dir = os.path.join(PREDDIR, "train_pred.csv")
    else:
        pred_dir = os.path.join(PREDDIR, "test_pred.csv")

    pred_file = pd.read_csv(pred_dir, index_col=0)
    customers = get_customers_id_from_indices(y_indices, id_dir=id_dir)
    
    if model_name in pred_file.columns:
        pred_file[model_name].loc[customers] = y_pred.reshape(-1)
        pred_file.to_csv(pred_dir)
    else:
        result = pd.DataFrame({"customer_ID": customers, 
                        model_name: y_pred.reshape(-1)
                        })
        pred_file = pred_file.merge(result, how='left', on='customer_ID')
        pred_file.set_index("customer_ID").to_csv(pred_dir)


def get_pred_data(id_dir, type="train", agg=1):
    if type == "train":
        pred_dir = os.path.join(PREDDIR, "train_pred.csv")
    else:
        pred_dir = os.path.join(PREDDIR, "test_pred.csv")

    pred_file = pd.read_csv(pred_dir, index_col=0)
    indices = np.arange(len(pred_file))
    customers = get_customers_id_from_indices(indices, id_dir=id_dir)
    
    cols  = [col for col in pred_file.columns if col not in ["customer_ID", "target"]]
    data = pred_file.loc[customers][cols].values.astype(np.float32)

    return data
