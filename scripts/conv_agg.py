import os
import gc
import tempfile
import click
import json
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import Sampler, BatchSampler
from pd.metric import amex_metric
from pd.utils import write_log
from pd.nn.conv import Conv, ConvAgg
from pd.utils import merge_with_pred, get_torch_agg_data, get_customers_data_indices
from pd.params import *


def train_conv_cv(cont_data, cat_data, labels, indices, config, model_name, tempdir=None, n_folds=5, seed=42):
    used_indices, other_indices = indices
    print(f"training the {model_name}", config)
    oof_predictions = np.zeros(len(used_indices))
    best_model_name, best_model_score = "", 0
    
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (trn_ind, val_ind) in enumerate(kfold.split(used_indices, labels[used_indices])):
        print('-'*50)
        print(f'Training fold {fold} ...')
        model = ConvAgg(input_dim=cont_data.shape[-1], in_cat_dim=cat_data.shape[-1], conv_channels=config["conv_channels"])
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
            model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=config["weight_decay"])
        criterion = torch.nn.BCELoss()

        x_cont_train, x_cont_val = cont_data[used_indices][trn_ind], cont_data[used_indices][val_ind]
        x_cat_train, x_cat_val = cat_data[used_indices][trn_ind], cat_data[used_indices][val_ind]
        y_train, y_val = labels[used_indices][trn_ind], labels[used_indices][val_ind]

        batch_sampler_indices = np.arange(len(trn_ind))
        np.random.shuffle(batch_sampler_indices)
        sampler = BatchSampler(batch_sampler_indices, batch_size=BATCH_SIZE, drop_last=False, )
        
        for epoch in range(config["num_epochs"]): 
            for idx, batch in enumerate(sampler):
                cont_feat = torch.as_tensor(x_cont_train[batch], dtype=torch.float32).to(device)
                cat_feat =  torch.as_tensor(x_cat_train[batch], dtype=torch.float32).to(device)
                clabel =  torch.as_tensor(y_train[batch], dtype=torch.float32).to(device)
                pred = model(cont_feat, cat_feat)
                #weight = clabel.clone()
                #weight[weight==0] = 4
                #criterion.weight = weight
                loss = criterion(pred, clabel.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                model_metric, gini, recall = amex_metric(clabel.cpu().detach().numpy(), pred.cpu().detach().numpy(), return_components=True)
                val_metrix = 0
                val_gini, val_recall = 0, 0
                if model_metric > valThreshold:
                    cont_feat = torch.as_tensor(x_cont_val, dtype=torch.float32).to(device)
                    cat_feat = torch.as_tensor(x_cat_val, dtype=torch.float32).to(device)
                    val_pred = model(cont_feat, cat_feat).cpu().detach().numpy()
                    val_metrix, val_gini, val_recall = amex_metric(y_val, val_pred, return_components=True)

                log_message = f"{epoch}, BCE loss: {loss.item():.3f},train -> amex {model_metric:.3f}, gini {gini:.3f}, recall {recall:.3f}, val -> amex {val_metrix:.3f} gini {val_gini:.3f}, recall {val_recall:.3f}"
                print(log_message)
                write_log(log=log_message, log_desc=model_name+"_log", out_dir=tempdir)
        
        cont_feat = torch.as_tensor(x_cont_val, dtype=torch.float32).to(device)
        cat_feat = torch.as_tensor(x_cat_val, dtype=torch.float32).to(device)
        val_pred = model(cont_feat, cat_feat).cpu().detach().numpy().reshape(-1, )
        oof_predictions[val_ind] = val_pred  # Add to out of folds array
        
        del x_cont_train, x_cat_train, y_train
        gc.collect()
        torch.cuda.empty_cache()
        # Compute fold metric
        score, gini, recall = amex_metric(y_val.reshape(-1, ), val_pred, return_components=True)
        torch.save(model.state_dict(), os.path.join(MODELDIR, f'{model_name}_{int(score*10000)}'))                
        if score > best_model_score:
            best_model_name = f'{model_name}_{int(score*10000)}'
            best_model_score = score

        pred_indices = used_indices[val_ind]
        merge_with_pred(val_pred, pred_indices, model_name=model_name, id_dir=config["id_dir"])
    
        print(f'Our fold {fold} CV score is {score}')
        del y_val, x_cont_val, x_cat_val, model
        gc.collect()
        torch.cuda.empty_cache()
    score, gini, recall = amex_metric(labels[used_indices].reshape(-1, ), oof_predictions, return_components=True)  # Compute out of folds metric
    print(f'Our out of folds CV score is {score}, Gini {gini}, recall {recall}')
    
    model_param = torch.load(os.path.join(MODELDIR, best_model_name), map_location=device)
    model.load_state_dict(model_param)
    
    if len(other_indices) > 0:
        cont_feat = torch.as_tensor(cont_data[other_indices], dtype=torch.float32).to(device)
        cat_feat = torch.as_tensor(cat_data[other_indices], dtype=torch.float32).to(device)
        other_pred = model(cont_feat, cat_feat).cpu().detach().numpy()
        merge_with_pred(other_pred, other_indices, model_name=model_name)
    
    return model


def test_conv(model, model_name, test_data_name=f"test_agg1_mean_q5_q95_q5_q95"):

    test_data_dir = f"{test_data_name}.npz"
    cont_data, cat_data, train_labels = get_torch_agg_data(data_dir=test_data_dir)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model.to(device)
    
    batch_sampler_indices = np.arange(len(cont_data))
    np.random.shuffle(batch_sampler_indices)
    sampler = BatchSampler(batch_sampler_indices, batch_size=BATCH_SIZE, drop_last=False, )
    
    test_pred = np.zeros(len(cont_data))
    for idx, batch in enumerate(sampler):
        cont_feat = torch.as_tensor(cont_data[batch], dtype=torch.float32).to(device)
        cat_feat =  torch.as_tensor(cat_data[batch], dtype=torch.float32).to(device)
        pred = model(cont_feat, cat_feat)        
        test_pred[batch] = pred  # Add to out of folds array

    del cont_data, cat_data
    gc.collect()

    with open(OUTDIR+f'{test_data_name}_id.json', 'r') as f:
            test_id_dict = json.load(f)

    result = pd.DataFrame({"customer_ID":test_id_dict.values(), 
                        "prediction":test_pred.reshape(-1)
                        }
                        )

    sub_file_dir = os.path.join(OUTDIR, f"{model_name}.csv")
    result.set_index("customer_ID").to_csv(sub_file_dir)
    
    merge_with_pred(test_pred, np.arange(len(test_pred)),
                    model_name=model_name, type="test", id_dir=f'{test_data_name}_id.json')
    

@click.command()
@click.option("--agg", default=1)
def run_experiment(agg):
    exp_name = f"train_agg{agg}_mean_q5_q95_q5_q95_data"
    config = {"weight_decay": 0.01, "num_epochs": 100, "conv_channels": 32, "id_dir": f'train_agg{agg}_mean_q5_q95_q5_q95_id.json'}
    model_name = f"conv{config['conv_channels']}_agg"

    
    run_info = config
    tempdir = tempfile.mkdtemp(prefix=f"pd_torch_{exp_name}_", dir=OUTDIR)
    with open(os.path.join(tempdir, "run_info.json"), "w") as fh:
        json.dump(run_info, fh, indent=4)   

    cont_data, cat_data, train_labels = get_torch_agg_data(data_dir=f"train_agg{agg}_mean_q5_q95_q5_q95.npz")
    
    model_name = f"conv_agg{agg}"
    indices = get_customers_data_indices(num_data_points=np.arange(14), id_dir=f'train_agg{agg}_mean_q5_q95_q5_q95_id.json')
    model = train_conv_cv(cont_data, cat_data, train_labels, indices, config, model_name=model_name, tempdir=tempdir, n_folds=5, seed=42)

    test_conv(model, model_name, test_data_name=f"test_agg{agg}_mean_q5_q95_q5_q95")


if __name__ == "__main__":
    run_experiment()