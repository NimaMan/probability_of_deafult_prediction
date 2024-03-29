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
from pd.nn.conv import ConvPred
from pd.utils import merge_with_pred, get_torch_agg_data, get_customers_data_indices, get_pred_data
from pd.params import *


def train_conv_cv(cont_data, pred_data, labels, indices, config, model_name, tempdir=None, n_folds=5, seed=42):
    used_indices, other_indices = indices
    print(f"training the {model_name}", config)
    oof_predictions = np.zeros(len(used_indices))
    best_model_name, best_model_score = "", 0
    
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (trn_ind, val_ind) in enumerate(kfold.split(used_indices, labels[used_indices])):
        print('-'*50)
        print(f'Training fold {fold} ...')
        model = ConvPred(input_dim=cont_data.shape[-1], conv_channels=config["conv_channels"], pred_dim=pred_data.shape[-1])
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
            model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=config["weight_decay"])
        criterion = torch.nn.BCELoss()

        x_cont_train, x_cont_val = cont_data[used_indices][trn_ind], cont_data[used_indices][val_ind]
        y_train, y_val = labels[used_indices][trn_ind], labels[used_indices][val_ind]
        x_pred_train, x_pred_val = pred_data[used_indices][trn_ind], pred_data[used_indices][val_ind]
        
        batch_sampler_indices = np.arange(len(trn_ind))
        np.random.shuffle(batch_sampler_indices)
        sampler = BatchSampler(batch_sampler_indices, batch_size=BATCH_SIZE, drop_last=False, )
        
        for epoch in range(config["num_epochs"]): 
            for idx, batch in enumerate(sampler):
                cont_feat = torch.as_tensor(x_cont_train[batch], dtype=torch.float32).to(device)
                pred_feat =  torch.as_tensor(x_pred_train[batch], dtype=torch.float32).to(device)
                clabel =  torch.as_tensor(y_train[batch], dtype=torch.float32).to(device)
                pred = model(cont_feat, pred_feat)
                #weight = clabel.clone()
                #weight[weight==0] = 4
                #criterion.weight = weight
                loss = criterion(pred, clabel)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                model_metric, gini, recall = amex_metric(clabel.cpu().detach().numpy(), pred.cpu().detach().numpy(), return_components=True)
                val_metrix = 0
                val_gini, val_recall = 0, 0
                if model_metric > valThreshold:
                    cont_feat = torch.as_tensor(x_cont_val, dtype=torch.float32).to(device)
                    pred_feat =  torch.as_tensor(x_pred_val, dtype=torch.float32).to(device)

                    val_pred = model(cont_feat, pred_feat).cpu().detach().numpy()
                    val_metrix, val_gini, val_recall = amex_metric(y_val, val_pred, return_components=True)

                log_message = f"{epoch}, BCE loss: {loss.item():.3f},train -> amex {model_metric:.3f}, gini {gini:.3f}, recall {recall:.3f}, val -> amex {val_metrix:.3f} gini {val_gini:.3f}, recall {val_recall:.3f}"
                print(log_message)
                write_log(log=log_message, log_desc=model_name+"_log", out_dir=tempdir)
        
        cont_feat = torch.as_tensor(x_cont_val, dtype=torch.float32).to(device)
        pred_feat =  torch.as_tensor(x_pred_val, dtype=torch.float32).to(device)
        val_pred = model(cont_feat, pred_feat).cpu().detach().numpy().reshape(-1, )
        oof_predictions[val_ind] = val_pred  # Add to out of folds array
        
        del x_cont_train, y_train
        gc.collect()
        torch.cuda.empty_cache()
        # Compute fold metric
        score, gini, recall = amex_metric(y_val.reshape(-1, ), val_pred, return_components=True)
        torch.save(model.state_dict(), os.path.join(MODELDIR, f'{model_name}_{int(score*10000)}'))                
        if score > best_model_score:
            best_model_name = f'{model_name}_{int(score*10000)}'
            best_model_score = score

        pred_indices = used_indices[val_ind]
        merge_with_pred(val_pred, pred_indices, model_name=model_name, id_dir=f'train_logistic_raw_all_mean_q5_q95_q5_q95_id.json')
    
        print(f'Our fold {fold} CV score is {score}')
        del y_val, x_cont_val
        gc.collect()
        torch.cuda.empty_cache()
    score, gini, recall = amex_metric(labels[used_indices].reshape(-1, ), oof_predictions, return_components=True)  # Compute out of folds metric
    print(f'Our out of folds CV score is {score}, Gini {gini}, recall {recall}')
    
    return model


def test_conv(model, model_name, ):

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model.to(device)
    
    test_data = np.load(OUTDIR+"test_logistic_raw_all_mean_q5_q95_q5_q95_data.npy")
    pred_data = get_pred_data(type="train", id_dir='test_logistic_raw_all_mean_q5_q95_q5_q95_id.json', cols=cols)
    
    batch_sampler_indices = np.arange(len(test_data))
    np.random.shuffle(batch_sampler_indices)
    sampler = BatchSampler(batch_sampler_indices, batch_size=BATCH_SIZE, drop_last=False, )
    
    test_pred = np.zeros(len(test_data))
    for idx, batch in enumerate(sampler):

        cont_feat = torch.as_tensor(test_data[batch], dtype=torch.float32).to(device)
        pred_feat =  torch.as_tensor(pred_data[batch], dtype=torch.float32).to(device)
        pred = model(cont_feat, pred_feat)
        test_pred[batch] = pred  # Add to out of folds array
 
    del test_data
    gc.collect()

    with open(OUTDIR+f'train_logistic_raw_all_mean_q5_q95_q5_q95_id.json', 'r') as f:
            test_id_dict = json.load(f)

    result = pd.DataFrame({"customer_ID":test_id_dict.values(), 
                        "prediction":test_pred.reshape(-1)
                        }
                        )

    sub_file_dir = os.path.join(OUTDIR, f"{model_name}.csv")
    result.set_index("customer_ID").to_csv(sub_file_dir)
    
    merge_with_pred(test_pred,  np.arange(len(test_pred)), model_name=model_name, id_dir=f'test_logistic_raw_all_mean_q5_q95_q5_q95_id.json')

    
def run_experiment():
    config = {"weight_decay": 0.01, "num_epochs": 95, "conv_channels": 32}
    model_name = f"conv{config['conv_channels']}_pred"
    cols = ['xgbm13_p0_agg4', 'catb13_agg4', 'xgbm_p0_agg4', 'catb_agg4',
       'xgbm13_p0_agg1', 'catb13_agg1', 'xgbm_p0_agg1', 'catb_agg1',
       'xgbm13_p0_agg2', 'catb13_agg2', 'xgbm_p0_agg2', 'catb_agg2',
       'xgbm13_p0_agg0', 'catb13_agg0', 'xgbm_p0_agg0', 'catb_agg0',
       'xgbmv213_p0_agg0', 'xgbmv2_p0_agg0']
    
    run_info = config
    tempdir = tempfile.mkdtemp(prefix=f"pd_torch_pred_", dir=OUTDIR)
    with open(os.path.join(tempdir, "run_info.json"), "w") as fh:
        json.dump(run_info, fh, indent=4)   

    train_data = np.load(OUTDIR+"train_logistic_raw_all_mean_q5_q95_q5_q95_data.npy")
    train_labels = np.load(OUTDIR+"train_logistic_raw_all_mean_q5_q95_q5_q95_labels.npy")
    pred_data = get_pred_data(type="train", id_dir='train_logistic_raw_all_mean_q5_q95_q5_q95_id.json', cols=cols)

    model_name = f"conv_pred"
    indices = get_customers_data_indices(num_data_points=np.arange(14), id_dir=f'train_logistic_raw_all_mean_q5_q95_q5_q95_id.json')
    model = train_conv_cv(train_data, pred_data, train_labels, indices, config, model_name=model_name, tempdir=tempdir, n_folds=5, seed=42)

    test_conv(model, model_name)


if __name__ == "__main__":
    run_experiment()