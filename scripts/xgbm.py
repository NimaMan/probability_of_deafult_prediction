import os
import gc
import tempfile
import click
import json
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

import joblib
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

import pd.metric as metric
from pd.utils import merge_with_pred_df, get_customers_id_from_indices
from pd.params import *
from pd.gmb_utils import get_agg_data, xgb_amex, focal_loss_xgb


def train_xgb_cv(data, test, labels, params, model_name, id_dir, n_folds=5, seed=42):
    
    print(f"training the {model_name}", params)
    oof_predictions = np.zeros(len(data))
    test_predictions = np.zeros(len(test))
    
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (trn_ind, val_ind) in enumerate(kfold.split(data, labels)):
        print(' ')
        print('-'*50)
        print(f'Training fold {fold} ...')
        x_train, x_val = data.iloc[trn_ind], data.iloc[val_ind]
        y_train, y_val = labels[trn_ind], labels[val_ind]
        
        xgb_train = xgb.DMatrix(x_train, y_train, )
        xgb_valid = xgb.DMatrix(x_val, y_val, )

        model =  xgb.train(
            obj = focal_loss_xgb,
            params = params,
            dtrain = xgb_train,
            maximize = True,
            num_boost_round = 10500,
            evals = [(xgb_train, 'train'), (xgb_valid, 'valid')],
            early_stopping_rounds = 100,
            verbose_eval = 100,
            feval = xgb_amex
            )
        val_pred = model.predict(xgb.DMatrix(x_val)) # Predict validation
        oof_predictions[val_ind] = val_pred  # Add to out of folds array
        
        # Predict the test set
        test_pred = model.predict(xgb.DMatrix(test))
        test_predictions += test_pred / n_folds
        
        # Compute fold metric
        score, gini, recall = metric.amex_metric(y_val.reshape(-1, ), val_pred, return_components=True)
        joblib.dump(model, os.path.join(MODELDIR, f'{model_name}_{int(score*10000)}'))
    
        print(f'Our fold {fold} CV score is {score}')
        del x_train, x_val, y_train, y_val, xgb_train, xgb_valid
        gc.collect()
    
    score, gini, recall = metric.amex_metric(labels.reshape(-1, ), oof_predictions, return_components=True)  # Compute out of folds metric
    print(f'Our out of folds CV score is {score}, Gini {gini}, recall {recall}')
    
    train_customers = get_customers_id_from_indices(np.arange(len(oof_predictions)), id_dir["train"])
    oof_df = pd.DataFrame({'customer_ID': train_customers, model_name: oof_predictions})
    merge_with_pred_df(oof_df, type="train")
    
    # Create a dataframe to store test prediction
    test_customers = get_customers_id_from_indices(np.arange(len(test_predictions)), id_dir["test"])
    test_df = pd.DataFrame({'customer_ID': test_customers, model_name: test_predictions})
    merge_with_pred_df(test_df, type="test")
 

def test_xgb(model, model_name, test_data_name=f"test_agg1_mean_q5_q95_q5_q95",):

    test_data_dir = f"{test_data_name}.npz"
    test_data, labels, cat_indices = get_agg_data(data_dir=test_data_dir)
    test_data = xgb.DMatrix(test_data)
    test_pred = model.predict(test_data) # Predict the test set
    del test_data
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

    id_dir = {"train": f"train_agg{agg}_mean_q5_q95_q5_q95_id.json", 
              "test": f"test_agg{agg}_mean_q5_q95_q5_q95_id.json",}    
    train_data, train_labels, cat_indices = get_agg_data(data_dir=f"train_agg{agg}_mean_q5_q95_q5_q95.npz", agg=agg)
    test_data, labels, cat_indices = get_agg_data(data_dir=f"test_agg{agg}_mean_q5_q95_q5_q95.npz")
    
    for seed in [42, 52, 62, 82]:
        model_name = f"xgbm_focal_seed{seed}_agg{agg}"
        params = {
        #'objective': focal_loss_xgb,
        #'metric': "binary_logloss",
        #'eval_metric':'logloss',
        'disable_default_eval_metric': 1,
        'seed': seed,
        'learning_rate': 0.05,
        'n_jobs': -1,
        'colsample_bytree': 0.6,
        'gamma':1.5,
        'min_child_weight':8,
        'lambda':70,
        'max_bin': 128,  # Deafult is 255
        'tree_method':'gpu_hist',
        'predictor':'gpu_predictor',
        'max_depth': 4, 
        'subsample': 0.8,
        }
        
        train_xgb_cv(train_data, test_data, train_labels, params, 
                            model_name=model_name, id_dir=id_dir, n_folds=5, seed=seed)


if __name__ == "__main__":
    run_experiment()