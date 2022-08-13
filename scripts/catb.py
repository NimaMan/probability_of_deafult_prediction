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
from catboost import CatBoostClassifier

import pd.metric as metric
from pd.utils import merge_with_pred, get_customers_data_indices
from pd.params import *
from pd.gmb_utils import get_agg_data, xgb_amex


def train_catb_cv(data, labels, indices, params, model_name, id_dir=None, n_folds=5, seed=42):
    """
    take the data in certain indices [for example c13 data]
    """
    used_indices, other_indices = indices
    print(f"training the {model_name}", params)
    oof_predictions = np.zeros(len(used_indices))
    best_model_name, best_model_score = "", 0
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (trn_ind, val_ind) in enumerate(kfold.split(data.iloc[used_indices], labels[used_indices], used_indices)):
        print(' ')
        print('-'*50)
        print(f'Training fold {fold} ...')
        x_train, x_val = data.iloc[used_indices].iloc[trn_ind], data.iloc[used_indices].iloc[val_ind]
        y_train, y_val = labels[used_indices][trn_ind], labels[used_indices][val_ind]
        
        cat_features = []
        for i in range(11):
            cat_features.append(f"c{i*4 + 1}")
            cat_features.append(f"c{i*4 + 2}")

        cat_features = [i for i in range(44) if i]
        model = CatBoostClassifier(iterations=10000, random_state=42, task_type=params["device"])
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], cat_features=cat_features,  verbose=100)
        val_pred = model.predict_proba(x_val)[:, 1]        
        oof_predictions[val_ind] = val_pred  # Add to out of folds array
        
        # Compute fold metric
        score, gini, recall = metric.amex_metric(y_val.reshape(-1, ), val_pred, return_components=True)
        joblib.dump(model, os.path.join(MODELDIR, f'{model_name}_{int(score*10000)}'))
        if score > best_model_score:
            best_model_name = f'{model_name}_{int(score*10000)}'
            best_model_score = score

        pred_indices = used_indices[val_ind]
        merge_with_pred(val_pred, pred_indices, model_name=model_name, id_dir=id_dir)
    
        print(f'Our fold {fold} CV score is {score}')
        del x_train, x_val, y_train, y_val
        gc.collect()
    score, gini, recall = metric.amex_metric(labels[used_indices].reshape(-1, ), oof_predictions, return_components=True)  # Compute out of folds metric
    print(f'Our out of folds CV score is {score}, Gini {gini}, recall {recall}')
    
    best_model = joblib.load(os.path.join(MODELDIR, best_model_name))
    if len(other_indices) > 0:
        other_pred = best_model.predict_proba(data.iloc[other_indices])[:, 1]
        merge_with_pred(other_pred, other_indices, model_name=model_name, id_dir=id_dir)
    
    return best_model


def test_catb(model, model_name, test_data_name=f"test_agg1_mean_q5_q95_q5_q95"):

    test_data_dir = f"{test_data_name}.npz"
    test_data, labels, cat_indices = get_agg_data(data_dir=test_data_dir)
    test_pred = model.predict_proba(test_data)[:, 1] # Predict the test set
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
    exp_name = f"train_agg{agg}_mean_q5_q95_q5_q95_data"
    params = {
        "device": "GPU"
        }
    
    id_dir = f"train_agg{agg}_mean_q5_q95_q5_q95_id.json"
    run_info = params
    #tempdir = tempfile.mkdtemp(prefix=f"pd_catb_{exp_name}_", dir=OUTDIR)
    #with open(os.path.join(tempdir, "run_info.json"), "w") as fh:
    #    json.dump(run_info, fh, indent=4)   

    train_data, train_labels, cat_indices = get_agg_data(data_dir=f"train_agg{agg}_mean_q5_q95_q5_q95.npz")
    
    model_name = f"catb13_agg{agg}"
    indices = get_customers_data_indices(num_data_points=[13], id_dir=f'train_agg{agg}_mean_q5_q95_q5_q95_id.json')
    model = train_catb_cv(train_data, train_labels, indices, params, model_name=model_name, id_dir=id_dir, n_folds=5, seed=42)

    model_name = f"catb_agg{agg}"
    indices = get_customers_data_indices(num_data_points=np.arange(14), id_dir=f'train_agg{agg}_mean_q5_q95_q5_q95_id.json')
    model = train_catb_cv(train_data, train_labels, indices, params, model_name=model_name, id_dir=id_dir, n_folds=5, seed=42)

    test_catb(model, model_name, test_data_name=f"test_agg{agg}_mean_q5_q95_q5_q95")


if __name__ == "__main__":
    run_experiment()