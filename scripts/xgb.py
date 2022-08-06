import os
import gc
import tempfile
import json
import warnings
warnings.filterwarnings('ignore')
import scipy as sp
import numpy as np
import pandas as pd

import random
import joblib
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import xgboost as xgb
from pd.params import *
from pd.gmb_utils import lgb_amex_metric


def get_agg_data(data_dir="train_agg_mean_q5_q95_q5_q95.npz"):
    d = np.load(OUTDIR+data_dir)
    #train_data = np.concatenate((d["d2"].astype(np.int32), d["d1"].reshape(d["d1"].shape[0], -1)), axis=1)
    train_labels = d["labels"]
    df2 = pd.DataFrame(d["d2"].astype(np.int32))
    df = pd.DataFrame(d["d1"].reshape(d["d1"].shape[0], -1))
    df = pd.concat((df2, df), axis=1,)
    df.columns = [f"c{i}" for i in range(df.shape[1])]
    cat_indices = list(np.arange(33))

    return df, train_labels, cat_indices


if __name__ == "__main__":
    agg = 0 
    exp_name = f"train_agg{agg}_mean_q5_q95_q5_q95_data"
    # XGB MODEL PARAMETERS
    random_state = 42
    xgb_parms = { 
    'max_depth':4, 
    'learning_rate':0.05, 
    'subsample':0.8,
    'colsample_bytree':0.6, 
    'eval_metric':'logloss',
    'objective':'binary:logistic',
    'tree_method':'gpu_hist',
    'predictor':'gpu_predictor',
    'random_state':random_state
    }
    
    run_info = params

    tempdir = tempfile.mkdtemp(prefix=f"pd_lgbm_{exp_name}_", dir=OUTDIR)
    with open(os.path.join(tempdir, "run_info.json"), "w") as fh:
        json.dump(run_info, fh, indent=4)

    train_data, train_labels, cat_indices = get_agg_data(data_dir=f"train_agg{agg}_mean_q5_q95_q5_q95.npz")
    X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=1/9, random_state=0, shuffle=True)
    validation_data = (X_test, y_test)

    
    print(f"Start training LGB {exp_name} with number of feature {X_train.shape[1]}", params)
     # TRAIN, VALID, TEST FOR FOLD K
    Xy_train = IterLoadForDMatrix(train.loc[train_idx], FEATURES, 'target')
    X_valid = train.loc[valid_idx, FEATURES]
    y_valid = train.loc[valid_idx, 'target']
    
    dtrain = xgb.DeviceQuantileDMatrix(Xy_train, max_bin=256)
    dvalid = xgb.DMatrix(data=X_valid, label=y_valid)
    
    # TRAIN MODEL FOLD K
    model = xgb.train(xgb_parms, 
                dtrain=dtrain,
                evals=[(dtrain,'train'),(dvalid,'valid')],
                num_boost_round=9999,
                early_stopping_rounds=100,
                verbose_eval=100) 
    model.save_model(f'XGB_v{VER}_fold{fold}.xgb')
    
    # GET FEATURE IMPORTANCE FOR FOLD K
    dd = model.get_score(importance_type='weight')
    df = pd.DataFrame({'feature':dd.keys(),f'importance_{fold}':dd.values()})
    importances.append(df)
            
    # INFER OOF FOLD K
    oof_preds = model.predict(dvalid)
    acc = amex_metric_mod(y_valid.values, oof_preds)
    print('Kaggle Metric =',acc,'\n')
    
    # SAVE OOF
    df = train.loc[valid_idx, ['customer_ID','target'] ].copy()
    df['oof_pred'] = oof_preds
    oof.append( df )
    
    del dtrain, Xy_train, dd, df
    del X_valid, y_valid, dvalid, model
    _ = gc.collect()
    
    joblib.dump(model, filename=MODELDIR+exp_name)
        
    del train_data, X_test, X_train, validation_data, lgb_train, lgb_valid

    test_data, labels, cat_indices = get_agg_data(data_dir=f"test_agg{agg}_mean_q5_q95_q5_q95.npz")

    test_pred = model.predict(test_data) # Predict the test set
    del test_data
    gc.collect()

    with open(OUTDIR+f'test_agg{agg}_mean_q5_q95_q5_q95_id.json', 'r') as f:
            test_id_dict = json.load(f)
    
    result = pd.DataFrame({"customer_ID":test_id_dict.values(), 
                        "prediction":test_pred.reshape(-1)
                        }
                        )

    sub_file_dir = os.path.join(OUTDIR, f"lgbm_{exp_name}_sub.csv")
    result.set_index("customer_ID").to_csv(sub_file_dir)


