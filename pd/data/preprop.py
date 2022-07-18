
import gc
import warnings
warnings.filterwarnings('ignore')
import scipy as sp
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
import itertools
from pd.params import *


def get_col_info(train_data=None, col_info_name="col_info", c13=False):
    cols = featureCols

    if train_data is None:
        train_data = pd.read_parquet(DATADIR+"train_data.parquet")
    if c13:
        train_customers = train_data.customer_ID
        train_count =  train_customers.value_counts()
        train_c13 = train_count[train_count==13].index
        train_data = train_data[train_data.customer_ID.isin(train_c13)]
        col_info_name = f"{col_info_name}13"

    for cat_col in CATCOLS:
        encoder = LabelEncoder()
        train_data[cat_col] = encoder.fit_transform(train_data[cat_col])
    
    col_info = {}
    for c in cols:
        col_info[c] = {}
        d = train_data[c]
        q2 = d.quantile(0.02)
        q98 = d.quantile(0.98)
        col_min_val = d.min()
        col_max_val = d.max()
        hist = np.histogram(d, range=[q2, q98], density=True, bins=100)
        
        col_info[c]["num_nan"] = 1 - d.dropna().shape[0]/d.shape[0]
        col_info[c]["q2"] = q2
        col_info[c]["q98"] = q98
        col_info[c]["q1"] = d.quantile(0.01)
        col_info[c]["q99"] = d.quantile(0.99)
        
        col_info[c]["min"] = col_min_val
        col_info[c]["max"] = col_max_val
        col_info[c]["mean"] = d.mean()
        col_info[c]["median"] = d.quantile(0.5)
        col_info[c]["hist"] = hist
        col_info[c]["max_prob_mass"] = hist[0].max()
        col_info[c]["num_nonzero_bins"] = np.count_nonzero(hist[0])
        
    with open(OUTDIR+f"{col_info_name}.pkl", "wb") as f:
        pickle.dump(col_info, f)

    return col_info


def get_kaggle_79_feat(data, train_labels):
    data_cont_agg = data.groupby("customer_ID")[ContCols].agg(['mean', 'std', 'min', 'max', 'last'])
    data_cont_agg.columns = ['_'.join(x) for x in data_cont_agg.columns]
    data_cont_agg.reset_index(inplace=True)

    data_cat_agg = data.groupby("customer_ID")[CATCOLS].agg(['count', 'last', 'nunique'])
    data_cat_agg.columns = ['_'.join(x) for x in data_cat_agg.columns]
    data_cat_agg.reset_index(inplace=True)
    data = data_cont_agg.merge(data_cat_agg, how='inner', on='customer_ID')
    
    if train_labels is not None:
        data = data.merge(train_labels, how='inner', on='customer_ID')
    
    del data_cont_agg, data_cat_agg
    gc.collect()
    
    return data


def get_raw_features(customer_ids, train_data, train_labels=None, test_mode=False, normalize=True):
    cols = featureCols
    # fill nan with mean of each columns 
    for c in cols:
        train_data[c] = train_data[c].fillna(col_info13[c]["mean"])
        if normalize:
            if c in ContCols:
                train_data[c] = (train_data[c] - col_info13[c]["q1"])/ (col_info13[c]["q99"] - col_info13[c]["q1"])
        
    customer_data = train_data.groupby("customer_ID")
    labels_array = np.zeros((len(set(customer_ids)) ,1))
    id_dict = {}
    d = np.zeros((len(set(customer_ids)), 13, len(cols)), dtype=np.float32) # init with zeros
    for idx, c in enumerate(set(customer_ids)):
        cd = customer_data.get_group(c)[cols].values
        num_data_point = cd.shape[0]
        d[idx, -num_data_point:, :] = cd
        id_dict[idx] = c
        if not test_mode:
            label = train_labels.loc[c]
            labels_array[idx] = label
    
    return d, labels_array, id_dict


def preprocess_data(data_type="train", c13=False, feat_type="raw_all"):
    
    if data_type == "train":
        data = pd.read_parquet(DATADIR+"train_data.parquet")
        train_labels = pd.read_csv(DATADIR+"train_labels.csv")
    else:
        data = pd.read_parquet(DATADIR+"test_data.parquet")
        train_labels = None
    
    customer_count =  data.customer_ID.value_counts()
    customers = customer_count.index
    if c13:
        print("getting data of the 13 customers")
        customers = customer_count[customer_count==13].index
        data = data[data.customer_ID.isin(customers)]
        data_type = f"{data_type}13"
    print('Starting feature engineer...')
    if feat_type == "kaggle97":
        data = get_kaggle_79_feat(data, train_labels)
    elif feat_type == "raw_all":
        data, labels_array, id_dict = get_raw_features(customers, data, train_labels=train_labels.set_index("customer_ID"))
    else:
        raise NotImplementedError
        
    if c13:
        np.save(OUTDIR+"c13_labels.npy", labels_array)
        np.save(OUTDIR+"c13_data.npy", data)
    else:
        data.to_parquet(OUTDIR+f"{data_type}_fe.parquet")