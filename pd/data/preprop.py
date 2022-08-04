
import gc
import json
import warnings
warnings.filterwarnings('ignore')
import scipy as sp
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
import itertools
from pd.params import *


def get_data_stats(d):
    stats = {}
    q2 = d.quantile(0.02)
    q98 = d.quantile(0.98)
    col_min_val = d.min()
    col_max_val = d.max()
    hist = np.histogram(d, range=[q2, q98], density=True, bins=100)

    stats["num_nan"] = 1 - d.dropna().shape[0]/d.shape[0]
    stats["q2"] = q2
    stats["q98"] = q98
    stats["q1"] = d.quantile(0.01)
    stats["q99"] = d.quantile(0.99)
    stats["q5"] = d.quantile(0.05)
    stats["q95"] = d.quantile(0.95)

    stats["min"] = col_min_val
    stats["max"] = col_max_val
    stats["mean"] = d.mean()
    stats["median"] = d.quantile(0.5)
    stats["hist"] = hist
    stats["max_prob_mass"] = hist[0].max()
    stats["num_nonzero_bins"] = np.count_nonzero(hist[0])
    for lb, ub in [("q1", "q99"), ("q2", "q98"), ("q5", "q95")]:
        fd = d[(d <= stats[ub]) & (d >= stats[lb])]
        stats[f"mean_{lb}_{ub}"] = fd.mean()
        stats[f"std_{lb}_{ub}"] = fd.std()
    
    return stats


def get_col_info(train_data=None, col_info_name="col_info", c13=False):
    cols = featureCols

    if train_data is None:
        train_data = pd.read_parquet(TRAINDATA)
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
        d = train_data[c]
        col_stats = get_data_stats(d)
        col_info[c] = col_stats

    with open(OUTDIR+f"{col_info_name}.pkl", "wb") as f:
        pickle.dump(col_info, f)

    return col_info


def scaler_transform(d, c, borders):
    
    return (d - col_info13[c][borders[0]])/(col_info13[c][borders[1]] - col_info13[c][borders[0]])


def logistic_transform(d, c, borders):
    mid_point = np.mean([col_info13[c]["q5"], col_info13[c]["q95"]])
    slope = 2.944/(col_info13[c]["q95"] - mid_point)
    
    return 1/(1 + np.exp(slope*(mid_point - d)))


def transform(d, c, borders, type="logistic"):
    if type == "scaler":
        return scaler_transform(d, c, borders)
    elif type == "logistic":
        return logistic_transform(d, c, borders)


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
                if (col_info13[c]["q99"] - col_info13[c]["q1"]) != 0:
                    train_data[c] = (train_data[c] - col_info13[c]["q1"])/(col_info13[c]["q99"] - col_info13[c]["q1"])
                    # some cols end up with NaN vals    

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


def get_raw_features_fill(customer_ids, train_data, train_labels=None, test_mode=False, 
                        normalizer="logistic", time_dim=13, fillna="mean_q5_q95", borders=("q5", "q95")):
    cols = featureCols
    # fill nan with mean of each columns
    fill_feats = [] 
    for c in cols:
        train_data[c] = train_data[c].fillna(col_info13[c][fillna])
        if c in ContCols:
            if normalizer is not None:
                if (col_info13[c][borders[1]] - col_info13[c][borders[0]]) != 0:
                    train_data[c] = transform(train_data[c], c, borders, type=normalizer)
                    fill_feats.append(transform(col_info13[c][fillna], c, borders, type=normalizer))
                else:
                    fill_feats.append(col_info13[c][fillna]) 
        else:
            fill_feats.append(col_info13[c][fillna]) 
                
    customer_data = train_data.groupby("customer_ID")
    labels_array = np.zeros((len(set(customer_ids)) ,1))
    id_dict = {}
    d = np.ones((len(set(customer_ids)), time_dim, len(cols)), dtype=np.float32)*np.array(fill_feats, dtype=np.float32).reshape(1, 1, len(cols))
    for idx, c in enumerate(set(customer_ids)):
        cd = customer_data.get_group(c)[cols].values
        num_data_point = cd.shape[0]
        d[idx, -num_data_point:, :] = cd
        id_dict[idx] = c
        if not test_mode:
            label = train_labels.loc[c]
            labels_array[idx] = label
    
    return d, labels_array, id_dict


def get_feat_comb(data, train_labels):
    nonk79_feat = [col for col in dataCols if col not in betterTransFeatsK79]

    data, labels_array, id_dict = get_raw_features_fill(customers, data, 
                                            train_labels=train_labels.set_index("customer_ID"), time_dim=data_time_dim, 
                                            fillna=fillna, borders=borders, normalize=normalize)

    cont_k97  = [ col for col in ContCols if col in betterTransFeatsK79]
    data_cont_agg = data.groupby("customer_ID")[cont_k97].agg(['mean', 'std', 'min', 'max', 'last'])
    data_cont_agg.columns = ['_'.join(x) for x in data_cont_agg.columns]
    data_cont_agg.reset_index(inplace=True)

    cat_k97  = [ col for col in ContCols if col in betterTransFeatsK79]
    data_cat_agg = data.groupby("customer_ID")[cat_k97].agg(['count', 'last', 'nunique'])
    data_cat_agg.columns = ['_'.join(x) for x in data_cat_agg.columns]
    data_cat_agg.reset_index(inplace=True)
    data = data_cont_agg.merge(data_cat_agg, how='inner', on='customer_ID')
    
    if train_labels is not None:
        data = data.merge(train_labels, how='inner', on='customer_ID')
    
    del data_cont_agg, data_cat_agg
    gc.collect()
    
    return data


def preprocess_data(data_type="train", feat_type="raw_all", time_dim=13, all_data=True, fillna="mean", borders=("q1", "q99"), normalizer="logistic",):
    """
    
    """
    if data_type == "train":
        data = pd.read_parquet(DATADIR+"train_data.parquet")
        train_labels = pd.read_csv(DATADIR+"train_labels.csv")
    else:
        data = pd.read_parquet(DATADIR+"test_data.parquet")
        train_labels = None
    
    customer_count =  data.customer_ID.value_counts()
    customers = customer_count.index
    output_file_name = data_type
    if not all_data:
        print(f"getting data of the {time_dim} customers")
        customers = customer_count[customer_count==time_dim].index
        data = data[data.customer_ID.isin(customers)]
        output_file_name = f"{data_type}{time_dim}"
    print('Starting feature engineer...')
    if feat_type == "kaggle79":
        data = get_kaggle_79_feat(data, train_labels)
    elif feat_type == "raw_all":
        data_time_dim = time_dim
        if all_data:
            data_time_dim = 13
        if data_type == "train":
            data, labels_array, id_dict = get_raw_features_fill(customers, data, 
                                            train_labels=train_labels.set_index("customer_ID"), time_dim=data_time_dim, 
                                            fillna=fillna, borders=borders, normalizer=normalizer)
        else:
            data, labels_array, id_dict = get_raw_features_fill(customers, data, test_mode=True, time_dim=data_time_dim, 
                                                                fillna=fillna, borders=borders, normalizer=normalizer,)
    else:
        raise NotImplementedError
    #if time_dim is not None:
    #    if data_type == "train":
    #        np.save(OUTDIR+f"{output_file_name}_{feat_type}_{fillna}_{borders[0]}_{borders[1]}_labels.npy", labels_array)
    #    np.save(OUTDIR+f"{output_file_name}_{feat_type}_{fillna}_{borders[0]}_{borders[1]}_data.npy", data)        
    #else:
    output_file_name = output_file_name + f"_{normalizer}"
    try:
        if data_type == "train":
            np.save(OUTDIR+f"{output_file_name}_{feat_type}_{fillna}_{borders[0]}_{borders[1]}_labels.npy", labels_array)
        with open(OUTDIR+f"{output_file_name}_{feat_type}_{fillna}_{borders[0]}_{borders[1]}_id.json", 'w') as fp:
            json.dump(id_dict, fp)
        np.save(OUTDIR+f"{output_file_name}_{feat_type}_{fillna}_{borders[0]}_{borders[1]}_data.npy", data)
    except Exception:
        data.to_parquet(OUTDIR+f"{output_file_name}_{feat_type}.parquet")