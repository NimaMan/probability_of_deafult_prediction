
import gc
import json
import warnings
warnings.filterwarnings('ignore')
import scipy as sp
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pd.data.scaler import transform
from pd.params import *


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


def get_raw_features_fill(train_data, train_labels=None, test_mode=False, 
                        normalizer="logistic", time_dim=13, fillna="mean_q5_q95", borders=("q5", "q95"), 
                        cols=None):
    
    customer_count =  train_data.customer_ID.value_counts()
    customer_ids = customer_count.index
    if train_labels is not None:
        train_labels.set_index("customer_ID", inplace=True)
    
    if cols is None:
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
            labels_array[idx] = train_labels.loc[c]
    
    return d, labels_array, id_dict


def get_col_names(agg=0):
    colo = None
    if agg == 0:
        cols13 = bestCols
        colo = ContCols
    elif agg == 1:
        cols13 = ContCols + ["B_38"]
    elif agg == 2:
        cols13 = [col for col in ContCols if col not in betterTransFeatsK79] + ["B_38"]
        colo = [col for col in ContCols if col in betterTransFeatsK79]
    elif agg == 3:
        cols13 = [col for col in ContCols if col not in betterTransFeatsK79] + ["B_38"]
        colo = ContCols
    #elif agg == 4:
    #    cols13 = [col for col in ContCols if col not in betterTransFeatsK79 and col not in MostNaNCols] + ["B_38"]
    #    colo = [col for col in set(MostNaNCols + betterTransFeatsK79) if col not in CATCOLS]
    
    elif agg == 4:
        cols13 = [col for col, metric in sfa_gbm.items() if metric[0]>0.4] + ["B_38"]
        colo =[col for col, metric in sfa_gbm.items() if metric[0]>0.2 if col not in CATCOLS and col not in cols13]
    
    return cols13, colo
    

def get_feat_comb(data_type="train", 
                        normalizer="logistic", time_dim=13, fillna="mean_q5_q95", borders=("q5", "q95"), 
                        agg=1):
    
    output_file_name = f"{data_type}_agg{agg}"
    if data_type == "train":
        test_mode = False
        data = pd.read_parquet(TRAINDATA)
        train_labels = pd.read_csv(DATADIR+"train_labels.csv")
    else:
        test_mode = True
        data = pd.read_parquet(DATADIR+"test_data.parquet")
        train_labels = None
    
    cols13, colo = get_col_names(agg=agg)
    data13, labels, id_dict = get_raw_features_fill(data, test_mode=test_mode,
                                            train_labels=train_labels, time_dim=time_dim, 
                                            fillna=fillna, borders=borders, normalizer=normalizer, cols=cols13)
    
    customers = [id_dict[idx] for idx in range(len(id_dict))]
    data_cat_agg = data.groupby("customer_ID")[CATCOLS].agg(['count', 'first', 'last', 'nunique'])
    data_cat_agg = data_cat_agg.loc[customers].values.astype(np.int32)

    if colo is not None:
        data_cont_agg = data.groupby("customer_ID")[colo].agg(['mean', 'std', 'min', 'max', 'first', 'last'])
        data_cont_agg = data_cont_agg.loc[customers].values.astype(np.float32)
        data_cat_agg = np.concatenate([data_cat_agg, data_cont_agg], axis=-1)
    
    output_file_name = f"{output_file_name}_{fillna}_{borders[0]}_{borders[1]}"
    out_dir = OUTDIR+f"{output_file_name}.npz"
    if test_mode:
        labels = np.empty(1)
    
    with open(OUTDIR+f"{output_file_name}_id.json", 'w') as fp:
            json.dump(id_dict, fp)
    np.savez(out_dir, d1=data13, d2=data_cat_agg, labels=labels)


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
            data, labels_array, id_dict = get_raw_features_fill(data, 
                                            train_labels=train_labels, time_dim=data_time_dim, 
                                            fillna=fillna, borders=borders, normalizer=normalizer)
        else:
            data, labels_array, id_dict = get_raw_features_fill(data, test_mode=True, time_dim=data_time_dim, 
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