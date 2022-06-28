
import json
import numpy as np
import pandas as pd
from pd.params import *

def get_col_info(col_info_name="col_info"):
    train_data = pd.read_parquet(DATADIR+"train_data.parquet")
    col_info = {}

    for c in ContCols:
        col_info[c] = {}
        d = train_data[c]
        q2 = d.quantile(0.02)
        q98 = d.quantile(0.98)
        col_min_val = d.min()
        col_max_val = d.max()
        hist = np.histogram(d, range=[q2, q98], density=True, bins=100)
        
        col_info[c]["q2"] = q2
        col_info[c]["q98"] = q98
        col_info[c]["min"] = col_min_val
        col_info[c]["max"] = col_max_val
        col_info[c]["hist"] = hist
        col_info[c]["max_prob_mass"] = hist[0].max()
        
    with open(OUTDIR+f"{col_info_name}.pkl", "wb") as f:
        pickle.dump(col_info, f)

    return col_info


def get_c13_data(customer_ids, customer_data, cols, train_labels=None, test_mode=False):
    d = np.zeros((len(set(customer_ids)), 13, len(cols)))

    labels_array = np.zeros((len(set(customer_ids)) ,1))
    id_dict = {}

    for idx, c in enumerate(set(customer_ids)):
        cd = customer_data.get_group(c)[cols].values
        d[idx, :, :] = cd
        id_dict[idx] = c
        if not test_mode:
            label = train_labels.loc[c]
            labels_array[idx] = label
    d = np.nan_to_num(d)
    if test_mode:
        return d, (None, id_dict)
    else:
        return d, (labels_array, id_dict)


def write_train13_npy(cols):
    train_data = pd.read_parquet(DATADIR+"train_data.parquet")
    train_labels = pd.read_csv(DATADIR+"train_labels.csv")
    train_labels.set_index("customer_ID", inplace=True)
    train_customers = train_data.customer_ID
    train_count =  train_customers.value_counts()
    train_c13 = train_count[train_count==13].index

    train_data = train_data[train_data.customer_ID.isin(train_c13)]
    train_g13 = train_data.groupby("customer_ID")

    dtrain, (labels_array, train_id_dict) = get_c13_data(train_c13, train_g13, cols=cols, 
                                            train_labels=train_labels, test_mode=False)

    #extracted_data = {"train_data":dtrain, "labels": labels_array, "train_ids":train_id_dict}
    np.save(OUTDIR+"train_data_c13.npy", dtrain)
    np.save(OUTDIR+"train_labels_c13.npy", labels_array)


def write_test13_npy(cols):
    test_data = pd.read_parquet(DATADIR+"test_data.parquet")
    test_customers = test_data.customer_ID
    test_count =  test_customers.value_counts()
    test_c13 = test_count[test_count==13].index

    test_data = test_data[test_data.customer_ID.isin(test_c13)]
    test_g13 = test_data.groupby("customer_ID")

    dtest, (labels_array, test_id_dict) = get_c13_data(test_c13, test_g13, cols=cols, 
                                            train_labels=None, test_mode=True)

    np.save(OUTDIR+"test_data_c13.npy", dtest)
    with open(OUTDIR+'test_c13_id_dict.json', 'w') as fp:
        json.dump(test_id_dict, fp)


def get_customer_data(customer_ids, customer_data, cols, train_labels=None, test_mode=False):
    d = np.zeros((len(set(customer_ids)), 13, len(cols)), dtype=np.float32)

    labels_array = np.zeros((len(set(customer_ids)) ,1))
    id_dict = {}

    for idx, c in enumerate(set(customer_ids)):
        cd = customer_data.get_group(c)[cols].values
        num_data_point = cd.shape[0]
        d[idx, -num_data_point:, :] = cd
        id_dict[idx] = c
        if not test_mode:
            label = train_labels.loc[c]
            labels_array[idx] = label
    d = np.nan_to_num(d)
    if test_mode:
        return d, (None, id_dict)
    else:
        return d, (labels_array, id_dict)


def write_train_npy(cols, col_info=None):
    train_data = pd.read_parquet(DATADIR+"train_data.parquet")
    train_labels = pd.read_csv(DATADIR+"train_labels.csv")
    train_labels.set_index("customer_ID", inplace=True)
    train_customers = train_data.customer_ID
    train_count =  train_customers.value_counts()
    train_customers = train_count.index

    if col_info is not None:
        for c in cols:
            train_data[c] = (train_data[c] - col_info[c]["q2"])/(col_info[c]["q98"] - col_info[c]["q2"])

    train_g = train_data.groupby("customer_ID")

    dtrain, (labels_array, train_id_dict) = get_customer_data(train_customers, train_g, cols=cols, 
                                            train_labels=train_labels, test_mode=False)

    #extracted_data = {"train_data":dtrain, "labels": labels_array, "train_ids":train_id_dict}
    np.save(OUTDIR+"train_data_all.npy", dtrain)
    np.save(OUTDIR+"train_labels_all.npy", labels_array)


def write_test_npy(cols, col_info=None):
    test_data = pd.read_parquet(DATADIR+"test_data.parquet")
    test_customers = test_data.customer_ID
    test_count =  test_customers.value_counts()
    test_customers = test_count.index
    if col_info is not None:
        for c in cols:
            test_data[c] = (test_data[c] - col_info[c]["q2"])/(col_info[c]["q98"] - col_info[c]["q2"])
    
    test_g = test_data.groupby("customer_ID")

    dtest, (labels_array, test_id_dict) = get_customer_data(test_customers, test_g, cols=cols, 
                                            train_labels=None, test_mode=True)

    np.save(OUTDIR+"test_data_all.npy", dtest)
    with open(OUTDIR+'test_customers_id_dict.json', 'w') as fp:
        json.dump(test_id_dict, fp)



if __name__ == "__main__":
    my_cols = [col for col in ContCols if col not in MostNaNCols]
    write_train_npy(my_cols)
    write_test_npy(my_cols)