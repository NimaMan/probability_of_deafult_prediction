
import json
import scipy as sp
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder



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


def get_col_info(train_data=None, col_info_name="col_info", c13=False, out_dir=None):
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

    with open(out_dir+f"{col_info_name}.pkl", "wb") as f:
        pickle.dump(col_info, f)

    return col_info



if __name__ == "__main__":
    
    get_col_info(train_data=None, col_info_name="col_info", c13=True)