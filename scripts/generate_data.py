#%%
import json
import click
import numpy as np
import pandas as pd
from pd.data.data_manip import write_train_npy, write_test_npy
from pd.data.preprop import preprocess_data, get_feat_comb
from pd.params import *

import gc
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from pd.params import *


def get_difference(data, num_features):
    df1 = []
    customer_ids = []
    for customer_id, df in tqdm(data.groupby(['customer_ID'])):
        # Get the differences
        diff_df1 = df[num_features].diff(1).iloc[[-1]].values.astype(np.float32)
        # Append to lists
        df1.append(diff_df1)
        customer_ids.append(customer_id)
    # Concatenate
    df1 = np.concatenate(df1, axis = 0)
    # Transform to dataframe
    df1 = pd.DataFrame(df1, columns = [col + '_diff1' for col in df[num_features].columns])
    # Add customer id
    df1['customer_ID'] = customer_ids
    return df1


def read_preprocess_data(lag=False):
    train = pd.read_parquet(TRAINDATA)
    features = train.drop(['customer_ID', 'S_2'], axis = 1).columns.to_list()
    num_features = [col for col in features if col not in CATCOLS]
    print('Starting training feature engineer...')
    train_num_agg = train.groupby("customer_ID")[num_features].agg(['mean', 'std', 'min', 'max', "first", 'last'])
    train_num_agg.columns = ['_'.join(x) for x in train_num_agg.columns]
    train_num_agg.reset_index(inplace = True)
    train_cat_agg = train.groupby("customer_ID")[CATCOLS].agg(['count', "first", 'last', 'nunique'])
    train_cat_agg.columns = ['_'.join(x) for x in train_cat_agg.columns]
    train_cat_agg.reset_index(inplace = True)
    train_labels = pd.read_csv(TRAINLABELS)
    # Transform float64 columns to float32

    if lag:
        for col in train_num_agg:
            if 'last' in col and col.replace('last', 'first') in train_num_agg:
                train_num_agg[col + '_lag_sub'] = train_num_agg[col] - train_num_agg[col.replace('last', 'first')]
                train_num_agg[col + '_lag_div'] = train_num_agg[col] / train_num_agg[col.replace('last', 'first')]

        cols = list(train_num_agg.dtypes[train_num_agg.dtypes == 'float64'].index)
    for col in tqdm(cols):
        train_num_agg[col] = train_num_agg[col].astype(np.float32)
    # Transform int64 columns to int32
    cols = list(train_cat_agg.dtypes[train_cat_agg.dtypes == 'int64'].index)
    for col in tqdm(cols):
        train_cat_agg[col] = train_cat_agg[col].astype(np.int32)
    # Get the difference
    train_diff = get_difference(train, num_features)
    train = train_num_agg.merge(train_cat_agg, how = 'inner', on = 'customer_ID').merge(train_diff, how = 'inner', on = 'customer_ID').merge(train_labels, how = 'inner', on = 'customer_ID')
    if lag:
        train.to_parquet(OUTDIR + "train_k7977_lag.parquet")
    else:
        train.to_parquet(OUTDIR + "train_k7977.parquet")
    
    del train_num_agg, train_cat_agg, train_diff, train
    gc.collect()
    
    test = pd.read_parquet(TESTDATA)
    print('Starting test feature engineer...')
    test_num_agg = test.groupby("customer_ID")[num_features].agg(['mean', 'std', 'min', 'max', "first", 'last'])
    test_num_agg.columns = ['_'.join(x) for x in test_num_agg.columns]
    test_num_agg.reset_index(inplace = True)
    test_cat_agg = test.groupby("customer_ID")[CATCOLS].agg(['count', "first", 'last', 'nunique'])
    test_cat_agg.columns = ['_'.join(x) for x in test_cat_agg.columns]
    test_cat_agg.reset_index(inplace = True)

    if lag:
        for col in test_num_agg:
            if 'last' in col and col.replace('last', 'first') in test_num_agg:
                test_num_agg[col + '_lag_sub'] = test_num_agg[col] - test_num_agg[col.replace('last', 'first')]
                test_num_agg[col + '_lag_div'] = test_num_agg[col] / test_num_agg[col.replace('last', 'first')]


    # Transform float64 columns to float32
    cols = list(test_num_agg.dtypes[test_num_agg.dtypes == 'float64'].index)
    for col in tqdm(cols):
        test_num_agg[col] = test_num_agg[col].astype(np.float32)
    # Transform int64 columns to int32
    cols = list(test_cat_agg.dtypes[test_cat_agg.dtypes == 'int64'].index)
    for col in tqdm(cols):
        test_cat_agg[col] = test_cat_agg[col].astype(np.int32)
    # Get the difference
    test_diff = get_difference(test, num_features)
    test = test_num_agg.merge(test_cat_agg, how = 'inner', on = 'customer_ID').merge(test_diff, how = 'inner', on = 'customer_ID')
    del test_num_agg, test_cat_agg, test_diff
    gc.collect()
    # Save files to disk
    if lag:
        test.to_parquet(OUTDIR + "test_k7977_lag.parquet")
    else:
        test.to_parquet(OUTDIR + "test_k7977.parquet")
    


@click.command()
@click.option("--agg", default=1)
def generate_data(agg):
    get_feat_comb(data_type="train", agg=agg, normalizer="logistic", time_dim=13, fillna="mean_q5_q95", borders=("q5", "q95"), )
    get_feat_comb(data_type="test", agg=agg, normalizer="logistic", time_dim=13, fillna="mean_q5_q95", borders=("q5", "q95"), )
    

if __name__ == "__main__":
    
    #get_col_info(train_data=None, col_info_name="col_info", c13=True)
    #preprocess_data(data_type="train", time_dim=12)
    #preprocess_data(data_type="train", time_dim=None, all_data=True, fillna="mean_q5_q95", borders=("q5", "q95"))
    #preprocess_data(data_type="test", time_dim=None, all_data=True, fillna="mean_q5_q95", borders=("q5", "q95"))
    #get_feat_comb(data_type="train", agg=1, normalizer="logistic", time_dim=13, fillna="mean_q5_q95", borders=("q5", "q95"), )
    #get_feat_comb(data_type="test", agg=1, normalizer="logistic", time_dim=13, fillna="mean_q5_q95", borders=("q5", "q95"), )
    #generate_data()

    read_preprocess_data(lag=True)
                