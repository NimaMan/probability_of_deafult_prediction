
import gc
import warnings
warnings.filterwarnings('ignore')
import scipy as sp
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import itertools
from pd.params import *


def preprocess_data(data_type="train"):
    
    if data_type == "train":
        data = pd.read_parquet(OUTDIR+"train_data.parquet")
        train_labels = pd.read_csv(DATADIR+"train_labels.csv")
    else:
        data = pd.read_parquet(OUTDIR+"test_data.parquet")
        train_labels = None
    
    print('Starting feature engineer...')
    
    data_cont_agg = data.groupby("customer_ID")[ContCols].agg(['mean', 'std', 'min', 'max', 'last'])
    data_cont_agg.columns = ['_'.join(x) for x in data_cont_agg.columns]
    data_cont_agg.reset_index(inplace=True)

    data_cat_agg = data.groupby("customer_ID")[CATCOLS].agg(['count', 'last', 'nunique'])
    data_cat_agg.columns = ['_'.join(x) for x in data_cat_agg.columns]
    data_cat_agg.reset_index(inplace=True)
    data = data_cont_agg.merge(data_cat_agg, how='inner', on='customer_ID')
    
    if train_labels is None:
        data = data_cont_agg.merge(train_labels, how='inner', on='customer_ID')
    
    del data_cont_agg, data_cont_agg
    gc.collect()
    
    data.to_parquet(OUTDIR+f"{data_type}_fe.parquet")