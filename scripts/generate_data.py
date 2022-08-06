#%%
import json
import numpy as np
import pandas as pd
from pd.data.data_manip import write_train_npy, write_test_npy
from pd.data.preprop import preprocess_data, get_feat_comb
    
from pd.params import *


if __name__ == "__main__":
    
    #get_col_info(train_data=None, col_info_name="col_info", c13=True)
    #preprocess_data(data_type="train", time_dim=12)
    #preprocess_data(data_type="train", time_dim=None, all_data=True, fillna="mean_q5_q95", borders=("q5", "q95"))
    #preprocess_data(data_type="test", time_dim=None, all_data=True, fillna="mean_q5_q95", borders=("q5", "q95"))
    get_feat_comb(data_type="train", agg=4, normalizer="logistic", time_dim=13, fillna="mean_q5_q95", borders=("q5", "q95"), )
    #get_feat_comb(data_type="test", agg=1, normalizer="logistic", time_dim=13, fillna="mean_q5_q95", borders=("q5", "q95"), )
                     