#%%
import json
import numpy as np
import pandas as pd
from pd.data.data_manip import write_train_npy, write_test_npy
from pd.data.preprop import preprocess_data, get_col_info
from pd.params import *


if __name__ == "__main__":
    #my_cols = [col for col in ContCols if col not in MostNaNCols]
    #my_cols = ManCols
    #from pd.data.data_manip import get_col_info
    #col_info = get_col_info() 

    #my_cols = []
    #for c in col_info.keys():
    #    if col_info[c]["max_prob_mass"] < 90:
    #        my_cols.append(c)

    #write_train_npy(my_cols)
    #write_test_npy(my_cols)
    
    #get_col_info(train_data=None, col_info_name="col_info", c13=True)
    #preprocess_data(data_type="train", time_dim=12)
    preprocess_data(data_type="train", time_dim=None, fillna="mean", borders=("q1", "q99"))
    #preprocess_data(data_type="test")