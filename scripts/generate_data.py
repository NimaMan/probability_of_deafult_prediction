#%%
import json
import numpy as np
import pandas as pd
import torch.nn 
from pd.data.data_manip import write_train_npy, write_test_npy, get_col_info
from pd.params import *


if __name__ == "__main__":
    #my_cols = [col for col in ContCols if col not in MostNaNCols]
    #my_cols = ManCols
    my_cols = []
    for c in col_info.keys():
        if col_info[c]["max_prob_mass"] < 90:
            my_cols.append(c)

    write_train_npy(my_cols)
    write_test_npy(my_cols)