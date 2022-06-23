#%%
import json
import numpy as np
import pandas as pd
import torch.nn 
from pd.data.data_manip import write_train_npy, write_test_npy
from pd.params import *


if __name__ == "__main__":
    my_cols = [col for col in ContCols if col not in MostNaNCols]
    my_cols = ManCols
    write_train_npy(my_cols)
    write_test_npy(my_cols)