import numpy as np
import pandas as pd
import torch 
import torch.nn 
import matplotlib.pyplot as plt
import warnings
from pd.nn.model import Conv

from pd.metric import amex_metric
from pd.data.loader import CustomerData, DataLoader
from pd.params import *
from pd.pred import pred_test_npy
from sklearn.model_selection import train_test_split
from pd.nn.train_utils import train_torch_model


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
    
    return d, labels_array


train_data = pd.read_parquet(DATADIR+"train_data.parquet")
train_labels = pd.read_csv(DATADIR+"train_labels.csv")
train_labels.set_index("customer_ID", inplace=True)

nzs = []
dist_col_27 = []
for c in col_info.keys():
    nz = np.count_nonzero(col_info[c]["hist"][0])
    nzs.append(nz)
    if nz < 27:
        dist_col_27.append(c)


train_data, train_labels = get_customer_data(train_data.customer_ID.unique(), 
                                        train_data.groupby("customer_ID"), 
                                        cols=dist_col_27, train_labels=train_labels)


train_data = np.nan_to_num(train_data)  ## This maybe a huge asumtions to make
X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=1/9, random_state=0, shuffle=True)
validation_data = (X_test, y_test)

train_dataset = CustomerData(X_train, train_labels=y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)


model_name = "Conv_col27"
model = Conv(input_dim=91)
model = train_torch_model(model, train_loader, num_epochs=150, validation_data=validation_data, 
                            output_model_name=model_name)
