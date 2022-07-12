#%%
import gc
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
from bes.nn.es_module import ESModule
import torch 
import torch.nn as nn
import torch.nn.functional as F


class MLP(ESModule):

    def __init__(self, input_dim, hidden_dim=128,):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.nf1 = nn.LayerNorm([hidden_dim])
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.nf2 = nn.LayerNorm([hidden_dim])
        self.fc3 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.nf3 = nn.LayerNorm([hidden_dim])
        self.fc4 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.nf4 = nn.LayerNorm([hidden_dim])
        self.fc5 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.nf5 = nn.LayerNorm([hidden_dim])
        
        self.fcout = nn.Linear(in_features=hidden_dim, out_features=1)
    
    def forward(self, h, return_featues=False):
        h = F.selu(self.fc1(h))
        r = self.nf1(h)
        h = F.selu(self.fc2(r))
        h = self.nf2(h)
        h = F.selu(self.fc3(h))
        r = self.nf3(h+r)
        h = F.selu(self.fc4(r))
        h = self.nf4(h)
        h = F.selu(self.fc5(h))
        h = self.nf5(h+r)
        if return_featues:
            return torch.sigmoid(self.fcout(h)), h
        
        return torch.sigmoid(self.fcout(h))


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


def train_mlp(X, train_labels, model_name="mlp_col27_sum", num_epochs=100):

    X_train, X_test, y_train, y_test = train_test_split(X, train_labels, test_size=1/9, random_state=0, shuffle=True)
    validation_data = (X_test, y_test)

    train_dataset = CustomerData(X_train, train_labels=y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    
    mlp = MLP(input_dim=X.shape[-1])
    mlp = train_torch_model(mlp, train_loader, num_epochs=num_epochs, validation_data=validation_data, 
                                output_model_name=model_name)
    
    return mlp


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

X = train_data.sum(axis=1) 
for idx, c in enumerate(dist_col_27):
    X[:, idx] = X[:, idx]/col_info[c]["max"]

X = X/X.max(axis=0)

del train_data
gc.collect()


mlp = train_mlp(X, train_labels, num_epochs=30)
mlp_pred, mlp_feat =  mlp(torch.as_tensor(X, dtype=torch.float32), return_featues=True)

del X
gc.collect()

model_name = "conv_90_780_18_5"
conv = Conv()
model_param = torch.load(OUTDIR+model_name)
conv.load_state_dict(model_param)
cont_data = np.load(OUTDIR+"train_data_all.npy")
conv_pred, conv_feat =  conv(torch.as_tensor(cont_data, dtype=torch.float32), return_featues=True)

X = torch.cat((mlp_feat, conv_feat, mlp_pred, conv_pred), dim=-1)
del cont_data
del conv_feat
del mlp_feat
gc.collect()
np.save(OUTDIR+"agg_feat.npy", X.detach().numpy())
#train_mlp(X, train_labels, model_name="mlp_agg")

#%%