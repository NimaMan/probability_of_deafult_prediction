#%%
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import gc
import json
import numpy as np
import pandas as pd
import torch 
import torch.nn 
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from pd.nn.model import Conv
from pd.data.loader import CustomerData, DTwithLabelRatio
from pd.nn.train_utils import train_torch_model
from pd.metric import amex_metric
from pd.params import *
from pd.pred import pred_test_npy as predict


if __name__ == "__main__":
    
    model_name = "conv_raw_all"
    train_data = np.load(OUTDIR+"train_raw_all_data.npy")
    train_labels = np.load(OUTDIR+"train_raw_all_labels.npy")
    X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=1/9, random_state=0, shuffle=True)
    validation_data = (X_test, y_test)

    train_dataset = CustomerData(X_train, train_labels=y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

    #train_dataset = DTwithLabelRatio(X_train, train_labels=y_train, batch_size=BATCH_SIZE)
    #train_loader = DataLoader(train_dataset, batch_size=1)

    
    model = Conv(input_dim=X_train.shape[-1], conv_channels=25)
    model = train_torch_model(model, train_loader, num_epochs=100, validation_data=validation_data, 
                            output_model_name=model_name)

    torch.save(model.state_dict(), OUTDIR+model_name)
    test_pred = model(torch.as_tensor(train_data, dtype=torch.float32))
    m =  amex_metric(train_labels, test_pred.detach().numpy())
    print("performance", m)
    
    del train_data
    del train_dataset
    gc.collect()
    predict(model, model_name=model_name)

# %%
