#%%
import json
import numpy as np
import pandas as pd
import torch 
import torch.nn 
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from pd.nn.model import MLP, Conv
from pd.nn.train_utils import train_torch_model
from pd.data.data_manip import write_train_npy, write_test_npy
from pd.metric import amex_metric
from pd.params import *
from pd.pred import pred_test_npy as predict

class CustomerData(Dataset):
    def __init__(self, data:np.array, test_mode=False, train_labels=None):
        self.data = data
        self.test_mode = test_mode
        self.train_labels = train_labels
        
    def __len__(self):
        return len(self.train_labels)

    def __getitem__(self, index):        
        feat =  torch.as_tensor(self.data[index], dtype=torch.float32)

        if self.test_mode:
            return feat, index
        else:
            customer_label = torch.as_tensor(self.train_labels[index], dtype=torch.float32)
            return feat, customer_label



if __name__ == "__main__":
    
    my_cols = [col for col in ContCols if col not in MostNaNCols]
    my_cols = ManCols
    #write_train_npy(my_cols)
    #write_test_npy(my_cols)

    model_name = "conv_all_man_feat"
    train_data = np.load(OUTDIR+"train_data_all.npy")
    train_labels = np.load(OUTDIR+"train_labels_all.npy")
    X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=1/9, random_state=0, shuffle=True)

    train_dataset = CustomerData(X_train, train_labels=y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

    validation_data = (X_test, y_test)
    
    model = Conv()
    model = train_torch_model(model, train_loader,  num_epochs=100, validation_data=validation_data, 
                            output_model_name=model_name)

    torch.save(model.state_dict(), OUTDIR+model_name)
    test_pred = model(torch.as_tensor(train_data, dtype=torch.float32))
    m =  amex_metric(train_labels, test_pred.detach().numpy())
    print("performance", m)
    
    del train_data
    del train_dataset
    
    predict(model, model_name=model_name)

# %%
