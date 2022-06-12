#%%
import numpy as np
import pandas as pd
import torch 
import torch.nn 
from torch.utils.data import Dataset, DataLoader
from pd.nn.model import MLP, Conv
from pd.nn.train_utils import train

from pd.metric import amex_metric
from pd.params import *
    

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

    train_data = np.load(OUTDIR+"train_data_c13.npy")
    train_labels = np.load(OUTDIR+"train_labels_c13.npy")
    train_dataset = CustomerData(train_data, train_labels=train_labels)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

    model = Conv()

    model = train(model, train_loader, num_epochs=15)
    torch.save(model.state_dict(), OUTDIR+model._get_name())
    pred = model(torch.as_tensor(train_data, dtype=torch.float32))
    m =  amex_metric(train_labels, pred.detach().numpy())
    print("performance", m)

# %%
