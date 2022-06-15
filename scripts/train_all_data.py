#%%
import json
import numpy as np
import pandas as pd
import torch 
import torch.nn 
from torch.utils.data import Dataset, DataLoader
from pd.nn.model import MLP, Conv
from pd.data.data_manip import write_train_npy, write_test_npy
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


def eval():
    test_customers = pd.read_parquet(DATADIR+"test_data.parquet", columns=["customer_ID"])
    test_customer_ids = test_customers.customer_ID.unique()

    test_data = np.load(OUTDIR+"test_data_all.npy")

    with open(OUTDIR+'test_cuctomers_id_dict.json', 'r') as f:
            test_id_dict = json.load(f)

    model = Conv()
    model_param = torch.load(OUTDIR+"Conv_all")
    model.load_state_dict(model_param)
    model.eval()
    pred =  model(torch.as_tensor(test_data, dtype=torch.float32))

    result = pd.DataFrame({"customer_ID":test_id_dict.values(), 
                        "prediction":pred.detach().numpy().reshape(-1)
                        }
                        )
    result.set_index("customer_ID").to_csv(OUTDIR+"sub.csv")


if __name__ == "__main__":
    my_cols = [col for col in ContCols if col not in MostNaNCols]
    write_train_npy(my_cols)
    write_test_npy(my_cols)
    model_name = "conv_all"
    train_data = np.load(OUTDIR+"train_data_all.npy")
    train_labels = np.load(OUTDIR+"train_labels_all.npy")
    train_dataset = CustomerData(train_data, train_labels=train_labels)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

    model = Conv()

    model = train(model, train_loader, num_epochs=15)
    torch.save(model.state_dict(), OUTDIR+model_name)
    pred = model(torch.as_tensor(train_data, dtype=torch.float32))
    m =  amex_metric(train_labels, pred.detach().numpy())
    print("performance", m)
    del train_data
    del train_dataset
    eval(model)

# %%