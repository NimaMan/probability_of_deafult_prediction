#%%
import json
import numpy as np
import pandas as pd
import torch 
import torch.nn 
from torch.utils.data import Dataset, DataLoader
from pd.nn.model import MLP, Conv
from pd.nn.train_utils import train

from pd.metric import amex_metric
from pd.params import *



test_customers = pd.read_parquet(DATADIR+"test_data.parquet", columns=["customer_ID"])
test_data = np.load(OUTDIR+"test_data_c13.npy")

with open(OUTDIR+'test_c13_id_dict.json', 'r') as f:
        test_id_dict = json.load(f)

model = Conv()
model_param = torch.load(OUTDIR+"Conv")
model.load_state_dict(model_param)
model.eval()

pred =  model(torch.as_tensor(test_data, dtype=torch.float32))

results =  np.zeros((1, test_data.shape[0]))
test_customer_ids = test_customers.unique()

result = pd.DataFrame({"customer_ID":test_customer_ids, "prediction":pred.detach().numpy().reshape(-1)})

result.set_index("customer_ID").to_csv("sub.csv")

# %%
