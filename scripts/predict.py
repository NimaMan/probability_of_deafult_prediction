
import click
import os
import numpy as np
import pandas as pd
import torch 
import json
from torch.utils.data import Dataset, DataLoader

from pd.params import *
from pd.nn.model import Conv


class Data(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):        
        return self.data[index], index

        
@click.command()
@click.option("--model_name", default="conv13_32_all")
@click.option("--test_data", default=None)
def run_experiment(model_name, test_data=None):

    if test_data is None:
            test_data = np.load(OUTDIR+"test_raw_all_data.npy")
    with open(OUTDIR+'test_raw_all_id.json', 'r') as f:
            test_id_dict = json.load(f)

    model = Conv()
    model_param = torch.load(OUTDIR+model_name)
    model.load_state_dict(model_param)
    
    print("start prediction.....")
    test_data = torch.from_numpy(test_data)
    dataset = Data(test_data)
    loader = DataLoader(dataset, batch_size=5000)
    model.eval()
    pred = torch.zeros((test_data.shape[0], 1))
    for idx, (feat, indices) in enumerate(loader):
        batch_pred = model(feat)
        pred[indices] = batch_pred
    
    result = pd.DataFrame({"customer_ID":test_id_dict.values(), 
                        "prediction":pred.detach().numpy().reshape(-1)
                        }
                        )
    sub_file_dir = os.path.join(OUTDIR, model_name + "sub.csv")
    result.set_index("customer_ID").to_csv(sub_file_dir)

if __name__ == "__main__":
    run_experiment()