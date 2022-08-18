#%%
import click
import gc
import os
import joblib
import torch 
import numpy as np
import pandas as pd
import json
from torch.utils.data import Dataset, DataLoader

from pd.params import *
from pd.nn.conv import Conv, ConvPred
from pd.utils import merge_with_pred, get_customers_data_indices, get_pred_data_df
from pd.gmb_utils import get_agg_data

#from memory_profiler import profile



class Data(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):        
        return torch.from_numpy(self.data[index]), index

@click.command()
@click.option("--model_name", default="conv13_32_all")
@click.option("--test_data", default=None)
def run_experiment(model_name, test_data=None):

    if test_data is None:
            test_data = np.load(OUTDIR+"test_raw_all_data.npy", )
    with open(OUTDIR+'test_customers_id_dict.json', 'r') as f:
            test_id_dict = json.load(f)

    model = Conv()
    model_param = torch.load(MODELDIR+model_name)
    model.load_state_dict(model_param)
    
    print("start prediction.....")
    #test_data = torch.from_numpy(test_data)
    dataset = Data(test_data)
    loader = DataLoader(dataset, batch_size=20000)
    model.eval()
    pred = np.zeros(test_data.shape[0])
    #pred = model(test_data)
    for idx, (feat, indices) in enumerate(loader):
        batch_pred = model(feat)
        pred[indices] = batch_pred.detach().numpy().reshape(-1)
        print(idx)
    
    print("Writing down results.. ")
    result = pd.DataFrame({"customer_ID":test_id_dict.values(), 
                        "prediction":pred
                        }
                        )
    sub_file_dir = os.path.join(OUTDIR, model_name + "sub.csv")
    result.set_index("customer_ID").to_csv(sub_file_dir)



def test_catb(model_names):

    n_folds = len(model_names)
    
    for model_name, mn_list in model_names.items():
        test_predictions = np.zeros(924621)
        agg = int(model_name[-1])   
        test_data_name =  f"test_agg{agg}_mean_q5_q95_q5_q95"
        test_data_dir = f"{test_data_name}.npz"
        test_data, labels, cat_indices = get_agg_data(data_dir=test_data_dir)

        for md in mn_list:
            model = joblib.load(os.path.join(MODELDIR, md))
        
            test_pred = model.predict_proba(test_data)[:, 1]
            test_predictions += test_pred / n_folds    
        
        del test_data
        gc.collect()

        with open(OUTDIR+f'{test_data_name}_id.json', 'r') as f:
            test_id_dict = json.load(f)

        result = pd.DataFrame({"customer_ID":test_id_dict.values(), 
                            model_name:test_predictions.reshape(-1)
                            }
                            )

        sub_file_dir = os.path.join(OUTDIR, f"{model_name}.csv")
        result.set_index("customer_ID").to_csv(sub_file_dir)
        
        merge_with_pred(test_predictions, np.arange(len(test_predictions)),
                        model_name=model_name, type="test", id_dir=f'{test_data_name}_id.json')
        

def predict_catb_models(cat_models):
    mds = {}
    for model in cat_models:
        mds[model] = []
        for model_name in all_models:
            if model in model_name:
                mds[model].append(model_name)
    
    test_catb(mds)



if __name__ == "__main__":
    all_models = os.listdir(MODELDIR)
    cat_models = ["catb13_agg4", "catb13_agg1", "catb13_agg2"]
    predict_catb_models(cat_models)


    
