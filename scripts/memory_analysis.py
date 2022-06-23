
import torch 
import pandas as pd 
from pd.params import DATADIR, CATCOLS
from memory_profiler import profile


@profile
def load_data(train=True, scaler=None):
    if train:
        data = pd.read_parquet(DATADIR+"train_data.parquet")
        labels = pd.read_csv(DATADIR+"train_labels.csv")

    else:
        data = pd.read_parquet(DATADIR+"test_data.parquet")
        labels = None
    
    cont_cols = [col for col in data.columns.to_list() if col not in CATCOLS + ["customer_ID", "S_2", "target"]]
    # data prep 
    #train_data[["D_63", "D_64"]] = train_data[["D_63", "D_64"]].astype("category").apply(lambda x: x.cat.codes)
    #cat_cols = ["D_63", "D_64"]
    #train_data = train_data.dropna(how="any", axis=1) 
    ## transform the cont cols 
    if False:
        scaler = get_scaler(data[cont_cols].values)
        cont_data = scaler.transform(data[cont_cols].values)
    # deal with na data
    customer_data = data.groupby("customer_ID").mean().fillna(0)
    if not train:
        labels = customer_data.index.values
    cont_data = customer_data[cont_cols].values
    cont_data = torch.as_tensor(cont_data, dtype=torch.float32)
    cat_data = customer_data[CATCOLS].values        
    cat_data = torch.as_tensor(cat_data, dtype=torch.float32).mean(dim=0)
    feat = (cont_data, cat_data)
        
    return feat, labels

if __name__ == '__main__':
    load_data()