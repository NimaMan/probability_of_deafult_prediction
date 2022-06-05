#%%
import pandas as pd 
import torch 
from torch.utils.data import Dataset, DataLoader
from pd.data.scaler import get_scaler
from pd.params import DATADIR, CATCOLS


def agg_cat_cont_feat(cont_feat, cat_feat, agg_type="contOnly"):
    if agg_type == "contOnly":
        return cont_feat
    elif agg_type == "catOnly":
        return cat_feat


class CustomerData(Dataset):
    def __init__(self, data, cat_cols=[], test_mode=False, train_labels=None):
        self.data = data
        self.test_mode = test_mode
        self.cat_cols = cat_cols
        customer_indices = data["customer_ID"].reset_index().set_index("customer_ID").groupby('customer_ID').apply(lambda x : x.to_numpy().reshape(-1, )).to_dict()
        self.customer_ids =tuple(customer_indices)
        self.customer_indices = tuple(customer_indices.values())
        self.train_labels = train_labels
        self.data_columns = data.columns.to_list()
        self.cont_cols = [col for col in self.data_columns if col not in cat_cols + ["customer_ID", "S_2", "target"]]

    def __len__(self):
        return len(self.customer_indices)

    def __getitem__(self, index):
        customer_data_indices = self.customer_indices[index]
        skiprows = customer_data_indices[0] + 1
        nrows = customer_data_indices[-1] - customer_data_indices[0] + 1
        customer_data = self.data.iloc[skiprows: skiprows+nrows]
        customer_id = customer_data.customer_ID.iloc[0]
        
        #customer_data.drop(["customer_ID", "S_2"], axis=1, inplace=True)
        
        customer_cont_data = customer_data[self.cont_cols].fillna(0, axis=1)
        customer_cont_tensor_data = torch.as_tensor(customer_cont_data.values, dtype=torch.float32)
        cont_feat = customer_cont_tensor_data.mean(dim=0)

        customer_cat_data = customer_data[self.cat_cols].values        
        cat_feat = torch.as_tensor(customer_cat_data, dtype=torch.float32).mean(dim=0)
        feat = (cont_feat, cat_feat)
        if self.test_mode:
            return feat, index
        else:
            customer_label = torch.as_tensor(self.train_labels.loc[customer_id].values, dtype=torch.float32)
            return feat, customer_label



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

if __name__ == "__main__":
    (cont_feat, cat_feat), labels = load_data(train=True,)
# %%
