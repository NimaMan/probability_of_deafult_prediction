#%%
import numpy as np
import pandas as pd 
import torch 
from torch.utils.data import Dataset, DataLoader
from pd.params import *


def agg_cat_cont_feat(cont_feat, cat_feat, agg_type="contOnly"):
    if agg_type == "contOnly":
        return cont_feat
    elif agg_type == "catOnly":
        return cat_feat


class CustomerDataPDDataFrame(Dataset):
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



def load_data_all(train=True, scaler=None):
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


def get_data_rows():
    data_size = 5531451
    chunks = 50
    b = data_size//chunks
    row_indices = [i*b+1 for i in range(chunks)]
    idx = np.random.randint(low=0, high=chunks-1)
    skiprows = range(1, row_indices[idx])
    nrows = b

    return skiprows, nrows
    

def load_full_data_with_cat_cont_feat(train=True, scaler=None):
    if train:
        skiprows, nrows = get_data_rows()
        data = pd.read_csv(DATADIR+"train_data.csv", skiprows=skiprows, nrows=nrows, header=0, engine="c", index_col=0)
        labels = pd.read_csv(DATADIR+"train_labels.csv")
        labels = labels[labels.customer_ID.isin(data.customer_ID.values)]

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


def load_npy_data(batch_size, seed=None):
    if seed is not None:
        np.random.seed(seed=seed)
    d = np.load(OUTDIR+"train_data_all.npy")
    train_labels = np.load(OUTDIR+"train_labels_all.npy")

    batch_indices =  np.random.randint(low=0, high=len(d))

    return d[batch_indices], train_labels[batch_indices]


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


class DTwithLabelRatio(Dataset):
    """
    Currently, we are randomly sampling data from ones and zeros with given ratio. 
    This is not guaranteed to use all the data. The next version may take that into account
    """
    def __init__(self, data:np.array, test_mode=False, train_labels=None, batch_size=2, ones_ratio=0.5):
        self.data = data
        self.train_labels = train_labels
        self.ones_ratio = ones_ratio
        self.batch_size = batch_size
        self.ones_indices = np.argwhere(train_labels==1)[:, 0]
        self.zeros_indices = np.argwhere(train_labels==0)[:, 0]
        
        
    def __len__(self):
        return len(self.train_labels//self.batch_size)

    def __getitem__(self, index): 
        indices = self.smaple_data_with_label_ratio(self.train_labels, self.batch_size, ones_ratio=self.ones_ratio)
        
        feat =  torch.as_tensor(self.data[indices], dtype=torch.float32)
        customer_label = torch.as_tensor(self.train_labels[indices], dtype=torch.float32)
        
        return feat, customer_label

    def smaple_data_with_label_ratio(self, batch_size, ones_ratio=0.5):
        # This will not use all the training data and some data will be used repetitivly 
        ones_smaple_size = int(batch_size*ones_ratio)
        zeros_smaple_size = batch_size - ones_smaple_size
        ones_sample_indices = np.random.randint(low=0, high=len(self.ones_indices), size=ones_smaple_size)
        zeros_sample_indices = np.random.randint(low=0, high=len(self.zeros_indices), size=zeros_smaple_size)

        return np.concatenate((ones_sample_indices, zeros_sample_indices))

    def smaple_data_label_ratio_with_pred_determined_indices(self, batch_size, ones_ratio=0.5):
        ones_smaple_size = int(batch_size*ones_ratio)
        zeros_smaple_size = batch_size - ones_smaple_size
        # Choose a random index for ones (make sure you include all the data)
        
        # Choose a random index from zeros
        

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    train_data = np.load(OUTDIR+"train_data_all.npy")
    train_labels = np.load(OUTDIR+"train_labels_all.npy")
    X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=1/9, random_state=0, shuffle=True)
# %%
