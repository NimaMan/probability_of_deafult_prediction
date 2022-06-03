
import torch 
from torch.utils.data import Dataset, DataLoader


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

