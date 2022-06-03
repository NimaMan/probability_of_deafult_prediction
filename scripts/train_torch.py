
#%%

import numpy as np
import pandas as pd
from sympy import im 
import torch 
import torch.nn 
from pd.nn.model import MLP

from pd.metric import amex_metric
from pd.data.loader import CustomerData, DataLoader
from pd.params import *



if __name__ == "__main__":
    train_data = pd.read_parquet(data_dir+"train_data.parquet")
    train_labels = pd.read_csv(data_dir+"train_labels.csv", engine="pyarrow")

    # data prep 
    train_data[["D_63", "D_64"]] = train_data[["D_63", "D_64"]].astype("category").apply(lambda x: x.cat.codes)
    #cat_cols = ["D_63", "D_64"]
    #train_data = train_data.dropna(how="any", axis=1) 

    train_dataset = CustomerData(train_data, train_labels=train_labels.set_index("customer_ID"),
                    cat_cols=cat_cols)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
     
    model = MLP(input_dim=177, hidden_dim=[128])
    
    optimizer = torch.optim.Adam(model.parameters(),)
    criterion = torch.nn.BCELoss()

    for epoch in range(5): 
        for (cont_feat, cat_feat), clabel in train_loader:
            #feat = torch.cat((cont_feat, cat_feat), dim=-1)
            feat = cont_feat

            pred = model(feat)
            loss = criterion(pred, clabel)
            
            # Update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(epoch, loss.item(), amex_metric(clabel.detach().numpy(), pred.detach().numpy()))




    train_performance_loader = DataLoader(train_dataset, batch_size=train_data.shape[0])
    for cont_feat, cat_feat, customer_index in train_performance_loader:
        pred = model(cont_feat)

    m =  amex_metric(train_labels.target.values, pred.detach().numpy())

    print("performance", m)

# %%
