
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
from pd.data.scaler import get_scaler



if __name__ == "__main__":
    train_data = pd.read_parquet(DATADIR+"train_data.parquet")
    train_labels = pd.read_csv(DATADIR+"train_labels.csv", engine="pyarrow")
    cont_cols = [col for col in train_data.columns.to_list() if col not in CATCOLS + ["customer_ID", "S_2", "target"]]

    cont_cols = ["P_2", "B_1", "B_2", "R_1", "S_3", "D_41", "B_3", 
	"D_42", "D_43", "D_44", "B_4", "D_45", "B_5", "R_2", 
	"D_46", "D_47", "D_48", "D_49", "B_6", "B_7", "B_8", 
	"D_50", "D_51", 'B_9', "R_3", "D_52", "P_3", "B_10",
	"D_53", "S_5", "B_11", "S_6", "R_4", "S_7", "B_12", 
	"S_8", "D_55", "D_56", "D_58", "D_59", "D_60", "D_61",
	"S_11", "D_62", "D_64", "B_16", "B_17", "B_18", "B_19", 
	"B_20", "D_68", "B_22", "D_70", "S_15", "B_23", "D_74", 
	"D_75", "D_76", "D_78", "D_79", "B_28", "B_30", "B_33", 
	"S_22", "S_23", "S_24", "B_37", "R_26", "B_38", "B_40", 
	"D_124", "D_128", "D_129"]
    CATCOLS = [col for col in train_data.columns.to_list() if col not in cont_cols + ["customer_ID", "S_2", "target"]]
    # data prep 
    train_data[["D_63", "D_64"]] = train_data[["D_63", "D_64"]].astype("category").apply(lambda x: x.cat.codes)
    #cat_cols = ["D_63", "D_64"]
    #train_data = train_data.dropna(how="any", axis=1) 
    ## transform the cont cols 
    scaler = get_scaler(train_data[cont_cols].values)
    train_data[cont_cols] = scaler.transform(train_data[cont_cols].values)
    
    # create pytorch dataloader
    train_dataset = CustomerData(train_data, train_labels=train_labels.set_index("customer_ID"),
                    cat_cols=CATCOLS)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
     
    model = MLP(input_dim=73, hidden_dim=[128])
    
    optimizer = torch.optim.Adam(model.parameters(),)
    criterion = torch.nn.BCELoss()

    for epoch in range(15): 
        for (cont_feat, cat_feat), clabel in train_loader:
            #feat = torch.cat((cont_feat, cat_feat), dim=-1)
            feat = cont_feat

            pred = model(feat)
            #weight = clabel.clone()
            #weight[weight==0] = 4
            #criterion.weight = weight
            loss = criterion(pred, clabel)
            
            # Update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(epoch, loss.item(), amex_metric(clabel.detach().numpy(), pred.detach().numpy()))


    train_performance_loader = DataLoader(train_dataset, batch_size=train_data.shape[0])
    for (cont_feat, cat_feat), customer_index in train_performance_loader:
        pred = model(cont_feat)

    m =  amex_metric(train_labels.target.values, pred.detach().numpy())

    print("performance", m)

    test_mode = True
    if test_mode:
        del train_data
        del train_dataset
        del train_performance_loader

        test_data = pd.read_parquet(DATADIR+"test_data.parquet")
        test_data_size = test_data.shape[0]
        test_data[cont_cols] = scaler.transform(test_data[cont_cols].values)
    
        test_dataset = CustomerData(test_data, test_mode=True, cat_cols=CATCOLS)
        test_loader = DataLoader(test_dataset, batch_size=test_data.shape[0])
        
        for (cont_feat, cat_feat), customer_index in test_loader:
            pred = model(cont_feat)
        
        test_customer_ids = test_dataset.customer_ids
        result = pd.DataFrame({"customer_ID":test_customer_ids, "prediction":pred.detach().numpy().reshape(-1)})

        result.set_index("customer_ID").to_csv("sub.csv")

# %%
