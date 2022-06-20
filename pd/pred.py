
import os
import numpy as np
import pandas as pd
import torch 
import json

from pd.params import *
from pd.nn.model import Conv


def pred13(model):
    test_customers = pd.read_parquet(DATADIR+"test_data.parquet", columns=["customer_ID"])
    test_customer_ids = test_customers.customer_ID.unique()

    test_data = np.load(OUTDIR+"test_data_c13.npy")

    with open(OUTDIR+'test_c13_id_dict.json', 'r') as f:
            test_id_dict = json.load(f)

    #model = Conv()
    #model_param = torch.load(OUTDIR+"Conv")
    #model.load_state_dict(model_param)
    model.eval()

    pred =  model(torch.as_tensor(test_data, dtype=torch.float32))

    result = pd.DataFrame({"customer_ID":test_id_dict.values(), 
                        "prediction":pred.detach().numpy().reshape(-1)
                        }
                        )

    not13_customers = set(test_customer_ids) - set(test_id_dict.values())
    not13_prediction = pd.DataFrame({"customer_ID":list(not13_customers), "prediction":np.zeros(len(not13_customers))})
    result = result.append(not13_prediction)
    result.set_index("customer_ID").to_csv(OUTDIR+"sub.csv")


def pred_test_npy(model=None, model_name=""):
    test_data = np.load(OUTDIR+"test_data_all.npy")
    with open(OUTDIR+'test_customers_id_dict.json', 'r') as f:
            test_id_dict = json.load(f)

    if model is None:
        model = Conv()
        model_param = torch.load(OUTDIR+model_name)
        model.load_state_dict(model_param)
        
    model.eval()
    pred =  model(torch.as_tensor(test_data, dtype=torch.float32))
    result = pd.DataFrame({"customer_ID":test_id_dict.values(), 
                        "prediction":pred.detach().numpy().reshape(-1)
                        }
                        )
    sub_file_dir = os.path.join(OUTDIR, model_name + "sub.csv")
    result.set_index("customer_ID").to_csv(sub_file_dir)
