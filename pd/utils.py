import os 
import pickle 
import gzip 
from pd.params import *


def write_log(log, log_desc="log", out_dir=None):
    log_file_name = f"{log_desc}.txt"
    os.makedirs(out_dir, exist_ok=True)
    log_file_dir = os.path.join(out_dir, log_file_name)
    with open(log_file_dir, "a") as f:
        f.write(log)
        f.write("\n")


def wirte_data_pickle(data, name, outdir=None):

    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
    file_dir = os.path.join(outdir, name)
    with gzip.open(file_dir, "wb") as f:
        pickle.dump(data, f)


def load_pickle_data(path):
    with gzip.open(path, "rb") as f:
        data = pickle.load(f)

    return data


def get_customers_data_indices(num_data_points=[13], id_dir='train_agg1_mean_q5_q95_q5_q95_id.json'):
    import json 
    with open(OUTDIR+id_dir, 'r') as f:
            train_id_dict = json.load(f)
    train_id_dict = {val:key for key, val in train_id_dict.items()}
    train_customers_count = pd.read_parquet(TRAINDATA).customer_ID.value_counts().to_dict()
    indices = [int(i) for c, i in train_id_dict.items() if train_customers_count[c] in num_data_points]

    return indices


def get_customers_id_from_indices(indices, id_dir='train_agg1_mean_q5_q95_q5_q95_id.json'):
    import json 
    with open(OUTDIR+id_dir, 'r') as f:
            train_id_dict = json.load(f)
    customers = [train_id_dict[str(idx)] for idx in indices]
    
    return customers


def merge_with_pred(y_pred, y_indices, model_name, type="train", id_dir='train_agg1_mean_q5_q95_q5_q95_id.json'):
    if type == "train":
        pred_dir = os.path.join(PREDDIR, "train_pred.csv")
    else:
        pred_dir = os.path.join(PREDDIR, "test_pred.csv")

    pred_file = pd.read_csv(pred_dir, index_col=0)
    
    customers = get_customers_id_from_indices(y_indices, id_dir=id_dir)
    result = pd.DataFrame({"customer_ID": customers, 
                        model_name: y_pred.reshape(-1)
                        }
                        )
    
    pred_file = pred_file.merge(result, how='left', on='customer_ID')
    pred_file.set_index("customer_ID").to_csv(pred_dir)