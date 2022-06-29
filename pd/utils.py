import os 
import pickle 
import gzip 


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