import os 
import pickle 
import gzip 


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