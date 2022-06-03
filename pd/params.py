import pandas as pd

# dirs
data_dir = "/Users/nimamanaf/Desktop/kaggle/pd/data/"
OUTDIR = "/Users/nimamanaf/Desktop/kaggle/pd/data/out/"
# data
data_temp = pd.read_csv(data_dir+"train_data.csv", nrows=5)

cat_cols = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']
cont_cols = [col for col in data_temp.columns.to_list() if col not in cat_cols + ["customer_ID", "S_2", "target"]]


## Learning 
BATCH_SIZE = 5000
