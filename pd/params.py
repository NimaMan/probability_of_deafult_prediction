
import sys 
import pandas as pd

# dirs
DATADIR = None
OUTDIR = None
if sys.platform == "darwin":
    DATADIR = "/Users/nimamanaf/Desktop/kaggle/pd/data/"
    OUTDIR = "/Users/nimamanaf/Desktop/kaggle/pd/data/out/"

# data
CATCOLS = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']


## Learning 
BATCH_SIZE = 15000

## log 
PerfThreshold = 0.75
logBestIndiv = 20