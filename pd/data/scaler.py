import numpy as np
from sklearn import preprocessing
from pd.params import *


def get_scaler(x_train, normalizer="standard"):
    if normalizer == "standard":
        return preprocessing.StandardScaler().fit(x_train)
    elif normalizer == "min-max":
        return preprocessing.MinMaxScaler().fit(x_train)


def scaler_transform(d, c, borders):
    
    return (d - col_info13[c][borders[0]])/(col_info13[c][borders[1]] - col_info13[c][borders[0]])


def logistic_transform(d, c, borders):
    mid_point = np.mean([col_info13[c]["q5"], col_info13[c]["q95"]])
    slope = 2.944/(col_info13[c]["q95"] - mid_point)
    
    return 1/(1 + np.exp(slope*(mid_point - d)))


def transform(d, c, borders, type="logistic"):
    if type == "scaler":
        return scaler_transform(d, c, borders)
    elif type == "logistic":
        return logistic_transform(d, c, borders)
