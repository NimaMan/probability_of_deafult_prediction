from sklearn import preprocessing


def get_normalizer(x_train, normalizer="standard"):
    if normalizer == "standard":
        return preprocessing.StandardScaler().fit(x_train)
    elif normalizer == "min-max":
        return preprocessing.MinMaxScaler().fit(x_train)
