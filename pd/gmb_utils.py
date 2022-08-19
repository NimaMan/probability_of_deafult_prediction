
import os
import pathlib
import numpy as np
import pandas as pd 
import json
import gc
import random
import joblib
from sklearn.model_selection import StratifiedKFold, train_test_split
from scipy.misc import derivative
import lightgbm as lgb
import pd.metric as metric
from pd.params import *
from pd.utils import get_pred_data


def amex_metric(y_true, y_pred, return_components=False):
    labels = np.transpose(np.array([y_true, y_pred]))
    labels = labels[labels[:, 1].argsort()[::-1]]
    weights = np.where(labels[:,0]==0, 20, 1)
    cut_vals = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four = np.sum(cut_vals[:,0]) / np.sum(labels[:,0])
    gini = [0,0]
    for i in [1,0]:
        labels = np.transpose(np.array([y_true, y_pred]))
        labels = labels[labels[:, i].argsort()[::-1]]
        weight = np.where(labels[:,0]==0, 20, 1)
        weight_random = np.cumsum(weight / np.sum(weight))
        total_pos = np.sum(labels[:, 0] *  weight)
        cum_pos_found = np.cumsum(labels[:, 0] * weight)
        lorentz = cum_pos_found / total_pos
        gini[i] = np.sum((lorentz - weight_random) * weight)
    if return_components:
        return 0.5 * (gini[1]/gini[0] + top_four), gini[1]/gini[0], top_four

    return 0.5 * (gini[1]/gini[0] + top_four)


def lgb_amex_metric(y_pred, y_true):
    y_true = y_true.get_label()
    score, gini, recall = amex_metric(y_true, y_pred, return_components=True)
    return f'amex_metric gini {gini:.3f} recall {recall:.3f}', score, True


def xgb_amex(y_pred, y_true):
    y_true = y_true.get_label()
    score, gini, recall = amex_metric(y_true, y_pred, return_components=True)
    
    #return f'gini_{int(gini*1e7)}_recall_{int(recall*1e7)}', score
    return f'amex', score

def train_lgbm(data, labels, params, feature=None, tempdir=None, n_folds=5, seed=42):
    
    oof_predictions = np.zeros(len(data))
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (trn_ind, val_ind) in enumerate(kfold.split(data, labels)):
        print(' ')
        print('-'*50)
        print(f'Training fold {fold} of {feature} feature...')
        x_train, x_val = data[trn_ind], data[val_ind]
        y_train, y_val = labels[trn_ind], labels[val_ind]
        lgb_train = lgb.Dataset(x_train.reshape(x_train.shape[0], -1), y_train, categorical_feature="auto")
        lgb_valid = lgb.Dataset(x_val.reshape(x_val.shape[0], -1), y_val, categorical_feature="auto")

        model = lgb.train(
            params = params,
            train_set = lgb_train,
            num_boost_round = 1000,
            valid_sets = [lgb_train, lgb_valid],
            early_stopping_rounds = 100,
            verbose_eval = 100,
            feval = lgb_amex_metric
            )
        # Save best model
        model_dir = os.path.join(OUTDIR, "Models")
        model_name =  f'{n_folds}fold_seed{seed}_{feature}.pkl'
        pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True) 
        joblib.dump(model, os.path.join(model_dir, model_name))
        val_pred = model.predict(x_val) # Predict validation
        oof_predictions[val_ind] = val_pred  # Add to out of folds array
        # Compute fold metric
        score, gini, recall = metric.amex_metric(y_val.reshape(-1, ), val_pred, return_components=True)
        print(f'Our fold {fold} CV score is {score}')
        del x_train, x_val, y_train, y_val, lgb_train, lgb_valid
        gc.collect()
    score, gini, recall = metric.amex_metric(labels.reshape(-1, ), oof_predictions, return_components=True)  # Compute out of folds metric
    print(f'Our out of folds CV score is {score}')
    # Create a dataframe to store out of folds predictions
    oof_dir = os.path.join(tempdir, f'{n_folds}fold_seed{seed}_{feature}.npy')
    np.save(oof_dir, oof_predictions)
    
    return (score, gini, recall)


def get_agg_data(data_dir="train_agg_mean_q5_q95_q5_q95.npz",  pred_feat=False, agg=1):
    d = np.load(OUTDIR+data_dir)
    #train_data = np.concatenate((d["d2"].astype(np.int32), d["d1"].reshape(d["d1"].shape[0], -1)), axis=1)
    train_labels = d["labels"]
    df2 = pd.DataFrame(d["d2"].astype(np.int32))
    df = pd.DataFrame(d["d1"].reshape(d["d1"].shape[0], -1))
    df = pd.concat((df2, df), axis=1,)
    df.columns = [f"c{i}" for i in range(df.shape[1])]
    cat_indices = list(np.arange(33))
    if pred_feat:
        pred_data = get_pred_data(type=data_dir.split("_")[0], id_dir=f'{data_dir.split(".")[0]}_id.json', agg=agg)
        pred_cols = [f"p{i}" for i in range(pred_data.shape[1])]
        df[pred_cols] = pred_data
    return df, train_labels, cat_indices


def focal_loss_lgb(y_pred, dtrain, alpha=0.03, gamma=2):
    a,g = alpha, gamma
    y_true = dtrain.label
    def fl(x,t):
        p = 1/(1+np.exp(-x))
        return -( a*t + (1-a)*(1-t) ) * (( 1 - ( t*p + (1-t)*(1-p)) )**g) * ( t*np.log(p)+(1-t)*np.log(1-p) )
    partial_fl = lambda x: fl(x, y_true)
    grad = derivative(partial_fl, y_pred, n=1, dx=1e-6)
    hess = derivative(partial_fl, y_pred, n=2, dx=1e-6)
    return grad, hess


def focal_loss_lgb_eval_error(y_pred, dtrain, alpha=0.03, gamma=2):
    a,g = alpha, gamma
    y_true = dtrain.label
    p = 1/(1+np.exp(-y_pred))
    loss = -( a*y_true + (1-a)*(1-y_true) ) * (( 1 - ( y_true*p + (1-y_true)*(1-p)) )**g) * ( y_true*np.log(p)+(1-y_true)*np.log(1-p) )
    return 'focal_loss', np.mean(loss), False 


def focal_loss_xgb(pred,dtrain,gamma_indct=1.2):
    # retrieve data from dtrain matrix
    label = dtrain.get_label()
    # compute the prediction with sigmoid
    sigmoid_pred = 1.0 / (1.0 + np.exp(-pred))
    # gradient
    # complex gradient with different parts
    grad_first_part = (label+((-1)**label)*sigmoid_pred)**gamma_indct
    grad_second_part = label - sigmoid_pred
    grad_third_part = gamma_indct*(1-label-sigmoid_pred)
    grad_log_part = np.log(1-label-((-1)**label)*sigmoid_pred + 1e-7)       # add a small number to avoid numerical instability
    # combine the gradient
    grad = -grad_first_part*(grad_second_part+grad_third_part*grad_log_part)
    # combine the gradient parts to get hessian
    hess_first_term = gamma_indct*(label+((-1)**label)*sigmoid_pred)**(gamma_indct-1)*sigmoid_pred*(1.0 - sigmoid_pred)*(grad_second_part+grad_third_part*grad_log_part)
    hess_second_term = (-sigmoid_pred*(1.0 - sigmoid_pred)-gamma_indct*sigmoid_pred*(1.0 - sigmoid_pred)*grad_log_part-((1/(1-label-((-1)**label)*sigmoid_pred))*sigmoid_pred*(1.0 - sigmoid_pred)))*grad_first_part
    # get the final 2nd order derivative
    hess = -(hess_first_term+hess_second_term)
    
    return grad, hess


class FocalLossObjective(object):
    def calc_ders_range(self, approxes, targets, weights):
        # approxes, targets, weights are indexed containers of floats
        # (containers with only __len__ and __getitem__ defined).
        # weights parameter can be None.
        # Returns list of pairs (der1, der2)
        gamma = 2.
        # alpha = 1.
        assert len(approxes) == len(targets)
        if weights is not None:
            assert len(weights) == len(approxes)
        
        exponents = []
        for index in xrange(len(approxes)):
            exponents.append(math.exp(approxes[index]))

        result = []
        for index in xrange(len(targets)):
            p = exponents[index] / (1 + exponents[index])

            if targets[index] > 0.0:
                der1 = -((1-p)**(gamma-1))*(gamma * math.log(p) * p + p - 1)/p
                der2 = gamma*((1-p)**gamma)*((gamma*p-1)*math.log(p)+2*(p-1))
            else:
                der1 = (p**(gamma-1)) * (gamma * math.log(1 - p) - p)/(1 - p)
                der2 = p**(gamma-2)*((p*(2*gamma*(p-1)-p))/(p-1)**2 + (gamma-1)*gamma*math.log(1 - p))

            if weights is not None:
                der1 *= weights[index]
                der2 *= weights[index]

            result.append((der1, der2))

        return result