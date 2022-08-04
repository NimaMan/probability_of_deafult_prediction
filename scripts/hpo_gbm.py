
import gc
from ntpath import join
import tempfile
import os 
import json
import lightgbm as lgb
import numpy as np
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.lightgbm import TuneReportCheckpointCallback

import pd.metric as metric
from pd.gmb_utils import lgb_amex_metric
from pd.params import *


def train_lgbm(params, tempdir=None, seed=42):
    
    data = np.load(OUTDIR+"train_raw_all_data.npy")
    labels = np.load(OUTDIR+"train_raw_all_labels.npy")            
    
    x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size=0.15)
    lgb_train = lgb.Dataset(x_train.reshape(x_train.shape[0], -1), y_train, categorical_feature="auto")
    lgb_valid = lgb.Dataset(x_val.reshape(x_val.shape[0], -1), y_val, categorical_feature="auto")

    model = lgb.train(
            params = params,
            train_set = lgb_train,
            num_boost_round = 1000,
            valid_sets = [lgb_train, lgb_valid],
            early_stopping_rounds = 100,
            verbose_eval = 100,
            feval = lgb_amex_metric,
            #callbacks=[TuneReportCheckpointCallback({
            #        "binary_error": "eval-binary_error",
            #        "binary_logloss": "eval-binary_logloss",
            #    })],
            )

    val_pred = model.predict(x_val.shape[0], -1) # Predict validation
    score, gini, recall = metric.amex_metric(y_val.reshape(-1, ), val_pred, return_components=True)
    del x_train, x_val, y_train, y_val, lgb_train, lgb_valid
    gc.collect()
    tune.report(amex=score, gini=gini, recall=recall, done=True, binary_error=recall)

    return (score, gini, recall)


if __name__ == "__main__":
    n_workers = 32   
    exp_name = "hpo_lgbm_all"
    ray.init(num_cpus=n_workers, ignore_reinit_error=True)

    params = {
        'objective': 'binary',
        'metric': "binary_logloss",
        "boosting_type": tune.grid_search(["gbdt", "dart"]),
        #'boosting': 'dart',
        'seed': 42,
        'num_leaves': tune.grid_search([10, 100, 1000]),
        'learning_rate': tune.loguniform(1e-8, 1e-1),
        'feature_fraction': 0.20,
        'bagging_freq': 10,
        'bagging_fraction': 0.50,
        'n_jobs': -1,
        'lambda_l2': tune.randint(1, 32),
        'min_data_in_leaf': tune.randint(40, 4000)
        }

    run_info = params
    tempdir = tempfile.mkdtemp(prefix=exp_name, dir=OUTDIR)
    from ray.tune.logger import LoggerCallback
    class CustomLoggerCallback(LoggerCallback):
        """Custom logger interface"""

        def __init__(self,):
            self._trial_files = {}
            self._filename = os.path.join(tempdir, "hpo_lgbm_log.txt")

        def log_trial_start(self, trial: "Trial"):
            trial_logfile = os.path.join(trial.logdir, self._filename)
            self._trial_files[trial] = open(trial_logfile, "at")

        def log_trial_result(self, iteration: int, trial: "Trial", result):
            if trial in self._trial_files:
                self._trial_files[trial].write(json.dumps(result))

        def on_trial_complete(self, iteration: int, trials,
                            trial, **info):
            if trial in self._trial_files:
                self._trial_files[trial].close()
                del self._trial_files[trial]

    analysis = tune.run(
        train_lgbm,
        metric="binary_error",
        mode="min",
        config=params,
        name=exp_name,
        callbacks=[CustomLoggerCallback()],
        num_samples=2,
        scheduler=ASHAScheduler(),
    )
     
    print("Best hyperparameters found were: ", analysis.best_config)

    import pickle
    with open(os.path.join(tempdir, "analysis.pkl", "wb")) as f:
        pickle.dump(f)
    