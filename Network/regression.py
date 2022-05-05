import os
import itertools
import numpy as np
from tqdm import tqdm

import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error

RANDOM_STATE = 123


def XGBoost(X_train, X_test, y_train, y_test, params):
    """
    Perform XGBoost regression with best performing parameters. Calculate Normalized dMSE
    :param: Training and testing data. Params are the best parameters from Cross validation
    :return: None
    """
    Dmatrix = xgb.DMatrix(data=X_train, label=y_train)
    params = {"objective": 'reg:squarederror',
              "colsample_bytree": params[0],
              "learning_rate": params[1],
              "max_depth": params[2],
              "alpha": params[3]}
    xgb_reg = xgb.train(params=params, dtrain=Dmatrix)

    Dmatrix_test = xgb.DMatrix(data=X_test)
    preds = xgb_reg.predict(Dmatrix_test)

    MSE = mean_squared_error(preds, y_test, squared=True)
    Norm_MSE = MSE / np.var(X_test)

    print('Normalized test MSE: {0:0.4f}'.format(Norm_MSE))


def XGBoost_CV(X_train, y_train):
    """
    Perform cross validation on a XGBoost regressor
    :param: training data
    :return: best performing parameters
    """
    def CV(col, lr, depth, alpha):
        params = {"objective": 'reg:squarederror',
                  "colsample_bytree": col,
                  "learning_rate": lr,
                  "max_depth": depth,
                  "alpha": alpha}

        cv_results = xgb.cv(dtrain=Dmatrix, params=params, nfold=10,
                            num_boost_round=30, early_stopping_rounds=10,
                            metrics='rmse', as_pandas=True, seed=RANDOM_STATE)
        return np.sqrt(cv_results['test-rmse-mean'].min())

    Dmatrix = xgb.DMatrix(data=X_train, label=y_train)
    max_depth = np.arange(1, 5)
    alpha = np.arange(0.1, 0.6, 0.1)
    learning_rate = np.arange(0.1, 0.6, 0.1)
    col = np.arange(0.1, 0.6, 0.1)
    permutation = [[p[0], p[1], p[2], p[3]] for p in itertools.product(col, learning_rate, max_depth, alpha)]
    results = []
    for col, lr, depth, alpha in tqdm(permutation):
        results.append([CV(col, lr, depth, alpha), col, lr, depth, alpha])

    best_results = sorted(results, key=lambda x: x[0])
    return best_results[1:]


def RidgeRegression(X_train, X_test, y_train, y_test):
    """
    Linear least squares with l2 regularization.
    :return: None
    """
    reg = Ridge(random_state=RANDOM_STATE, max_iter=1000)
    params = {'alpha': np.arange(0.5, 2, 0.5)}
    reg_tuning = GridSearchCV(reg, params, cv=5, scoring='neg_mean_squared_error')
    reg_tuning.fit(X_train, y_train)

    preds = reg_tuning.best_estimator_.predict(X_test)
    MSE = mean_squared_error(preds, y_test, squared=True)
    Norm_MSE = MSE / np.var(X_test)

    print('Normalized test MSE: {0:0.4f}'.format(Norm_MSE))
    print('Best parameters: {}'.format(reg_tuning.best_params_))
