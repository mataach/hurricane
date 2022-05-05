from tqdm import tqdm
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

RANDOM_STATE = 123


def XGBoostClf_CV(X_train, y_train):
    """
    Perform cross validation for an XGBoost classifier
    :param:Training data
    :return: best performing parameters
    """
    labels = np.argmax(y_train, axis=1)
    Dmatrix = xgb.DMatrix(data=X_train, label=labels)

    def CV(depth, alpha, lr):
        params = {
            'objective': 'multi:softmax',
            'max_depth': depth,
            'alpha': alpha,
            'learning_rate': lr,
            'num_class': 5
        }
        xgb_cv = xgb.cv(params=params, dtrain=Dmatrix, nfold=5,
                        num_boost_round=50, early_stopping_rounds=10,
                        metrics="merror", seed=RANDOM_STATE, as_pandas=True)
        return xgb_cv['test-merror-mean'][0]

    max_depth = np.arange(1, 11)
    alpha = np.arange(1, 11)
    learning_rate = np.arange(0.1, 1.5, 0.1)
    permutation = [[p[0], p[1], p[2]] for p in itertools.product(max_depth, alpha, learning_rate)]
    results = []
    for depth, alpha, lr in tqdm(permutation):
        results.append([CV(depth, alpha, lr), depth, alpha, lr])

    best_results = sorted(results, key=lambda x: x[0])[0]
    return best_results[1:]


def XGBoostClassifier(X_train, X_test, y_train, y_test, params):
    """
    Calculate accuracy score for training and testing data on a XGBoost classifier
    :param X, y: Training and testing data
    :param params: best performing parameters obtained from `XGBoostClf_CV`
    :return: None
    """
    parameters = {
        'objective': 'multi:softmax',
        'max_depth': params[0],
        'alpha': params[1],
        'learning_rate': params[2]
    }
    train_labels, test_labels = np.argmax(y_train, axis=1), np.argmax(y_test, axis=1)
    xgb_clf = xgb.XGBClassifier(**parameters)
    xgb_clf.fit(X_train, train_labels)

    print('XGBoost model training accuracy score: {0:0.4f}'.format(
        accuracy_score(train_labels, xgb_clf.predict(X_train))))
    print('XGBoost model testing accuracy score: {0:0.4f}'.format(
        accuracy_score(test_labels, xgb_clf.predict(X_test))))


def DummyClf(X_train, y_train):
    """
    Dummy classifier that chooses the most frequent label. Used for baseline
    :param: training data
    :return:
    """
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    print('Baseline training score: {}'.format(dummy.score(X_train, y_train)))


def Logistic_Regression(X_train, X_test, y_train, y_test, visualize=False):
    """
    Perform logistic regression classification
    :param X, y: Training and testing data.
    Note that testing data is one-hot -> need to be transformed back into numerical labels
    :return: None
    """
    y_train, y_test = np.argmax(y_train, axis=1), np.argmax(y_test, axis=1)
    lr = LogisticRegression(solver='liblinear', random_state=RANDOM_STATE, max_iter=100)
    params = {'penalty': ['l1', 'l2'], 'C': np.linspace(0.5, 3.5, 3)}
    lr_tuning = GridSearchCV(lr, params, cv=5, scoring='accuracy')
    lr_tuning.fit(X_train, y_train)

    if visualize:
        return lr_tuning
    else:
        print('Best training accuracy: {0:0.4f}'.format(lr_tuning.score(X_train, y_train)))
        print('Best testing accuracy: {0:0.4f}'.format(lr_tuning.score(X_test, y_test)))
        print('Best parameters: {}'.format(lr_tuning.best_params_))


def Visualize_LogisticRegressor(X_train, X_test, y_train, y_test, component):
    lr_tuning = Logistic_Regression(X_train, X_test, y_train, y_test, visualize=True)

    # Visualize datatframe
    coeffs = pd.DataFrame([component, lr_tuning.best_estimator_.coef_[0]]).T
    coeffs.rename(columns={0: 'PCA component', 1: 'Coefficients'}, inplace=True)
    coeffs.sort_values('Coefficients', inplace=True)
    print(coeffs)

    # Visualize PCA components with strongest coefficients
    plt.figure(figsize=(10, 5), facecolor='w', dpi=150)
    plt.style.use('seaborn')
    coeffs.sort_values(by='Coefficients')['Coefficients'].plot(kind='barh', legend=None, cmap='Paired')
    plt.title('PCA components with the strongest coefficients', size=15)
    plt.ylabel('PCA components', size=12)
    plt.xlabel('Coefficients value', size=12)
    plt.show()





