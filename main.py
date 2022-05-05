import os
import numpy as np
import pandas as pd
import random

import torch.nn
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from Network.classification import DummyClf
from Network.classification import XGBoostClf_CV, XGBoostClassifier
from Network.classification import Logistic_Regression, Visualize_LogisticRegressor
from Network.regression import XGBoost_CV, XGBoost, RidgeRegression
from Network.mlp import Classifier, Trainer


class LoadData:
    def __init__(self, is_clf, is_balanced):
        self.regression = 1 if not is_clf else 0
        self.classification = 1 if is_clf else 0
        self.is_balanced = is_balanced

        self.root = os.path.abspath(os.getcwd())
        self.data_dir = self.root + '/data'

    @staticmethod
    def encode_labels(labels):
        """
        Class 0: No hurricanes this month
        Class 1: There are >0 and <=2 hurricanes this month
        Class 2: There are >2 and <=6 hurricanes this month
        Class 3: There are >6 and <=13 hurricanes this month
        Class 4: There are > 13 hurricanes this month
        :return: one-hot encoded labels
        """
        classes = []
        arr_labels = labels.to_numpy()
        for hurricane in arr_labels:
            if hurricane == 0: classes.append(0)
            elif 0 < hurricane <= 2: classes.append(1)
            elif 2 < hurricane <= 6: classes.append(2)
            elif 6 < hurricane <= 13: classes.append(3)
            else: classes.append(4)
        one_hot_labels = pd.DataFrame(data=classes, columns=['class'])
        one_hot_labels = pd.get_dummies(one_hot_labels, columns=['class'], prefix_sep=["_"])
        return one_hot_labels

    @staticmethod
    def balance_data(data, labels):
        """
        Unbalanced data with label=0 hurricanes dominates -> remove unbalanced data
        :param data: PCA data
        :param labels: number of hurricanes >=0
        :return: Balanced dataset
        """
        random.seed(RANDOM_STATE)
        zeros = labels[labels == 0].index.to_list()
        # Choose multiple random zero-label items without repetition
        remove = random.sample(zeros, k=len(labels)-len(zeros))
        X = data.drop(index=remove)
        y = labels.drop(index=remove)
        return X, y

    def load_data(self, file, pca):
        # get indices of the data that will be used for regression
        pca_df = pd.read_csv(self.data_dir + pca, index_col='Unnamed: 0')
        samples = list(pca_df.index)
        # get the labels of those data indices
        data_df = pd.read_csv(self.data_dir + file, index_col='Unnamed: 0')
        data_df = data_df[data_df.index.isin(samples)]
        # target
        labels = data_df['Hurricane activity']
        # remove null labels and data with null labels
        null_labels = labels[labels.isnull()].index.tolist()
        labels.dropna(inplace=True)
        pca_df.drop(index=null_labels, inplace=True)
        # Balance data?
        X, y = self.balance_data(pca_df, labels) if self.is_balanced else (pca_df, labels)
        # Encode into one-hot encoding if classification
        if self.classification: y = self.encode_labels(y)
        # Separate training and testing dataframes. Divide 80:20 split
        X_train, X_test, y_train, y_test = train_test_split(X.to_numpy(), y.to_numpy(),
                                                            train_size=0.75, random_state=RANDOM_STATE)
        return X_train, X_test, y_train, y_test, list(pca_df.columns)

    def Loader(self, X_train, X_test, y_train, y_test):
        # batch size is the largest common factor between the training and test size
        size = 33 if not self.is_balanced else 1

        X_train_tensor, y_train_tensor = torch.Tensor(X_train), torch.Tensor(y_train)
        X_test_tensor, y_test_tensor = torch.Tensor(X_test), torch.Tensor(y_test)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        train_loader = DataLoader(train_dataset, batch_size=size)
        test_loader = DataLoader(test_dataset, batch_size=len(X_test))

        return train_loader, test_loader


if __name__ == '__main__':
    """
    How to reproduce results: Uncomment the desired classifier/regressor to be used
    """
    RANDOM_STATE = 123

    Data = LoadData(is_clf=True, is_balanced=True)
    X_train, X_test, y_train, y_test, component = Data.load_data(file='/all_data.csv', pca='/pca_data.csv')

    ### Dummy
    # DummyClf(X_train, y_train)

    ### Classification Boosting
    # params = XGBoostClf_CV(X_train, y_train)
    # XGBoostClassifier(X_train, X_test, y_train, y_test, params=[2, 0.1, 0.01])

    ### Regression Boosting
    # params = XGBoost_CV(X_train, y_train)
    # XGBoost(X_train, X_test, y_train, y_test, params=[0.19, 0.3, 2, 0.1])

    ### Linear Regression -- Elastic Net
    # RidgeRegression(X_train, X_test, y_train, y_test)

    ### Logistic Regression
    # Logistic_Regression(X_train, X_test, y_train, y_test)
    # Visualize_LogisticRegressor(X_train, X_test, y_train, y_test, component)

    ### MLP
    # train_loader, test_loader = Data.Loader(X_train, X_test, y_train, y_test)
    # net = Classifier().MLP
    # trainer = Trainer(
    #     model=net,
    #     epochs=50,
    #     criterion=torch.nn.CrossEntropyLoss(),
    #     optim=torch.optim.Adam(net.parameters(), lr=1e-3)
    # )
    # trainer.train(train_loader)
    # trainer.test(test_loader)
    # trainer.visualize_confusion_matrix(y_test, test_loader)





