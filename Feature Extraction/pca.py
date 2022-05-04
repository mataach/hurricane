import pandas as pd
import os
import numpy as np

from sklearn.decomposition import PCA


def nullIndices(data, max_missing_entries=100):
    # find  indices from dataframe that have more than a 100 missing entries
    null_info = data.isnull().sum()
    null_indices = list()
    for idx, missing_data in enumerate(null_info):
        if missing_data > max_missing_entries: null_indices.append(null_info.index[idx])
    return null_indices


def preprocessing(data, drop_indices):
    # drop indices from dataframe present in 'drop_indices'
    data = data.drop(drop_indices, axis=1)
    # Drop NaN values of the remaining indices
    data = data.dropna(),
    # Standarize data
    data_stdz = (data - data.mean()) / data.std()
    return data_stdz


def dataToPCA(data, components):
    pca = PCA(n_components=components)
    pca.fit(data)


def transformData(data, drop_indices, pca_components):
    """
    Function used to transform time series data into PCA data to be used
    as an input to the regression model
    :param data: time series data unprocessed
    :param drop_indices: indices to be dropped in case it contains indices not present in the training data
    :param pca_components: projection matrix obtained from the training data
    :return: numpy array or dataframe?? of transformed data
    """
    # data_stdz @ pca.components_.T
    data_stdz = preprocessing(data, drop_indices)
    return data_stdz @ pca_components.T


if __name__ == '__main__':
    root = os.path.abspath(os.getcwd())
    directory = os.path.split(os.path.split(root)[0])[0]
    years = 'Data/years'
    year_directory = os.path.join(directory, years)

    file = 'all_data.csv'

    data = pd.read_csv(os.path.join(year_directory, file))
    null_indices = nullIndices(data)
    data_stdz = preprocessing(data, null_indices)
