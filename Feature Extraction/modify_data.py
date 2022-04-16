import os
import pandas as pd
import numpy as np
from collections import defaultdict


def getIndicesNames(directory):
    """
    This function gets the names of all the sea index for building the full dataset in `combineData` function
    :param directory: index folder
    :return: list containing all the sea indices
    """
    names = ['Month']
    for filename in os.scandir(directory):
        file_extension = os.path.splitext(filename)[1]
            names.append(filename.name[:-4])
    return names

def createYearCSV(directory, year):
    """
    This function creates a new folder where data is separated by year instead of by sea index
    :param directory: index folder
    :param year: year to create new csv file
    :return: csv file created
    """
    features = dict()
    for filename in os.scandir(directory):
        file_extension = os.path.splitext(filename)[1]
        if filename.is_file() and file_extension == '.csv':                 # make sure is not reading hidden files
            df = pd.read_csv(filename.path, index_col='Year').dropna()
            index_name = filename.name[:-4]
            if year in df.index:
                feature_vector = list(df.loc[year].to_numpy())
                features[index_name] = feature_vector
    return pd.DataFrame.from_dict(features, orient='columns')

def getYearData(directory, year, names):
    """
    Return a dictionary of each years data. Missing values are filled with null
    :param directory: year folder created using the `createYearCSV file
    :param year: year to take the data from
    :param names: a list of indices names which will be the features (keys) of the newly created csv file
    :return: dictionary of the data of that year. fill in missing indices values with null values
    """
    months = np.arange(1, 13)
    file_name = '{}.csv'.format(year)
    file = os.path.join(directory, file_name)
    df = pd.read_csv(file).dropna()
    df['Month'] = months                            # add the month as a feature
    year_data = df.to_dict(orient='list')
    del year_data['Unnamed: 0']                     # delete this unnecessary column of the df
    for name in names:
        if name not in year_data.keys():
            year_data[name] = [None]*len(months)    # fill in missing indices with null entries
    return year_data

def combineData(indices_directory, year_directory, init_year, end_year):
    """
    Combine all the year data inside `year_directory` into a single csv file
    :param indices_directory:
    :param year_directory:
    :param init_year:
    :param end_year:
    :return: dataframe of all the data
    """
    indices_names = getIndicesNames(indices_directory)
    data = defaultdict(list)
    for year in range(init_year, end_year+1):
        year_data = getYearData(year_directory, year, indices_names)
        for feature, value in year_data.items(): data[feature].append(value)
    data = {feature: np.array(value).flatten() for feature, value in data.items()}
    return pd.DataFrame.from_dict(data, orient='columns')

if __name__ == '__main__':
    root = os.path.abspath(os.getcwd())
    directory = os.path.split(os.path.split(root)[0])[0]
    indices = 'Data/indices'
    years = 'Data/years'
    indices_directory = os.path.join(directory, indices)
    year_directory = os.path.join(directory, years)

    init_year, end_year = 1948, 2022
    # for year in range(init_year, end_year+1):
    #     year_data = createYearCSV(indices_directory, year)
    #     year_data.to_csv(os.path.join(year_directory, '{}.csv'.format(year)))


    data = combineData(indices_directory, year_directory, init_year, end_year)
    data.to_csv(os.path.join(year_directory, 'all_data.csv'))


