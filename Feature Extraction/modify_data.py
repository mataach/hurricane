import os
import pandas as pd


def createYearCSV(directory, year):
    features = dict()
    for filename in os.scandir(directory):
        file_extension = os.path.splitext(filename)[1]
        if filename.is_file() and file_extension == '.csv':                 # make sure is not reading hidden files
            df = pd.read_csv(filename.path, index_col='Year').dropna()
            index_name = filename.name[:-4]
            print(index_name)
            print(df.index.dtype == int)
            if year in df.index:
                feature_vector = list(df.loc[year].to_numpy())
                features[index_name] = feature_vector
    return pd.DataFrame.from_dict(features, orient='columns')


if __name__ == '__main__':
    root = os.path.abspath(os.getcwd())
    directory = os.path.split(os.path.split(root)[0])[0]
    indices = 'Data/indices'
    years = 'Data/years'
    indices_directory = os.path.join(directory, indices)
    year_directory = os.path.join(directory, years)

    init_year, end_year = 1948, 2022
    for year in range(init_year, end_year+1):
        year_data = createYearCSV(indices_directory, year)
        year_data.to_csv(os.path.join(year_directory, '{}.csv'.format(year)))