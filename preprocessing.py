import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def import_data(data_path):
    dtype_dict = {"season_id" : int}
    data = pd.read_csv(data_path, dtype=dtype_dict)
    return data

def clean_data(data):
    data = data[data['season_id'] >= 21985]
    data = data.dropna()
    data = data.drop_duplicates(subset='game_id')
    data.drop(columns=['video_available_home', 'video_available_away', 'team_abbreviation_home', 'team_name_home', 'game_date', 'matchup_home', 'team_abbreviation_away', 'team_name_away', 'matchup_away', 'wl_away'], axis=1, inplace=True)
    return data

def encode_data(data):
    categorical_columns = ['team_id_home', 'team_id_away', 'season_id']
    dummies = pd.get_dummies(data, columns=categorical_columns)
    # add dummy columns to original data
    data = pd.concat([data, dummies], axis=1)
    data.drop(columns=categorical_columns, axis=1, inplace=True)

    label_mapping = {'W': 1, 'L': 0}
    # map win/loss to 1/0
    data['wl_home'] = data['wl_home'].replace(label_mapping)
    # data['wl_home'] = data['wl_home'].map(label_mapping)

    return data

def scale_data(data):
    numerical_columns = ['min', 'fgm_home',
       'fga_home', 'fg_pct_home', 'fg3m_home', 'fg3a_home', 'fg3_pct_home',
       'ftm_home', 'fta_home', 'ft_pct_home', 'oreb_home', 'dreb_home',
       'reb_home', 'ast_home', 'stl_home', 'blk_home', 'tov_home', 'pf_home',
       'pts_home', 'plus_minus_home', 'fgm_away', 'fga_away', 'fg_pct_away', 'fg3m_away', 'fg3a_away',
       'fg3_pct_away', 'ftm_away', 'fta_away', 'ft_pct_away', 'oreb_away',
       'dreb_away', 'reb_away', 'ast_away', 'stl_away', 'blk_away', 'tov_away',
       'pf_away', 'pts_away', 'plus_minus_away']
    for column in numerical_columns:
        print(f"Data shape: {data.shape}")
        print(f"Column '{column}' shape: {data[column].values.reshape(-1, 1).shape}")
        scaler = StandardScaler()
        column_data = np.array(data[column]).reshape(-1, 1)
        test_data = scaler.fit_transform(column_data)
        data[column] = test_data
    return data

def split_data(data):
    X = data.drop(columns=['wl_home'])
    y = data['wl_home']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test

def prep_all(data_path):
    data = import_data(data_path)
    data = clean_data(data)
    data = encode_data(data)
    data = scale_data(data)
    X_train, X_test, y_train, y_test = split_data(data)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    data_path = "data/game.csv"
    data_path = "data/game.csv"
    X_train, X_test, y_train, y_test = prep_all(data_path)
