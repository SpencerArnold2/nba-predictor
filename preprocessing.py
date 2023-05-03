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
    data = average_data(data)
    data.drop(columns=['video_available_home', 'video_available_away', 'team_abbreviation_home', 'team_name_home', 'game_date', 'matchup_home', 'team_abbreviation_away', 'team_name_away', 'matchup_away', 'wl_away'], axis=1, inplace=True)
    return data

def average_data(df):
    home_df = df.filter(regex='home|season_id|game_date|game_id').copy()
    away_df = df.filter(regex='away|season_id|game_date|game_id').copy()

    home_df.columns = home_df.columns.str.replace('_home', '')
    away_df.columns = away_df.columns.str.replace('_away', '')
    all_games = pd.DataFrame(columns=home_df.columns)

    # Add home and away data to the new dataframe
    for column in home_df.columns:
        all_games[column] = pd.concat([home_df[column], away_df[column]], ignore_index=True)


    # Sort the DataFrame
    all_games = all_games.sort_values(by=['team_id', 'season_id', 'game_date'])

    # Group the DataFrame
    grouped = all_games.groupby(['team_id', 'season_id'])

    # Calculate the rolling average for each stat column
    columns_to_average = [column for column in all_games.columns if column not in ['team_id', 'season_id', 'game_date', 'game_id']]
    rolling_averages = grouped[columns_to_average].apply(lambda x: x.rolling(window=len(x), min_periods=1).mean().shift(1))
    rolling_averages['team_id'] = all_games['team_id']
    rolling_averages['season_id'] = all_games['season_id']
    rolling_averages['game_date'] = all_games['game_date']


    rolling_averages_home = rolling_averages.add_suffix('_home')
    rolling_averages_home.rename(columns={'team_id_home': 'team_id', 'season_id_home': 'season_id', 'game_date_home': 'game_date'}, inplace=True)
    df = pd.merge(df, rolling_averages_home, left_on=['team_id_home', 'season_id', 'game_date'], right_on=['team_id', 'season_id', 'game_date'], how='left')

    rolling_averages_away = rolling_averages.add_suffix('_away')
    rolling_averages_away.rename(columns={'team_id_away': 'team_id', 'season_id_away': 'season_id', 'game_date_away': 'game_date'}, inplace=True)
    df = pd.merge(df, rolling_averages_away, left_on=['team_id_away', 'season_id', 'game_date'], right_on=['team_id', 'season_id', 'game_date'], how='left')


    # Drop columns with the suffix '_x'
    df = df[df.columns.drop(list(df.filter(regex='_x')))]

    # Rename columns to remove the '_y' suffix
    df.columns = df.columns.str.replace('_y$', '')

    # If you want to drop any remaining duplicate columns, you can use the following code:
    df = df.loc[:, ~df.columns.duplicated()]
    df.drop(['team_id'], axis=1, inplace=True)
    df.dropna(subset=['fgm_home', 'fgm_away'],inplace=True)
    return df


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
    data = data.loc[:, ~data.columns.duplicated()]

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
        # print(f"Data shape: {data.shape}")
        # print(f"Column '{column}' shape: {data[column].values.reshape(-1, 1).shape}")
        scaler = StandardScaler()
        column_data = np.array(data[column]).reshape(-1, 1)
        test_data = scaler.fit_transform(column_data)
        data[f"{column}"] = test_data
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
    X_train, X_test, y_train, y_test = prep_all(data_path)
