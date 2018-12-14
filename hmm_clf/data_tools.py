import numpy as np
import pandas as pd
from random import shuffle

list_features = ['voltage_measured', 'current_measured', 'temperature_measured',
                 'current_charge', 'voltage_charge', 'puissance_charge']


def shuffle_df(df):
    battery_ids = np.unique(df['battery_nb'])

    lst = []
    for i in battery_ids:
        sub = df.loc[df['battery_nb'] == i]

        charge_ids = np.unique(sub['charge_nb'])
        charge_ids = charge_ids[~np.isnan(charge_ids)]

        for j in charge_ids:
            lst.append(sub.loc[sub['charge_nb'] == j])
    shuffle(lst)
    return pd.concat(lst)


def get_charge_list(df, drop_quality=False, drop_charge_nb=False, drop_battery_nb=False):
    df = df[
        ['battery_nb', 'charge_nb', 'voltage_measured', 'current_measured', 'temperature_measured', 'current_charge',
         'voltage_charge', 'puissance_charge', 'quality']]


    if 'quality' in df.columns and drop_quality:
        df = df.drop(['quality'], axis=1)

    lst_battery = np.unique(df['battery_nb'])
    tmp_lst = []

    for battery in lst_battery:

        battery_df = df.loc[df['battery_nb'] == battery]

        if drop_battery_nb:
            battery_df = battery_df.drop(['battery_nb'], axis=1)

        lst_charges = np.unique(battery_df['charge_nb'])
        lst_charges = lst_charges[~np.isnan(lst_charges)]

        for charge in lst_charges:
            charges_df = battery_df.loc[battery_df['charge_nb'] == charge]

            if drop_charge_nb == True:
                charges_df = charges_df.drop(['charge_nb'], axis=1)

            # vals = charges_df.values
            tmp_lst.append(charges_df.dropna())

    return tmp_lst


def split_training_validation(lst_charges, battery_number):
    lst_train = []
    lst_valid = []

    for charge in lst_charges:
        if charge.iloc[0].loc['battery_nb'] == battery_number:
            lst_valid.append(charge)
        else:
            lst_train.append(charge)

    return lst_train, lst_valid


def battery_list(df):
    return np.unique(df['battery_nb'])


def split_by_quality(charges):
    quality_charges = {1: [], 2: [], 3: []}
    for charge in charges:
        q_ = round(charge['quality'].mean())
        quality_charges[q_].append(charge)
    return quality_charges


def remove_cols_to_df(df):
    return df[list_features]


def remove_cols(_qual_list):
    qual_list = {1: [], 2: [], 3: []}
    for i in [1, 2, 3]:
        for j in range(len(_qual_list[i])):
            qual_list[i].append(remove_cols_to_df(_qual_list[i][j]))
    return qual_list


def add_features(df):
    # add new features
    df['puissance_charge'] = df['current_charge'] * df['voltage_charge']
    df = df.dropna(subset=list_features)
    return df


def normalize(df):
    from sklearn import preprocessing

    for f in list_features:
        # # Create x, where x the 'scores' column's values as floats
        # x = df[[f]].values.astype(float)
        #
        # x
        #
        # # Create a minimum and maximum processor object
        # min_max_scaler = preprocessing.MinMaxScaler()
        #
        # # Create an object to transform the data to fit minmax processor
        # x_scaled = min_max_scaler.fit_transform(x)
        #
        # #temp = pd.DataFrame(x_scaled)
        #
        #
        # print(df.iloc[-1])
        # df[f] = pd.DataFrame(x)
        # print(df.iloc[-1])
        df[f] = (df[f] - df[f].min()) / (df[f].max() - df[f].min())

    return df
