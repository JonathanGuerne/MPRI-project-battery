import pandas as pd
import numpy as np
import datetime
import random
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from random import shuffle
from numpy import linalg as la

class Model:

    def __init__(self, df_train, df_validation, df_test, param_grid, log, include_discharges):
        self.param_grid = param_grid
        self.log = log
        self.type = 'charge_nb'
        self.include_discharges = include_discharges
        # Models
        self.svm = SVC(C=10,kernel='rbf',degree=3,gamma=0.000001, decision_function_shape='ovo')
        self.clf = GridSearchCV(self.svm, param_grid, cv=10, n_jobs=-1)
        self.models_list = []
        # Train data
        self.df_train = df_train
        # Validation data
        self.df_validation = df_validation
        # Test data
        self.df_test = df_test
        # Processed and normalised set
        self.X_train = None
        self.X_validation = None
        self.X_test = None
        # Scaler
        self.scaler = RobustScaler()

    def print_log(self, str):
        log = '{} - {}\n'.format(datetime.datetime.now(), str)
        print(log)
        if self.log:
            with open("log/log.txt", "a") as file:
                file.write(log)

    @staticmethod
    def get_features_item(battery_charge_df, type):

        volt_df = battery_charge_df['voltage_measured']
        curr_df = battery_charge_df['current_measured']
        temp_df = battery_charge_df['temperature_measured']
        curr_charge_df = battery_charge_df['current_charge']
        volt_charge_df = battery_charge_df['voltage_charge']

        voltage_gradient = la.norm(np.gradient(volt_df), ord=0)
        current_gradient = la.norm(np.gradient(curr_df), ord=0)
        temperature_gradient = la.norm(np.gradient(temp_df), ord=0)
        current_charge_gradient = la.norm(np.gradient(curr_charge_df), ord=0)
        voltage_charge_gradient = la.norm(np.gradient(volt_charge_df), ord=0)

        features = {'voltage_gradient_{}'.format(type): voltage_gradient,
                    'current_gradient_{}'.format(type): current_gradient,
                    'temperature_gradient_{}'.format(type): temperature_gradient,
                    'current_charge_gradient_{}'.format(type): current_charge_gradient,
                    'voltage_charge_gradient_{}'.format(type): voltage_charge_gradient}

        return features

    # def get_features_charge_item(self, battery_charge_df, type):
    #     df_size = len(battery_charge_df['voltage_measured'])
    #     df_size_third = int(df_size/3)
    #
    #     volt_df = battery_charge_df['voltage_measured']
    #     curr_df = battery_charge_df['current_measured']
    #     temp_df = battery_charge_df['temperature_measured']
    #     curr_charge_df = battery_charge_df['current_charge']
    #     volt_charge_df = battery_charge_df['voltage_charge']
    #     cap_df = battery_charge_df['capacity']
    #
    #
    #     voltage_measured1 = np.mean(volt_df.iloc[0:df_size_third])
    #     voltage_measured2 = np.mean(volt_df.iloc[df_size_third:df_size_third * 2])
    #     voltage_measured3 = np.mean(volt_df.iloc[df_size_third * 2:df_size])
    #     voltage_gradient = la.norm(np.gradient(volt_df), ord=0)
    #
    #     current_measured1 = np.mean(curr_df.iloc[0:df_size_third])
    #     current_measured2 = np.mean(curr_df.iloc[df_size_third:df_size_third * 2])
    #     current_measured3 = np.mean(curr_df.iloc[df_size_third * 2:df_size])
    #     current_gradient = la.norm(np.gradient(volt_df), ord=0)
    #
    #     # power1 = voltage_measured1 * current_measured1
    #     # power2 = voltage_measured2 * current_measured2
    #     # power3 = voltage_measured3 * current_measured3
    #
    #     temperature_measured1 = np.mean(temp_df.iloc[0:df_size_third])
    #     temperature_measured2 = np.mean(temp_df.iloc[df_size_third:df_size_third * 2])
    #     temperature_measured3 = np.mean(temp_df.iloc[df_size_third * 2:df_size])
    #     temperature_gradient = la.norm(np.gradient(temp_df), ord=0)
    #
    #     current_charge1 = np.mean(curr_charge_df.iloc[0:df_size_third])
    #     current_charge2 = np.mean(curr_charge_df.iloc[df_size_third:df_size_third * 2])
    #     current_charge3 = np.mean(curr_charge_df.iloc[df_size_third * 2:df_size])
    #     current_charge_gradient = la.norm(np.gradient(curr_charge_df), ord=0)
    #
    #     voltage_charge1 = np.mean(volt_charge_df.iloc[0:df_size_third])
    #     voltage_charge2 = np.mean(volt_charge_df.iloc[df_size_third:df_size_third * 2])
    #     voltage_charge3 = np.mean(volt_charge_df.iloc[df_size_third * 2:df_size])
    #     voltage_charge_gradient = la.norm(np.gradient(volt_charge_df), ord=0)
    #
    #     # power_charge1 = current_charge1 * voltage_charge1
    #     # power_charge2 = current_charge2 * voltage_charge2
    #     # power_charge3 = current_charge3 * voltage_charge3
    #
    #     # features = {'voltage_measured1_{}'.format(type): voltage_measured1, 'voltage_measured2_{}'.format(type): voltage_measured2,
    #     #             'voltage_measured3_{}'.format(type): voltage_measured3, 'current_measured1_{}'.format(type): current_measured1,
    #     #             'current_measured2_{}'.format(type): current_measured2, 'current_measured3_{}'.format(type): current_measured3,
    #     #             'temperature_measured1_{}'.format(type): temperature_measured1, 'temperature_measured2_{}'.format(type): temperature_measured2,
    #     #             'temperature_measured3_{}'.format(type): temperature_measured3, 'current_charge1_{}'.format(type) : current_charge1,
    #     #             'current_charge2_{}'.format(type) : current_charge2, 'current_charge3_{}'.format(type) : current_charge3,
    #     #             'voltage_charge1_{}'.format(type): voltage_charge1, 'voltage_charge2_{}'.format(type): voltage_charge2,
    #     #             'voltage_charge3_{}'.format(type): voltage_charge3,
    #     #             'voltage_gradient_{}'.format(type): voltage_gradient, 'current_gradient_{}'.format(type): current_gradient,
    #     #             'temperature_gradient_{}'.format(type): temperature_gradient, 'current_charge_gradient_{}'.format(type): current_charge_gradient,
    #     #             'voltage_charge_gradient_{}'.format(type): voltage_charge_gradient}
    #
    #     features = {'voltage_gradient_{}'.format(type): voltage_gradient, 'current_gradient_{}'.format(type): current_gradient,
    #                 'temperature_gradient_{}'.format(type): temperature_gradient, 'current_charge_gradient_{}'.format(type): current_charge_gradient,
    #                 'voltage_charge_gradient_{}'.format(type): voltage_charge_gradient}
    #
    #     if battery_charge_df['capacity'].any():
    #         capacity1 = np.mean(cap_df.iloc[0:df_size_third])
    #         capacity2 = np.mean(cap_df.iloc[df_size_third:df_size_third * 2])
    #         capacity3 = np.mean(cap_df.iloc[df_size_third * 2:df_size])
    #         capacity_gradient = la.norm(np.gradient(cap_df), ord=0)
    #
    #         # features['capacity1'] = capacity1
    #         # features['capacity2'] = capacity2
    #         # features['capacity3'] = capacity3
    #         features['capacity_gradient'] = capacity_gradient
    #
    #     return features

    @staticmethod
    def split_x_y(df):
        return df.drop('label', axis=1), df['label']

    def process_data(self, df, distinct_batteries=None):
        features_list = []

        if distinct_batteries is None:
            distinct_batteries = df['battery_nb'].unique()

        for battery_id in distinct_batteries:
            features_list = features_list + self.get_battery_by_id(df, battery_id)

        shuffle(features_list)

        return pd.DataFrame(features_list)

    def get_battery_by_id(self, df, battery_id):
        features_list = []
        battery_df = df[df['battery_nb'] == battery_id]
        distinct_charges = df[self.type].dropna().unique()

        discharge_id = 1

        for charge_id in distinct_charges:
            battery_charge_df = battery_df[battery_df[self.type] == charge_id]
            battery_discharge_df = battery_df[battery_df['discharge_nb'] == discharge_id]
            features = {}

            if battery_discharge_df.empty:
                discharge_id -= 1
                battery_discharge_df = battery_df[battery_df['discharge_nb'] == discharge_id]

            if not battery_charge_df.empty:
                features_charge = self.get_features_item(battery_charge_df, 'charge')
                features.update(features_charge)

                if self.include_discharges:
                    features_discharge = self.get_features_item(battery_discharge_df, 'discharge')
                    features.update(features_discharge)
                    discharge_id += 1

                if 'quality' in battery_charge_df.keys() and 'quality' in battery_discharge_df.keys():
                    quality_charge = int(round(np.nanmean(battery_charge_df['quality']), 0))
                    quality_discharge = int(round(np.nanmean(battery_discharge_df['quality']), 0))
                    quality = int(round((quality_charge+quality_discharge)/2))
                    features['label'] = quality

                features_list.append(features)

        return features_list

    def leave_one_out(self):
        distinct_batteries = self.df_train['battery_nb'].unique()

        validation = self.process_data(self.df_validation)
        self.X_validation, y_validation = self.split_x_y(validation)

        self.print_log('FEATURES : {}'.format(list(self.X_validation.keys())))

        prediction_validation_list = []
        f1_score_validation_list = []
        f1_score_test_list = []

        for i in range(len(distinct_batteries)):
            distinct_batteries_copy = list(distinct_batteries.copy())
            self.X_test, y_test = self.split_x_y(self.process_data(self.df_train, [distinct_batteries_copy[i]]))
            del distinct_batteries_copy[i]
            self.X_train, y_train = self.split_x_y(self.process_data(self.df_train, distinct_batteries_copy))

            predict_list_validation, f1_score_validation, f1_score_test = self.svm_fit(y_train, y_validation, y_test)

            self.models_list.append(self.svm)

            prediction_validation_list.append(predict_list_validation)
            f1_score_validation_list.append(f1_score_validation)
            f1_score_test_list.append(f1_score_test)

        prediction_validation = np.around(np.mean(np.array(prediction_validation_list), 0))
        f1_validation = np.mean(np.array(f1_score_validation_list))
        f1_test = np.mean(np.array(f1_score_test_list))

        self.print_log('PREDICTION VALIDATION : {}'.format(list(prediction_validation)))
        print('F1 SCORE VALIDATION COMPARISON : {}',format(f1_score(y_validation, prediction_validation, average='macro')))
        self.print_log('F1 SCORE VALIDATION :{}'.format(f1_validation))
        self.print_log('F1 SCORE TEST :{}'.format(f1_test))

        return prediction_validation, f1_validation, f1_test

    def predict_test(self):
        self.X_test = self.process_data(self.df_test)

        prediction_test_list = []

        for model in self.models_list:
            prediction = model.predict(self.X_test)
            prediction_test_list.append(prediction)

        prediction_test = np.around(np.mean(np.array(prediction_test_list), 0))
        self.print_log('PREDICTION TEST : {}'.format(list(prediction_test)))
        return prediction_test

    def grid_search_fit(self):
        train = self.process_data(self.df_train)
        self.X_train, y_train = self.split_x_y(train)

        self.clf.fit(self.X_train, y_train)
        self.print_log('BEST PARAMETERS : {}'.format(self.clf.best_params_))
        self.svm.set_params(**self.clf.best_params_)

    def svm_fit(self, y_train, y_validation, y_test):
        model = self.svm

        model.fit(self.X_train, y_train)

        y_predict_validation = model.predict(self.X_validation)
        predict_list_validation = y_predict_validation.tolist()
        f1_score_validation = f1_score(y_validation, y_predict_validation, average='macro')

        y_predict_test = model.predict(self.X_test)
        f1_score_test = f1_score(y_test, y_predict_test, average='macro')

        return predict_list_validation, f1_score_validation, f1_score_test
