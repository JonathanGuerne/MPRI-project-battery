import pandas as pd
import numpy as np
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score

class RandomForest:

    def __init__(self, param_grid, log):
        # Print what are the parameters that will be used for the GridSearchCV
        self.log = log
        self.print_log('--------------------------------------------------- NEW RUN ---------------------------------------------------')
        self.print_log('/!\\ WARNING : DISCHARGE INFORMATION ISN\'T TAKE IN COUNT')
        self.print_log('PARAM GRID : {}'.format(param_grid))
        # Data
        self.df_train = None
        self.df_validation = None
        self.df_test = None
        self.df_train_processed = None
        self.df_validation_processed = None
        self.df_test_processed = None
        # Models
        self.random_forest = RandomForestClassifier()
        self.clf = GridSearchCV(self.random_forest, param_grid, cv=10, n_jobs=-1)
        # Load the data
        self.load_data()
        # Separate labels and features
        self.X_train = self.df_train_processed.drop('label', axis=1)
        self.y_train = self.df_train_processed['label']
        self.X_test = self.df_validation_processed.drop('label', axis=1)
        self.y_test = self.df_validation_processed['label']
        # Print what are the actual features
        self.print_log('FEATURES : {}'.format(list(self.X_train.keys())))

    def load_data(self):
        self.df_train = pd.read_pickle("datas/training_set.pckl")
        self.df_validation = pd.read_pickle("datas/validation_set.pckl")
        #self.df_test = pd.read_pickle("datas/test_set.pckl")

        self.df_train_processed = self.process_data(self.df_train)
        self.df_validation_processed = self.process_data(self.df_validation)
        #self.df_test_processed = self.process_data(self.df_test)

    def print_log(self, str):
        log = '{} - {}\n'.format(datetime.datetime.now(), str)
        print(log)
        if self.log:
            with open("log/log.txt", "a") as file:
                file.write(log)

    @staticmethod
    def get_features_charge_item(battery_charge_df):
        voltage_mean = np.mean(battery_charge_df['voltage_measured'])
        voltage_min = np.min(battery_charge_df['voltage_measured'])
        voltage_max = np.max(battery_charge_df['voltage_measured'])
        charge_nb = np.mean(battery_charge_df['charge_nb'])
        voltage_measured = np.mean(battery_charge_df['voltage_measured'])
        current_measured = np.mean(battery_charge_df['current_measured'])
        temperature_measured = np.mean(battery_charge_df['temperature_measured'])
        current_charge = np.mean(battery_charge_df['current_charge'])
        voltage_charge = np.mean(battery_charge_df['voltage_charge'])

        return {'voltage_mean': voltage_mean, 'voltage_max': voltage_max, 'voltage_min': voltage_min,
                'charge_nb': charge_nb, 'voltage_measured':voltage_measured, 'current_measured': current_measured,
                'temperature_measured': temperature_measured, 'current_charge':current_charge,
                'voltage_charge': voltage_charge}

    def process_data(self, df):
        distinct_batteries = df['battery_nb'].unique()
        features_dict = []
        for battery_id in distinct_batteries:
            battery_df = df[df['battery_nb'] == battery_id]
            distinct_charges = df['charge_nb'].dropna().unique()
            for charge_id in distinct_charges:
                battery_charge_df = battery_df[battery_df['charge_nb'] == charge_id]
                if battery_charge_df.empty:
                    continue

                quality = int(round(np.nanmean(battery_charge_df['quality']), 0))
                features = self.get_features_charge_item(battery_charge_df)
                features['label'] = quality
                features_dict.append(features)

        return pd.DataFrame(features_dict)

    def rf_fit(self, **params):
        self.random_forest.set_params(**params)
        self.random_forest.fit(self.X_train, self.y_train)
        self.print_log('NOW PREDICTING WITH RANDOM FOREST MODEL....')
        self.print_log('FEATURES IMPORTANCE : {}'.format(self.random_forest.feature_importances_))
        self.print_log('BEST PARAMETERS : {}'.format(params))
        self.predict(self.random_forest)

    def clf_fit(self):
        self.clf.fit(self.X_train, self.y_train)
        self.print_log('NOW PREDICTING WITH GRID SEARCH MODEL....')
        self.predict(self.clf)

    def predict(self, model):
        y_predict = model.predict(self.X_test)
        self.print_log('GROUND TRUTH : {}'.format(self.y_test.tolist()))
        self.print_log('PREDICTION : {}'.format(y_predict.tolist()))
        self.print_log('CONFUSION MATRIX : \n{}'.format(confusion_matrix(self.y_test, y_predict)))
        self.print_log('F1 SCORE : {}\n-------'.format(f1_score(self.y_test, y_predict, average='macro')))

if __name__ == '__main__':
    param_grid = {
        'n_estimators': [100, 200, 300, 400, 500, 600, 700],
        'max_features': ['auto', 'sqrt', 'log2']
    }

    rf = RandomForest(param_grid, True)
    rf.clf_fit()
    rf.rf_fit(**rf.clf.best_params_)




