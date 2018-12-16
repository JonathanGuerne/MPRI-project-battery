from random_forest.random_forest import RandomForest
from SVM.svm import SVM
from hmm_clf.data_tools import get_charge_list, remove_cols_to_df, add_features, gradient
from hmm_clf.HMM_as_script import predict, eval_loocv
import pandas as pd
import datetime


class Fusion:

    def __init__(self, df_train, df_validation, df_test):
        self.df_train = df_train
        self.df_validation = df_validation
        self.df_test = df_test
        self.result_dict = {}

    def run_hmm(self):
        df_validation = add_features(self.df_validation)
        df_train = add_features(self.df_train)
        df_test = add_features(self.df_test)

        validation_charges_list = get_charge_list(df_validation, False, False, False)
        validation_charges_hide = []
        for charge in validation_charges_list:
            validation_charges_hide.append(remove_cols_to_df(charge))
        validation_charges_hide = gradient(validation_charges_hide)

        test_charge_list = get_charge_list(df_test, drop_quality=True)
        test_charge_hide = []
        for charge in test_charge_list:
            test_charge_hide.append(remove_cols_to_df(charge))
        test_charge_hide = gradient(test_charge_hide)

        now = datetime.datetime.now()
        file_name = now.strftime("%d-%m-%y_gradient_norm.csv")

        params = {'n_iter': {1: 5, 2: 5, 3: 5},
                  'factor': 70,
                  'n_state': 6,
                  'sampeling': 'None',
                  'nb_charge_saved': 80,
                  'covariance': 'diag'}

        prediction = predict(df_train, test_charge_hide, params)
        f1_validation, f1_test = eval_loocv(df_train, validation_charges_list, validation_charges_hide, params, verbose=False)
        self.result_dict['hmm'] = [prediction, f1_validation, f1_test]

    def run_rf(self):
        param_grid = {
            "n_estimators": [5, 10, 20, 50, 75, 100, 300],
            "max_depth": [2, 3, 5, 10, None],
            'class_weight': ['balanced', 'balanced_subsample'],
            'max_features': ['auto', 'sqrt', 'log2']
        }
        rf = RandomForest(param_grid, self.df_train, self.df_validation, self.df_test, log=False, include_discharges=True)
        prediction, f1_validation, f1_test = rf.model.leave_one_out()
        self.result_dict['rf'] = [prediction, f1_validation, f1_test]

    def run_svm(self):
        param_grid = {
            "C": [0.1, 1, 10, 100],
            "kernel": ['rbf', 'sigmoid'],
            "degree": [4, 5, 6],
            'gamma': [0.1, 1, 10, 100]
        }
        svm = SVM(param_grid, self.df_train, self.df_validation, self.df_test, log=True, include_discharges=False)
        prediction, f1_validation, f1_test = svm.model.leave_one_out()
        self.result_dict['svm'] = [prediction, f1_validation, f1_test]

    def define_best(self):
        list_models = ['hmm', 'rf', 'svm']
        self.run_hmm()
        self.run_rf()
        self.run_svm()

        for m1 in list_models:
            current_model = self.result_dict[m1]

            for m2 in list_models:
                if m2 is not current_model:
                    if abs(self.result_dict[current_model][1] - self.result_dict[m2][1]) < 10:
                        print(m2)
                        print(current_model)


if __name__ == '__main__':
    df_validation = pd.read_pickle("datas/validation_set.pckl")
    df_train = pd.read_pickle("datas/training_set.pckl")
    df_test = pd.read_pickle("datas/test_set.pckl")

    fusion = Fusion(df_train, df_validation, df_test)
    fusion.define_best()
