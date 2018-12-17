import pandas as pd
import datetime
from SVM.Model import Model


class SVM:

    def __init__(self, param_grid, df_train, df_validation, df_test, log, include_discharges=False):
        # Print what are the parameters that will be used for the GridSearchCV
        self.log = log
        self.include_discharges = include_discharges
        # New run
        dashes = '---------------------------------------------------'
        self.print_log('{} NEW RUN {}'.format(dashes, dashes))
        self.print_log('PARAM GRID : {}'.format(param_grid))
        # Data
        self.df_train = df_train
        self.df_validation = df_validation
        self.df_test = df_test
        # Load the data
        #self.load_data()
        # Models
        self.model = Model(self.df_train, self.df_validation, self.df_test, param_grid, log, include_discharges)
        # Is discharge data include ?
        if self.include_discharges:
            verb = "ARE"
        else:
            verb = "ARE'NT"

        self.print_log("/!\ WARNING : DISCHARGE DATA {} TAKE IN COUNT".format(verb))


    def print_log(self, str):
        log = '{} - {}\n'.format(datetime.datetime.now(), str)
        print(log)
        if self.log:
            with open("log/log.txt", "a") as file:
                file.write(log)

    def fit_all(self):
        self.model.grid_search_fit()
        self.model.leave_one_out()
        self.model.predict_test()


if __name__ == '__main__':
    param_grid = {
        "C": [10,100],
        "kernel": ['rbf', 'sigmoid'],
        "coef0": [0.0],
        'gamma': ['auto',0.000001],
        'decision_function_shape': ['ovo','ovr']
    }

    df_train = pd.read_pickle("../datas/training_set.pckl")
    df_validation = pd.read_pickle("../datas/validation_set.pckl")
    df_test = pd.read_pickle("../datas/test_set.pckl")

    svm = SVM(param_grid, df_train, df_validation, df_test, log=True, include_discharges=True)
    svm.fit_all()




