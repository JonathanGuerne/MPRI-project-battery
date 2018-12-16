import pandas as pd
import datetime
from Model import Model

class SVM:

    def __init__(self, param_grid, log, include_discharges=False):
        # Print what are the parameters that will be used for the GridSearchCV
        self.log = log
        self.include_discharges = include_discharges
        # New run
        dashes = '---------------------------------------------------'
        self.print_log('{} NEW RUN {}'.format(dashes, dashes))
        self.print_log('PARAM GRID : {}'.format(param_grid))
        # Data
        self.df_train = None
        self.df_validation = None
        self.df_test = None
        # Load the data
        self.load_data()
        # Models
        self.model = Model(self.df_train, self.df_validation, self.df_test, param_grid, log, include_discharges)
        # Is discharge data include ?
        if self.include_discharges:
            verb = "ARE"
        else:
            verb = "ARE'NT"

        self.print_log("/!\ WARNING : DISCHARGE DATA {} TAKE IN COUNT".format(verb))

    def load_data(self):
        self.df_train = pd.read_pickle("../data/mpri_challenge/training_set.pckl")
        self.df_validation = pd.read_pickle("../data/mpri_challenge/validation_set.pckl")
        self.df_test = pd.read_pickle("../data/mpri_challenge/test_set.pckl")
        self.print_log("Data loaded")

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
        "C": [0.1, 1, 10, 100],
        "kernel": ['rbf','sigmoid'],
        "degree": [4,5,6],
        'gamma': [0.1, 1, 10, 100]
            }

    svm = SVM(param_grid, log=True, include_discharges=False)
    svm.fit_all()




