import datetime
import datetime
import os.path

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from hmm_clf.data_tools import remove_cols, split_by_quality, get_charge_list, battery_list, shuffle_df, \
    split_training_validation, remove_cols_to_df
from hmm_clf.evaluation_tools import evaluate_model
from hmm_clf.file_tools import creat_file, _save


class HMMBattery():
    def __init__(self, covariance='full', nb_charge_saved=55, nb_states=6, factor=32, n_iter={1: 10, 2: 10, 3: 12},
                 sampeling='log', verbose=True):
        self.n_state = nb_states
        self.n_iter = n_iter
        self.nb_charge_saved = nb_charge_saved
        self.factor = factor
        self.verbose = verbose
        self.sampeling = sampeling
        self.covariance = covariance
        self.models = None

    def params(self):
        return {'n_state': self.n_state,
                'n_iter': self.n_iter,
                'nb_charge_saved': self.nb_charge_saved,
                'factor': self.factor,
                'sampeling': self.sampeling,
                'covariance': self.activation}

    def load_params(self, params_dic):
        self.n_state = params_dic['n_state']
        self.n_iter = params_dic['n_iter']
        self.nb_charge_saved = params_dic['nb_charge_saved']
        self.factor = params_dic['factor']
        self.sampeling = params_dic['sampeling']
        self.covariance = params_dic['covariance']

    def reduce_list_size(self, _list, nb_slice):
        ids = [i * int(len(_list) / nb_slice) for i in range(0, nb_slice, 1)]
        return [_list[i] for i in ids]

    def log_sampeling(self, arr, size):
        _max = len(arr) - 1
        _exposant = 2.0
        ids = np.round(np.power(np.linspace(0, 1, size), _exposant) * _max)

        return arr[ids.astype(int)]

    def concatenate_cepstrums(self, dataset):
        X = np.concatenate(dataset)
        lengths = [c.shape[0] for c in dataset]
        return X, lengths

    def model_generator(self, quality, quality_train_charges_list):

        model_ = hmm.GaussianHMM(n_components=self.n_state,
                                 n_iter=self.n_iter[quality],
                                 tol=0.1,
                                 covariance_type=self.covariance)

        df_data = quality_train_charges_list
        q_data = []

        if self.sampeling == 'homogene':
            for i in range(len(df_data)):
                sub = df_data[i].values
                q_data.append(sub[[j for j in range(sub[i].shape[0]) if j % self.factor == 0]])

        elif self.sampeling == 'mean':
            for i in range(len(df_data)):
                sub = df_data[i].values
                sz = int(sub.shape[0] / self.factor) * self.factor
                q_data.append(sub[:sz].reshape(-1, self.factor, 5).mean(axis=1))

        elif self.sampeling == 'log':
            for i in range(len(df_data)):
                sub = df_data[i].values
                log_samp = self.log_sampeling(sub, int(sub.shape[0] / self.factor))
                q_data.append(log_samp)

        q_data = self.reduce_list_size(q_data, self.nb_charge_saved)

        X, lengths = self.concatenate_cepstrums(np.asarray(q_data))

        model_.fit(X, lengths=lengths)
        if self.verbose:
            print(model_.monitor_)

        return model_

    def get_all_models(self, all_quality_train):

        self.models = [self.model_generator(1, all_quality_train[1]),
                       self.model_generator(2, all_quality_train[2]),
                       self.model_generator(3, all_quality_train[3])]

        return self.models


def run_grid_search(df_train, validation_list, validation_list_hide, output_file=None):
    if output_file is not None and not os.path.isfile(output_file):
        creat_file(output_file)

    for iter_3 in range(10, 21, 2):
        for iter_2 in range(10, 21, 2):
            for iter_1 in range(10, 21, 2):
                for factor in [5, 25, 50]:
                    for nb_charges_saved in [40, 60]:
                        for n_state in [5, 6]:
                            for c in ['full', 'diag']:
                                for sampeling in ['log', 'mean']:
                                    n_iter = {1: iter_1, 2: iter_2, 3: iter_3}
                                    params = {'n_iter': n_iter, 'factor': factor, 'n_state': n_state,
                                              'sampeling': sampeling, 'nb_charge_saved': nb_charges_saved,
                                              'covariance': c}

                                    eval_loocv(params, output_file)


def eval_loocv(df_train, validation_charge_list, validation_charge_hide, params=None, output_file=None):
    scores, classifiers = eval_params(params)

    preds_all_clf = []

    labels = []

    for _hmm in classifiers:
        labels, preds = evaluate_model(_hmm.models, validation_charges_list, validation_charges_hide)
        preds_all_clf.append(preds)

    preds_valid = np.round(np.mean(preds_all_clf, axis=0))

    f1_train = np.mean(scores)
    f1_valid = f1_score(labels, preds_valid, average='macro')

    print("f1-score training set: {}".format(f1_train))
    print("f1-score validation set: {}".format(f1_valid))

    if output_file != None:
        _save(output_file, _hmm, f1_valid, f1_train)


def eval_loocv_(params=None, output_file=None):
    scores, classifiers = eval_params(params)

    all_train_charges = remove_cols(split_by_quality(get_charge_list(df_train)))
    _hmm = HMMBattery(verbose=False)

    if params is not None:
        _hmm.load_params(params)

    _hmm.get_all_models(all_train_charges)

    labels, preds = evaluate_model(_hmm.models, validation_charges_list, validation_charges_hide)
    print(confusion_matrix(labels, preds))

    f1_train = np.mean(scores)
    f1_valid = f1_score(labels, preds, average='macro')

    print("f1-score training set: {}".format(f1_train))
    print("f1-score validation set: {}".format(f1_valid))

    if output_file != None:
        _save(output_file, _hmm, f1_valid, f1_train)



if __name__ == '__main__':

    df_validation = pd.read_pickle("../mpri_challenge/validation_set.pckl")
    df_train = pd.read_pickle("../mpri_challenge/training_set.pckl")

    validation_charges_list = get_charge_list(df_validation, False, False, False)
    validation_charges_hide = []
    for charge in validation_charges_list:
        validation_charges_hide.append(remove_cols_to_df(charge))

    now = datetime.datetime.now()
    file_name = now.strftime("%d-%m-%y_search.csv")

    run_grid_search(file_name)
