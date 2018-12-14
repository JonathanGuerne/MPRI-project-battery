import datetime

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.metrics import f1_score, confusion_matrix

from hmm_clf.data_tools import get_charge_list, remove_cols_to_df, battery_list, shuffle_df, split_training_validation, \
    split_by_quality, remove_cols, add_features, normalize
from hmm_clf.file_tools import _save


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
                                 tol=0.001,
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
                q_data.append(sub[:sz].reshape(-1, self.factor, 6).mean(axis=1))

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


         # self.nb_charge_saved = np.min([len(value) for key, value in all_quality_train.items()])

        self.models = [self.model_generator(1, all_quality_train[1]),
                       self.model_generator(2, all_quality_train[2]),
                       self.model_generator(3, all_quality_train[3])]

        return self.models

def test(dataset, models, verbose=False):
    X, lengths = dataset, [len(dataset)]
    scores = [m.score(X, lengths) for m in models]
    if verbose:
        print(f"Best model: {np.argmax(scores) + 1}")
    return scores


def evaluate_model(models, valid_data, valid_hide, verbose=False):
    labels = []
    predictions = []

    for i in range(len(valid_data)):
        label = round(valid_data[i]['quality'].mean())

        scores = test(valid_hide[i], models, verbose=verbose)
        prediction = np.argmax(scores) + 1

        labels.append(label)
        predictions.append(prediction)

    return labels, predictions

def eval_params(df_train, params, verbose=False):
    _f1_scores = []
    _weights = []

    fulltrain_charges_list = get_charge_list((df_train), False, False, False)

    for battery_number in battery_list(df_train):


        train_charges, valid_charges = split_training_validation(fulltrain_charges_list, battery_number)
        __qal_train_charges = split_by_quality(train_charges)
        qal_train_charges = remove_cols(__qal_train_charges)



        valid_charges_hide = []
        for charge in valid_charges:
            valid_charges_hide.append(remove_cols_to_df(charge))

        hmmB = HMMBattery(verbose=False)
        if params is not None:
            hmmB.load_params(params)
        hmmB.get_all_models(qal_train_charges)

        labels, preds = evaluate_model(hmmB.models, valid_charges, valid_charges_hide)

        _f1_scores.append(f1_score(labels, preds, average='macro'))
        _weights.append(len(valid_charges))

        if verbose:
            print(confusion_matrix(labels, preds))
            print(_f1_scores[-1])


    t = np.sum(np.multiply(_f1_scores,_weights))/sum(_weights)

    #_f1_scores = [_weights[i]*_f1_scores[i] for i in range(len(_f1_scores))]

    return t

def eval_loocv(df_train, validation_charges_list, validation_charges_hide, params=None, output_file=None, verbose=False):
    scores = eval_params(df_train, params, verbose=verbose)

    preds_all_clf = []

    train_charges = get_charge_list(df_train)
    __qal_train_charges = split_by_quality(train_charges)
    qal_train_charges = remove_cols(__qal_train_charges)

    hmmB = HMMBattery(verbose=False)
    if params is not None:
        hmmB.load_params(params)
    hmmB.get_all_models(qal_train_charges)

    labels, preds_valid = evaluate_model(hmmB.models, validation_charges_list, validation_charges_hide)

    f1_train = scores
    f1_valid = f1_score(labels, preds_valid, average='macro')

    print("f1-score training set: {}".format(f1_train))
    print("f1-score validation set: {}".format(f1_valid))

    print(confusion_matrix(labels,preds_valid))

    if output_file != None:
        _save(output_file, hmmB, f1_valid, f1_train)

    del hmmB
    del preds_all_clf
    del labels

    return f1_train

if __name__ == '__main__':

    df_validation = pd.read_pickle("../mpri_challenge/validation_set.pckl")
    df_train = pd.read_pickle("../mpri_challenge/training_set.pckl")



    df_validation = add_features(df_validation)
    df_validation = normalize(df_validation)

    df_train = add_features(df_train)
    df_train = normalize(df_train)


    validation_charges_list = get_charge_list(df_validation, False, False, False)
    validation_charges_hide = []

    for charge in validation_charges_list:
        validation_charges_hide.append(remove_cols_to_df(charge))

    now = datetime.datetime.now()
    file_name = now.strftime("%d-%m-%y_50_charges_70_factor.csv")

    params = {'n_iter': {1: 16, 2: 16, 3: 18},
           'factor': 70,
           'n_state': 6,
           'sampeling': 'mean',
           'nb_charge_saved': 53,
           'covariance': 'diag'}


    eval_loocv(df_train,validation_charges_list,validation_charges_hide,params, verbose=True)

    # from hmm_clf.Grid_search import run_grid_search
    # best_params = run_grid_search(df_train, validation_charges_list, validation_charges_hide, file_name, verbose=True)
    # print(best_params)
