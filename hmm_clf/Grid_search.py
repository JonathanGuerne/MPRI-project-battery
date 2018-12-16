import os

import numpy as np

from hmm_clf.HMM_as_script import eval_loocv
from hmm_clf.data_tools import shuffle_df
from hmm_clf.file_tools import creat_file


def run_grid_search(df_train, validation_charges_list, validation_charges_hide, output_file=None, verbose=False):
    if output_file is not None and not os.path.isfile(output_file):
        creat_file(output_file)

    best_f1 = 0
    best_params = None

    for iter_3 in range(5,9):
        for iter_2 in range(5,9):
            for iter_1 in range(5,9):
                for n_state in [6]:
                    for c in ['diag']:#['full', 'diag', 'tied']:
                        for sampeling in ['None']:#['log', 'mean', 'homogene']:
                            for factor in [70]:
                                for nb_charges_saved in [80]:

                                    f1_mean = []

                                    n_iter = {1: iter_1, 2: iter_2, 3: iter_3}
                                    params = {'n_iter': n_iter, 'factor': factor, 'n_state': n_state,
                                              'sampeling': sampeling, 'nb_charge_saved': nb_charges_saved,
                                              'covariance': c}

                                    for _ in range(1):

                                        try:
                                            print('set of params {}'.format(params))
                                            f1 = eval_loocv(shuffle_df(df_train), validation_charges_list,
                                                            validation_charges_hide,
                                                            params,
                                                            output_file, verbose=verbose)
                                            f1_mean.append(f1)
                                        except ValueError as e:
                                            print('Error with set of params : {}'.format(e))

                                    f1_mean = np.mean(f1_mean)
                                    if f1_mean > best_f1:
                                        best_f1 = f1_mean
                                        best_params = params
    return best_params
