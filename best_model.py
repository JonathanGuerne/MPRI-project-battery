from random_forest.random_forest import RandomForest
from SVM.svm import SVM
# from hmm_clf.data_tools import get_charge_list, remove_cols_to_df, add_features, gradient
from hmm_clf.HMM_as_script import predict, eval_loocv, validation, validation_prediction_hmm
import pandas as pd
import numpy as np
import datetime

class Fusion:

    def __init__(self, df_train, df_validation, df_test):
        self.df_train = df_train
        self.df_validation = df_validation
        self.df_test = df_test
        self.result_dict = {}

    def run_hmm(self):
        predictions, f1_validation, f1_test = validation_prediction_hmm()
        self.result_dict['hmm'] = [predictions, np.mean([f1_validation, f1_test])]

    def run_rf(self):
        param_grid = {
            "n_estimators": [5, 10, 20, 50, 75, 100, 300],
            "max_depth": [2, 3, 5, 10, None],
            'class_weight': ['balanced', 'balanced_subsample'],
            'max_features': ['auto', 'sqrt', 'log2']
        }
        rf = RandomForest(param_grid, self.df_train, self.df_validation, self.df_test, log=False, include_discharges=True)
      #  rf.model.grid_search_fit()
        _, f1_validation, f1_test = rf.model.leave_one_out()
        prediction  = rf.model.predict_test()
        self.result_dict['rf'] = [prediction, np.mean([f1_validation,f1_test])]

    def run_svm(self):
        param_grid = {
            "C": [10, 100],
            "kernel": ['rbf', 'sigmoid'],
            "coef0": [0.0],
            'gamma': ['auto', 0.000001],
            'decision_function_shape': ['ovo', 'ovr']
        }
        svm = SVM(param_grid, self.df_train, self.df_validation, self.df_test, log=False, include_discharges=True)
     #   svm.model.grid_search_fit()
        _, f1_validation, f1_test = svm.model.leave_one_out()
        prediction = svm.model.predict_test()
        self.result_dict['svm'] = [prediction,np.mean([f1_test,f1_validation])]

    def define_best(self):
        list_models = ['hmm', 'rf', 'svm']
        self.run_hmm()
        self.run_rf()
        self.run_svm()

        pred_size = len(self.result_dict['svm'][0])

        fusion_preds = []
        for i in range(pred_size):

            print("value {}".format(i+1))
            vote = {1:0,2:0,3:0}

            for m in list_models:
                pred = self.result_dict[m][0][i]
                print("model {} :  {}".format(m,pred))
                vote[int(pred)] += self.result_dict[m][1]
            print("final decision {}".format(max(vote, key=vote.get)))
            fusion_preds.append(max(vote, key=vote.get))

        results = np.array(fusion_preds)
        # Save your results
        np.savetxt("datas/Fusion_2BTree_Fusion.csv", results.astype(int), fmt='%i')

        # for m1 in list_models:
        #     current_model = self.result_dict[m1]
        #
        #     for m2 in list_models:
        #         if m2 is not current_model:
        #             if abs(self.result_dict[current_model][1] - self.result_dict[m2][1]) < 10:
        #                 print(m2)
        #                 print(current_model)


if __name__ == '__main__':
    df_validation = pd.read_pickle("datas/validation_set.pckl")
    df_train = pd.read_pickle("datas/training_set.pckl")
    df_test = pd.read_pickle("datas/test_set.pckl")

    fusion = Fusion(df_train, df_validation, df_test)
    fusion.define_best()
