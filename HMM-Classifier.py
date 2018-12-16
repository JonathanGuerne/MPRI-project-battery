
from autologging import logged
import logging, sys
import pandas as pd
import numpy as np
from hmmlearn import hmm

@logged
class HMM_Classifier:

    def __init__(self, samp_per_charge=10, nb_charge_saved=10, nb_states=7,
     factor=6, verbose=False):
        
        self.classifier_version = 0.0
        
        logging.basicConfig(
            level=logging.INFO,
            filename='Logs/HMM-ClassifierV{}.log'
                .format(self.classifier_version),
            format="%(asctime)s|%(levelname)s|%(name)s|%(funcName)s|%(message)s")

        self.n_state = nb_states
        self.n_iter = {1:10, 2:10, 3:10}
        self.samp_per_charge = samp_per_charge
        self.nb_charge_saved = nb_charge_saved
        self.factor = factor
        self.verbose = verbose


    def load_data(self):
        self.df_train = pd.read_pickle("mpri_challenge/training_set.pckl")
        self.df_valid = pd.read_pickle("mpri_challenge/validation_set.pckl")
        self.df_test = pd.read_pickle("mpri_challenge/test_set.pckl")

        self.__log.info('Data load OK')

    def reduce_list_size(self, _list, nb_slice):
        ids = [i * int(len(_list)/nb_slice) for i in range(0,nb_slice,1)]
        return [_list[i] for i in ids]
    
    def concatenate_cepstrums(self, dataset):
        X = np.concatenate(dataset)
        lengths = [c.shape[0] for c in dataset]
        return X, lengths
    
    def model_generator(self, quality):
    
        model_ = hmm.GaussianHMM(n_components=self.n_state, n_iter=self.n_iter[quality])
        q_data = self.get_quality_charge_list(quality)
    
        # todo : reduce q_data size
        q_data = [self.reduce_list_size(q_data[i],self.samp_per_charge) for i in range(len(q_data))]
        # for i in range(len(q_data)):
        #     sub = q_data[i]
        #     q_data[i] = sub[[j for j in range(q_data[i].shape[0]) if j % self.factor == 0]]
        
        X, lengths = self.concatenate_cepstrums(np.asarray(q_data))
        q_data = self.reduce_list_size(q_data, self.nb_charge_saved)
    
        model_.fit(X, lengths=lengths)
        if self.verbose:
            print(model_.monitor_)
    
        return model_
    
    def get_all_models(self):
        return [self.model_generator(1),self.model_generator(2),
        self.model_generator(3)]

    def get_quality_charge_list(self, quality, train=True, drop_battery=True, isdebug=False):
        if train:
            test_battery = self.df_train[['battery_nb','charge_nb','voltage_measured','current_measured','temperature_measured','current_charge','voltage_charge','quality']]
        else:
            test_battery = self.df_valid[['battery_nb','charge_nb','voltage_measured','current_measured','temperature_measured','current_charge','voltage_charge','quality']]
        
        qual_ = test_battery.loc[test_battery['quality'] == quality]

        return self.get_charge_list(qual_)

    def get_charge_list(self, df, drop_battery=True, isdebug=False):
        if 'quality' in df.columns:
            df = df.drop(['quality'], axis=1)
        
        lst_battery = np.unique(df['battery_nb'])
        tmp_lst = []
        
        for battery in lst_battery:
            
            battery_df = df.loc[df['battery_nb'] == battery]
            
            if drop_battery:
                battery_df = battery_df.drop(['battery_nb'], axis=1)
            
            lst_charges = np.unique(battery_df['charge_nb'])
            lst_charges = lst_charges[~np.isnan(lst_charges)]

            if isdebug is True:
                print(lst_charges)
            
            for charge in lst_charges:
                vals = battery_df.loc[battery_df['charge_nb'] == charge].values
                tmp_lst.append(vals[~np.isnan(vals).any(axis=1)])

        return tmp_lst

    def test(self, dataset, models, verbose=False):
        X, lengths = dataset, [len(dataset)]    
        scores = [m.score(X, lengths) for m in models]
        if verbose:
            print(f"Best model: {np.argmax(scores) + 1}")
        return scores

    def evaluate(self):
        lst_charges = np.unique(self.df_valid['charge_nb'])
        lst_charges = lst_charges[~np.isnan(lst_charges)]

        labels = []
        predictions = []

        sub_valid = self.df_valid[['battery_nb','charge_nb','voltage_measured','current_measured','temperature_measured','current_charge','voltage_charge','quality']]

        for charge in lst_charges:
            data_validation = sub_valid.loc[sub_valid['charge_nb'] == charge]
            label = round(data_validation['quality'].mean())  

            charge_as_list = self.get_charge_list(data_validation)[0]
            scores = self.test(charge_as_list, self.get_all_models())
            prediction = np.argmax(scores) + 1

            labels.append(label)
            predictions.append(prediction)

        self.__log.info('Evaluate model')
        return labels, predictions

if __name__ == "__main__":

    hmm_b = HMM_Classifier()
    hmm_b.load_data()

    lst_samp_per_charge = [10,25,250]
    lst_nb_charge_saved = [10,25,250]
    lst_states = [5,7,10]

    b_preds = None
    b_labels = None
    best_f1 = 0

    for state in lst_states:
        for samp_per_charge in lst_samp_per_charge:
            for nb_charge_saved in lst_nb_charge_saved:
                hmm_b = HMM_Battery(nb_charge_saved=nb_charge_saved,samp_per_charge=samp_per_charge,nb_states=state)
                labels, preds = evaluate(hmm_b.get_all_models())

                _f1_score = f1_score(labels,preds,average='macro') 

                if (_f1_score > best_f1):
                    best_classifier = hmm_b
                    b_labels = labels
                    b_preds = preds

            print('-'*20)
            print('nb states = {}'.format(state))
            print('samp_per_charg = {}'.format(samp_per_charge))
            print('nb_charge_saved = {}'.format(nb_charge_saved))
            print('f1-score = {}'.format(f1_score(labels,preds,average='macro'))) 