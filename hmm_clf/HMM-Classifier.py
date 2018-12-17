
from autologging import logged
import logging, sys
import pandas as pd

@logged
class HMM_Classifier:

    def __init__(self):
        
        self.classifier_version = 0.0
        
        logging.basicConfig(
            level=logging.INFO,
            filename='Logs/HMM-ClassifierV{}.log'
                .format(self.classifier_version),
            format="%(asctime)s|%(levelname)s|%(name)s|%(funcName)s|%(message)s")

    def load_data(self):
        self.df_train = pd.read_pickle("datas/training_set.pckl")
        self.df_validation = pd.read_pickle("datas/validation_set.pckl")
        self.df_test = pd.read_pickle("datas/test_set.pckl")

        self.__log.info('Data load OK')





if __name__ == "__main__":

    hmm = HMM_Classifier()
    hmm.load_data()