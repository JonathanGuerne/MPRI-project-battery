
from autologging import logged
import logging, sys

@logged
class HMM_Classifier:

    _classifier_version = 0.0

    @property
    def classifier_version(self):
        return self._classifier_version


    def __init__(self):
        logging.basicConfig(level=logging.INFO,
            filename='Logs/HMM-ClassifierV{}.log'.format(self._classifier_version),
            format="%(asctime)s %(levelname)s:%(name)s:%(funcName)s:%(message)s")


    def load_data(self):
        self.__log.info("Loading data")

if __name__ == "__main__":

    hmm = HMM_Classifier()
    hmm.load_data()