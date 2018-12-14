import csv


def creat_file(output_file):
    with open(output_file,'w', newline='') as fd:
        out = ['nb_states','nb_charges_saved','factor','iter_1','iter_2',
               'iter_3','f1_score_valid','f1_score_train','sampeling','activation']
        writer = csv.writer(fd)
        writer.writerow(out)

def _save(output_file, clf, f1_valid, f1_train):
    with open(output_file,'a',newline='') as fd:
        out = [clf.n_state,
               clf.nb_charge_saved,
               clf.factor,
               clf.n_iter[1],
               clf.n_iter[2],
               clf.n_iter[3],
               f1_valid,
               f1_train,
              clf.sampeling,
              clf.covariance]
        writer = csv.writer(fd)
        writer.writerow(out)