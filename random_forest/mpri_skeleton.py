import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

"""
@Title: MPRI Challenge skeleton code
@Description: This file contains a skeleton of code for the MPRI Challenge 2018. This sample code provides a few sample methods to help you start
@Comments: This code is only a basic helper, you may need to change several things depending on the type of algorithm you are using! 
@Author: Simon Ruffieux (HES-SO//FR)
@Project:  MPRI Challenge 2018
"""


"""
    Go through the data of a set (train, valid or test)
    Computes and prints some statistics about the set given in parameter
"""
def print_set_stats(df_set):
    charge_counter = 0
    discharge_counter = 0
    complete_counter = 0

    distinct_batteries = df_set['battery_nb'].dropna().unique()  ##Get the list of batteries in the dataframe
    for battery_id in distinct_batteries:
        battery_df = df_set[df_set['battery_nb'] == battery_id]
        distinct_charges = df_set['charge_nb'].dropna().unique()  # seems there are some nan -> drop them
        ## Loop over each distinct charge and filter the dataframe to get only its data
        for charge_id in distinct_charges:
            # print("Battery: {} - Charge: {}".format(battery_id, charge_id))
            battery_charge_df = battery_df[battery_df['charge_nb'] == charge_id]
            if battery_charge_df.empty == False:
                charge_counter += 1
                battery_discharge_df = battery_df[battery_df['discharge_nb'] == charge_id]
                if battery_discharge_df.empty == False:
                    complete_counter += 1

        distinct_discharges = df_set['discharge_nb'].dropna().unique()  # seems there are some nan -> drop them
        for discharge_id in distinct_discharges:
            # print("Battery: {} - Charge: {}".format(battery_id, charge_id))
            battery_discharge_df = battery_df[battery_df['discharge_nb'] == discharge_id]
            if battery_discharge_df.empty == False:
                discharge_counter += 1

    print("\tThe set contains {} samples".format(len(df_set)))
    print("\tThe set contains {} batteries".format(len(distinct_batteries)))
    print("\tThe set contains {} charges".format(charge_counter))
    print("\tThe set contains {} discharge cycles".format(discharge_counter))
    print("\tThe set contains {} complete cycles".format(complete_counter))

"""
    Takes as input a dataframe containing the temporal information for a single charge item of a battery.
    Computes some features from the received dataframe and return them as a dictionary
"""
def get_features_charge_item(battery_charge_df):
    voltage_mean = np.mean(battery_charge_df['voltage_measured'])
    voltage_min = np.min(battery_charge_df['voltage_measured'])
    voltage_max = np.max(battery_charge_df['voltage_measured'])
    #TODO Compute and return additional features
    return {'voltage_mean': voltage_mean, 'voltage_max': voltage_max, 'voltage_min': voltage_min}


"""
    Application entry point
"""
def main():

    ## Load the provided pre-splitted datasets
    df_train = pd.read_pickle("datas/training_set.pckl")
    df_validation = pd.read_pickle("datas/validation_set.pckl")
    df_test = pd.read_pickle("datas/test_set.pckl")

    ## Print some statistics about the loaded sets:
    print("***************")
    print("  Statistics")
    print("***************")
    print("Training set:")
    print_set_stats(df_train)
    print("Validation set:")
    print_set_stats(df_validation)
    print("Test set:")
    print_set_stats(df_test)


    """
        Process the training set and compute some features for each battery charge cycle
        The code below provides an example for processing the data
    """
    distinct_batteries = df_train['battery_nb'].unique()  ##Get the list of batteries in the dataframe
    train_features_dict = []    ## Initialize the list of dictionaries that will contain all the features for training
    ## Loop over each distinct battery and filter the dataframe to get only its data
    for battery_id in distinct_batteries:
        battery_df = df_train[df_train['battery_nb'] == battery_id]
        distinct_charges = df_train['charge_nb'].dropna().unique()  # seems there are some nan -> drop them
        ## Loop over each distinct charge and filter the dataframe to get only its data
        for charge_id in distinct_charges:
            ## Filter the dataframe to get only part related to the considered charge
            battery_charge_df = battery_df[battery_df['charge_nb'] == charge_id]
            if battery_charge_df.empty == True:
                #("  Skip this one as there are no data (empty dataframe)")
                continue
            ## Get the label
            quality = int(round(np.nanmean(battery_charge_df['quality']), 0))
            ## Do the feature extraction on battery_charge_df
            features = get_features_charge_item(battery_charge_df)
            features['label'] = quality  ## Add the label back
            train_features_dict.append(features)

    ## Transform the list of dict into a dataframe (train_features_df)
    train_features_df = pd.DataFrame(train_features_dict)

    ## Split dataframe into labels and features for trainign algorithm
    labels = train_features_df['label']
    features = train_features_df.drop('label', axis=1)

    """
        Declare the algorithm model and train/evaluate it
        Pay attention, we have a leave-one-out scheme !
        
        Notes
         - Try to search for best hyper-parameters of the algorithm
         - Generate and plot the confusion matrix for your presentation/report
         - For a more advanced evaluation, you could try rotating the batteries used for learning and validation in order to get a better prediction of the accuracy on the test set        
    """

    #TODO Algorithm training and evaluation methodology
    # model = YourAlgorithmModel()
    # model.fit()

    """
        Do the prediction on the test set using your trained algorithm and save them as a csv file
        (Only do that once you achieved the best results according to your evaluation)
    """
    test_set = pd.read_pickle("datas/test_set.pckl")
    ## Process the test set and compute its features (!No quality!)
    #results = model.predict(test_set_features)
    results = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1]) ## You results should look like that (a numpy array containing 31 integers).
    # Save your results
    np.savetxt("datas/RF_Teachers_Ruffieux.csv", results.astype(int), fmt='%i')


if __name__ == '__main__':
    main()