from helper_functions import *
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from numpy import array,reshape

import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':

    encoder_path = 'Models/autoencoder_trad.h5'
    encoder = load_model(encoder_path)

    window_size = 50
    limit = 10**3

    #training_data_path_trial_1 = 'Raw Mat Files (CLA)/CLASubjectB1510193StLRHand.mat'
    #training_data_path_trial_2 = 'Raw Mat Files (CLA)/CLASubjectB1510193StLRHand.mat'
    testing_data_path_same_patient = 'Raw Mat Files (CLA)/CLASubjectE1601193StLRHand.mat'

    #training_set_1 = parse_dataset(training_data_path_trial_1, window_size, limit)
    #training_set_2 = parse_dataset(training_data_path_trial_2, window_size, limit)

    same_patient_testing_set = parse_dataset(testing_data_path_same_patient,window_size,limit)

    X_test = list(StandardScaler().fit_transform(flattenMatrixDataset(array(same_patient_testing_set[0]))))
    y_test = same_patient_testing_set[1]

    #X_1 = list(StandardScaler().fit_transform(flattenMatrixDataset(array(training_set_1[0]))))
    #X_2 = list(StandardScaler().fit_transform(flattenMatrixDataset(array(training_set_2[0]))))
    #y_1 = list(training_set_1[1])
    #y_2 = list(training_set_2[1])


    #X = array(X_1 + X_2)
    #y = y_1 + y_2

    X_1_0 = []
    X_1_1 = []

    X_2_0 = []
    X_2_1 = []

    X_3_0 = []
    X_3_1 = []

    for i in range(len(X_test)):
        X_reduced = encoder.predict(reshape(X_test[i],[1,2200]))

        if y_test[i] == 0:
            X_1_0.append(X_reduced[0][0])
            X_1_1.append(X_reduced[0][1])

        if y_test[i] == 1:
            X_2_0.append(X_reduced[0][0])
            X_2_1.append(X_reduced[0][1])

        if y_test[i] == 2:
            X_3_0.append(X_reduced[0][0])
            X_3_1.append(X_reduced[0][1])

    plt.scatter(X_1_0,X_1_1,c='b')
    plt.scatter(X_2_0, X_2_1, c='g')
    plt.scatter(X_3_0, X_3_1, c='r')
    plt.legend(['Left Hand', 'Right Hand', 'Neutral'], loc='upper right')
    plt.show()

