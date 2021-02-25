### this is for training a single generator for all labels ###
""" with the analysis of """
### weights = weights + N(0, sigma**2*(sqrt(2)/N)**2)
### columns of mean embedding = raw + N(0, sigma**2*(2/N)**2)

import numpy as np
# import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import util
import random
import socket
from sdgym import load_dataset
import argparse
import sys


import pandas as pd
import seaborn as sns
sns.set()
# %matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import  LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import xgboost

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterGrid
from autodp import privacy_calibrator
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score

from autodp import privacy_calibrator

import warnings
warnings.filterwarnings('ignore')

import os

#Results_PATH = "/".join([os.getenv("HOME"), "condMMD/"])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# 0 - LogisticRegression
# 1 - GaussianNB
# 2 - BernoulliNB
# 3 - LinearSVC
# 4 - DecisionTreeClassifier
# 5 - LinearDiscriminantAnalysis
# 6 - AdaBoostClassifier
# 7 - BaggingClassifier
# 8 - RandomForestClassifier
# 9 - GradientBoostingClassifier
# 10 - MLP
# 11 - XGBoost


args=argparse.ArgumentParser()
args.add_argument("--dataset", default="covtype")
args.add_argument("--private", type=int, default=0)
#args.add_argument("--epochs", nargs='+', type=int, default=[8000, 6000, 2000, 1000, 4000, 10000])
#args.add_argument("--epochs", default='200,2000,8000,6000,1000,4000,10000')
args.add_argument("--epochs", default='200')
args.add_argument("--batch", type=float, default=0.1)
#args.add_argument("--num_features", nargs='+', type=int, default=[500, 1000, 2000, 5000, 10000, 50000, 80000, 100000])
#args.add_argument("--num_features", default='500,1000,2000,5000,10000')
args.add_argument("--num_features", default='2000')

args.add_argument("--undersample", type=float, default=1)
args.add_argument("--repeat", type=int, default=2)
#args.add_argument('--classifiers', nargs='+', type=int, help='list of integers', default=[2])
args.add_argument('--classifiers', nargs='+', type=int, help='list of integers', default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
args.add_argument("--data_type", default='generated') #both, real, generated
arguments=args.parse_args()
print("arg", arguments)
list_epochs=[int(i) for i in arguments.epochs.split(',')]
list_num_features=[int(i) for i in arguments.num_features.split(',')]

############################### kernels to use ###############################
""" we use the random fourier feature representation for Gaussian kernel """

def RFF_Gauss(n_features, X, W):
    """ this is a Pytorch version of Wittawat's code for RFFKGauss"""

    W = torch.Tensor(W).to(device)
    X = X.to(device)

    XWT = torch.mm(X, torch.t(W)).to(device)
    Z1 = torch.cos(XWT)
    Z2 = torch.sin(XWT)

    Z = torch.cat((Z1, Z2),1) * torch.sqrt(2.0/torch.Tensor([n_features])).to(device)
    return Z


""" we use a weighted polynomial kernel for labels """

def Feature_labels(labels, weights):

    weights = torch.Tensor(weights)
    weights = weights.to(device)

    labels = labels.to(device)

    weighted_labels_feature = labels/weights

    return weighted_labels_feature


############################### end of kernels ###############################

############################### generative models to use ###############################
""" two types of generative models depending on the type of features in a given dataset """

class Generative_Model_homogeneous_data(nn.Module):

        def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, dataset):
            super(Generative_Model_homogeneous_data, self).__init__()

            self.input_size = input_size
            self.hidden_size_1 = hidden_size_1
            self.hidden_size_2 = hidden_size_2
            self.output_size = output_size

            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size_1)
            self.bn1 = torch.nn.BatchNorm1d(self.hidden_size_1)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(self.hidden_size_1, self.hidden_size_2)
            self.bn2 = torch.nn.BatchNorm1d(self.hidden_size_2)
            self.fc3 = torch.nn.Linear(self.hidden_size_2, self.output_size)

            self.dataset = dataset


        def forward(self, x):
            hidden = self.fc1(x)
            relu = self.relu(self.bn1(hidden))
            output = self.fc2(relu)
            output = self.relu(self.bn2(output))
            output = self.fc3(output)

            # if self.dataset=='credit':
            #     all_pos = self.relu(output[:,-1])
            #     output = torch.cat((output[:,:-1], all_pos[:,None]),1)

            return output


class Generative_Model_heterogeneous_data(nn.Module):

            def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, num_categorical_inputs, num_numerical_inputs):
                super(Generative_Model_heterogeneous_data, self).__init__()

                self.input_size = input_size
                self.hidden_size_1 = hidden_size_1
                self.hidden_size_2 = hidden_size_2
                self.output_size = output_size
                self.num_numerical_inputs = num_numerical_inputs
                self.num_categorical_inputs = num_categorical_inputs

                self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size_1)
                self.bn1 = torch.nn.BatchNorm1d(self.hidden_size_1)
                self.relu = torch.nn.ReLU()
                self.fc2 = torch.nn.Linear(self.hidden_size_1, self.hidden_size_2)
                self.bn2 = torch.nn.BatchNorm1d(self.hidden_size_2)
                self.fc3 = torch.nn.Linear(self.hidden_size_2, self.output_size)
                self.sigmoid = torch.nn.Sigmoid()

            def forward(self, x):
                hidden = self.fc1(x)
                relu = self.relu(self.bn1(hidden))
                output = self.fc2(relu)
                output = self.relu(self.bn2(output))
                output = self.fc3(output)

                output_numerical = self.relu(output[:, 0:self.num_numerical_inputs])  # these numerical values are non-negative
                output_categorical = self.sigmoid(output[:, self.num_numerical_inputs:])
                output_combined = torch.cat((output_numerical, output_categorical), 1)

                return output_combined

############################### end of generative models ###############################



def undersample(raw_input_features, raw_labels, undersampled_rate):
    """ we take a pre-processing step such that the dataset is a bit more balanced """
    idx_negative_label = raw_labels == 0
    idx_positive_label = raw_labels == 1

    pos_samps_input = raw_input_features[idx_positive_label, :]
    pos_samps_label = raw_labels[idx_positive_label]
    neg_samps_input = raw_input_features[idx_negative_label, :]
    neg_samps_label = raw_labels[idx_negative_label]

    # take random 10 percent of the negative labelled data
    in_keep = np.random.permutation(np.sum(idx_negative_label))
    under_sampling_rate = undersampled_rate  # 0.4
    in_keep = in_keep[0:np.int(np.sum(idx_negative_label) * under_sampling_rate)]

    neg_samps_input = neg_samps_input[in_keep, :]
    neg_samps_label = neg_samps_label[in_keep]

    feature_selected = np.concatenate((pos_samps_input, neg_samps_input))
    label_selected = np.concatenate((pos_samps_label, neg_samps_label))

    return feature_selected, label_selected






######################################## beginning of main script ############################################################################

def main(dataset, undersampled_rate, n_features_arg, mini_batch_size_arg, how_many_epochs_arg, is_priv_arg, seed_number):

    random.seed(seed_number)

    is_private = is_priv_arg

    ############################### data loading ##################################

    if dataset=='epileptic':
        print("epileptic seizure recognition dataset") # this is homogeneous

        if 'g0' not in socket.gethostname() and 'p0' not in socket.gethostname():
            data = pd.read_csv("../data/Epileptic/data.csv")
        else:
            # (1) load data
            data = pd.read_csv('/home/user/Dropbox_from/Current_research/privacy/DPDR/data/Epileptic/data.csv')

        feature_names = data.iloc[:, 1:-1].columns
        target = data.iloc[:, -1:].columns

        data_features = data[feature_names]
        data_target = data[target]

        for i, row in data_target.iterrows():
          if data_target.at[i,'y']!=1:
            data_target.at[i,'y'] = 0

        ###################

        raw_labels=np.array(data_target)
        raw_input_features=np.array(data_features)

        idx_negative_label = raw_labels == 0
        idx_positive_label = raw_labels == 1

        idx_negative_label=idx_negative_label.squeeze()
        idx_positive_label=idx_positive_label.squeeze()

        pos_samps_input = raw_input_features[idx_positive_label, :]
        pos_samps_label = raw_labels[idx_positive_label]
        neg_samps_input = raw_input_features[idx_negative_label, :]
        neg_samps_label = raw_labels[idx_negative_label]

        # take random 10 percent of the negative labelled data
        in_keep = np.random.permutation(np.sum(idx_negative_label))
        under_sampling_rate = undersampled_rate  # 0.01
        # under_sampling_rate = 0.3
        in_keep = in_keep[0:np.int(np.sum(idx_negative_label) * under_sampling_rate)]

        neg_samps_input = neg_samps_input[in_keep, :]
        neg_samps_label = neg_samps_label[in_keep]

        feature_selected = np.concatenate((pos_samps_input, neg_samps_input))
        label_selected = np.concatenate((pos_samps_label, neg_samps_label))

        label_selected=label_selected.squeeze()
        ####


        X_train, X_test, y_train, y_test = train_test_split(feature_selected, label_selected, train_size=0.70, test_size=0.30, random_state=seed_number)

        #X_train, X_test, y_train, y_test = train_test_split(data_features, data_target, train_size=0.70, test_size=0.30, random_state=seed_number)

        # unpack data
        #X_train = X_train.values
        #y_train = y_train.values.ravel()
        #X_test=np.array(X_test)
        #y_test=np.array(y_test)
        n_classes = 2


    elif dataset=="credit":

        print("Creditcard fraud detection dataset") # this is homogeneous

        if 'g0' not in socket.gethostname() and 'p0' not in socket.gethostname():
            data = pd.read_csv("../data/Kaggle_Credit/creditcard.csv")
        else:
            # (1) load data
            data = pd.read_csv(
                '/home/user/Dropbox_from/Current_research/privacy/DPDR/data/Kaggle_Credit/creditcard.csv')

        feature_names = data.iloc[:, 1:30].columns
        target = data.iloc[:, 30:].columns

        data_features = data[feature_names]
        data_target = data[target]
        print(data_features.shape)

        """ we take a pre-processing step such that the dataset is a bit more balanced """
        raw_input_features = data_features.values
        raw_labels = data_target.values.ravel()

        idx_negative_label = raw_labels == 0
        idx_positive_label = raw_labels == 1

        pos_samps_input = raw_input_features[idx_positive_label, :]
        pos_samps_label = raw_labels[idx_positive_label]
        neg_samps_input = raw_input_features[idx_negative_label, :]
        neg_samps_label = raw_labels[idx_negative_label]

        # take random 10 percent of the negative labelled data
        in_keep = np.random.permutation(np.sum(idx_negative_label))
        under_sampling_rate = undersampled_rate #0.01
        # under_sampling_rate = 0.3
        in_keep = in_keep[0:np.int(np.sum(idx_negative_label) * under_sampling_rate)]

        neg_samps_input = neg_samps_input[in_keep, :]
        neg_samps_label = neg_samps_label[in_keep]

        feature_selected = np.concatenate((pos_samps_input, neg_samps_input))
        label_selected = np.concatenate((pos_samps_label, neg_samps_label))

        X_train, X_test, y_train, y_test = train_test_split(feature_selected, label_selected, train_size=0.80,
                                                            test_size=0.20, random_state=seed_number)
        n_classes = 2


    elif dataset=='census':

        print("census dataset") # this is heterogenous

        print(socket.gethostname())
        if 'g0' not in socket.gethostname() and 'p0' not in socket.gethostname():
            data = np.load("../data/real/census/train.npy")
        else:
            data = np.load(
                "/home/user/Dropbox_from/Current_research/privacy/DPDR/data/real/census/train.npy")

        numerical_columns = [0, 5, 16, 17, 18, 29, 38]
        ordinal_columns = []
        categorical_columns = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                               30, 31, 32, 33, 34, 35, 36, 37, 38, 40]
        n_classes = 2

        data = data[:, numerical_columns + ordinal_columns + categorical_columns]

        num_numerical_inputs = len(numerical_columns)
        num_categorical_inputs = len(categorical_columns + ordinal_columns) - 1

        raw_input_features = data[:, :-1]
        raw_labels = data[:, -1]
        print('raw input features', raw_input_features.shape)

        """ we take a pre-processing step such that the dataset is a bit more balanced """
        idx_negative_label = raw_labels == 0
        idx_positive_label = raw_labels == 1

        pos_samps_input = raw_input_features[idx_positive_label, :]
        pos_samps_label = raw_labels[idx_positive_label]
        neg_samps_input = raw_input_features[idx_negative_label, :]
        neg_samps_label = raw_labels[idx_negative_label]

        # take random 10 percent of the negative labelled data
        in_keep = np.random.permutation(np.sum(idx_negative_label))
        under_sampling_rate = undersampled_rate #0.4
        in_keep = in_keep[0:np.int(np.sum(idx_negative_label) * under_sampling_rate)]

        neg_samps_input = neg_samps_input[in_keep, :]
        neg_samps_label = neg_samps_label[in_keep]

        feature_selected = np.concatenate((pos_samps_input, neg_samps_input))
        label_selected = np.concatenate((pos_samps_label, neg_samps_label))

        X_train, X_test, y_train, y_test = train_test_split(feature_selected, label_selected, train_size=0.80,
                                                            test_size=0.20, random_state=seed_number)


    elif dataset=='cervical':

        print("dataset is", dataset) # this is heterogenous
        print(socket.gethostname())
        if 'g0' not in socket.gethostname() and 'p0' not in socket.gethostname():
            df = pd.read_csv("../data/Cervical/kag_risk_factors_cervical_cancer.csv")
        else:
            df = pd.read_csv(
                "/home/user/Dropbox_from/Current_research/privacy/DPDR/data/Cervical/kag_risk_factors_cervical_cancer.csv")
            print("Loaded Cervical")

        # df.head()

        df_nan = df.replace("?", np.float64(np.nan))
        df_nan.head()

        #df1 = df_nan.convert_objects(convert_numeric=True)
        df1=df.apply(pd.to_numeric, errors="coerce")

        df1.columns = df1.columns.str.replace(' ', '')  # deleting spaces for ease of use

        """ this is the key in this data-preprocessing """
        df = df1[df1.isnull().sum(axis=1) < 10]

        numerical_df = ['Age', 'Numberofsexualpartners', 'Firstsexualintercourse', 'Numofpregnancies', 'Smokes(years)',
                        'Smokes(packs/year)', 'HormonalContraceptives(years)', 'IUD(years)', 'STDs(number)',
                        'STDs:Numberofdiagnosis',
                        'STDs:Timesincefirstdiagnosis', 'STDs:Timesincelastdiagnosis']
        categorical_df = ['Smokes', 'HormonalContraceptives', 'IUD', 'STDs', 'STDs:condylomatosis',
                          'STDs:vulvo-perinealcondylomatosis', 'STDs:syphilis', 'STDs:pelvicinflammatorydisease',
                          'STDs:genitalherpes', 'STDs:AIDS', 'STDs:cervicalcondylomatosis',
                          'STDs:molluscumcontagiosum', 'STDs:HIV', 'STDs:HepatitisB', 'STDs:HPV',
                          'Dx:Cancer', 'Dx:CIN', 'Dx:HPV', 'Dx', 'Hinselmann', 'Schiller', 'Citology', 'Biopsy']

        feature_names = numerical_df + categorical_df[:-1]
        num_numerical_inputs = len(numerical_df)
        num_categorical_inputs = len(categorical_df[:-1])

        for feature in numerical_df:
            # print(feature, '', df[feature].convert_objects(convert_numeric=True).mean())
            feature_mean = round(df[feature].median(), 1)
            df[feature] = df[feature].fillna(feature_mean)

        for feature in categorical_df:
            #df[feature] = df[feature].convert_objects(convert_numeric=True).fillna(0.0)
            df[feature] = df[feature].fillna(0.0)


        target = df['Biopsy']
        # feature_names = df.iloc[:, :-1].columns
        inputs = df[feature_names]
        print('raw input features', inputs.shape)

        # X_train, X_test, y_train, y_test = train_test_split(inputs, target, train_size=0.90, test_size=0.10,
        #                                                     random_state=seed_number)
        #
        #
        # y_train = y_train.values.ravel()  # X_train_pos
        # X_train = X_train.values
        n_classes = 2

        raw_input_features = inputs.values
        raw_labels = target.values.ravel()

        print('raw input features', raw_input_features.shape)

        """ we take a pre-processing step such that the dataset is a bit more balanced """
        idx_negative_label = raw_labels == 0
        idx_positive_label = raw_labels == 1

        pos_samps_input = raw_input_features[idx_positive_label, :]
        pos_samps_label = raw_labels[idx_positive_label]
        neg_samps_input = raw_input_features[idx_negative_label, :]
        neg_samps_label = raw_labels[idx_negative_label]

        # take random 10 percent of the negative labelled data
        in_keep = np.random.permutation(np.sum(idx_negative_label))
        under_sampling_rate = undersampled_rate #0.5
        in_keep = in_keep[0:np.int(np.sum(idx_negative_label) * under_sampling_rate)]

        neg_samps_input = neg_samps_input[in_keep, :]
        neg_samps_label = neg_samps_label[in_keep]

        feature_selected = np.concatenate((pos_samps_input, neg_samps_input))
        label_selected = np.concatenate((pos_samps_label, neg_samps_label))

        X_train, X_test, y_train, y_test = train_test_split(feature_selected, label_selected, train_size=0.80,
                                                            test_size=0.20, random_state=seed_number)

    elif dataset=='adult':

        print("dataset is", dataset) # this is heterogenous
        print(socket.gethostname())
        #if 'g0' not in socket.gethostname():
        data, categorical_columns, ordinal_columns = load_dataset('adult')
        # else:

        """ some specifics on this dataset """
        numerical_columns = list(set(np.arange(data[:, :-1].shape[1])) - set(categorical_columns + ordinal_columns))
        n_classes = 2

        data = data[:, numerical_columns + ordinal_columns + categorical_columns]

        num_numerical_inputs = len(numerical_columns)
        num_categorical_inputs = len(categorical_columns + ordinal_columns) - 1

        inputs = data[:, :-1]
        target = data[:, -1]

        inputs, target=undersample(inputs, target, undersampled_rate)

        X_train, X_test, y_train, y_test = train_test_split(inputs, target, train_size=0.90, test_size=0.10,
                                                            random_state=seed_number)



    elif dataset=='isolet':

        print("isolet dataset")
        print(socket.gethostname())
        if 'g0' not in socket.gethostname() and 'p0' not in socket.gethostname():
            data_features_npy = np.load('../data/Isolet/isolet_data.npy')
            data_target_npy = np.load('../data/Isolet/isolet_labels.npy')
        else:
            # (1) load data
            data_features_npy = np.load(
                '/home/user/Dropbox_from/Current_research/privacy/DPDR/data/Isolet/isolet_data.npy')
            data_target_npy = np.load(
                '/home/user/Dropbox_from/Current_research/privacy/DPDR//data/Isolet/isolet_labels.npy')

        print(data_features_npy.shape)
        # dtype = [('Col1', 'int32'), ('Col2', 'float32'), ('Col3', 'float32')]
        values = data_features_npy
        index = ['Row' + str(i) for i in range(1, len(values) + 1)]

        values_l = data_target_npy
        index_l = ['Row' + str(i) for i in range(1, len(values) + 1)]

        data_features = pd.DataFrame(values, index=index)
        data_target = pd.DataFrame(values_l, index=index_l)

        ####

        raw_labels = np.array(data_target)
        raw_input_features = np.array(data_features)

        idx_negative_label = raw_labels == 0
        idx_positive_label = raw_labels == 1

        idx_negative_label = idx_negative_label.squeeze()
        idx_positive_label = idx_positive_label.squeeze()

        pos_samps_input = raw_input_features[idx_positive_label, :]
        pos_samps_label = raw_labels[idx_positive_label]
        neg_samps_input = raw_input_features[idx_negative_label, :]
        neg_samps_label = raw_labels[idx_negative_label]

        # take random 10 percent of the negative labelled data
        in_keep = np.random.permutation(np.sum(idx_negative_label))
        under_sampling_rate = undersampled_rate  # 0.01
        # under_sampling_rate = 0.3
        in_keep = in_keep[0:np.int(np.sum(idx_negative_label) * under_sampling_rate)]

        neg_samps_input = neg_samps_input[in_keep, :]
        neg_samps_label = neg_samps_label[in_keep]

        feature_selected = np.concatenate((pos_samps_input, neg_samps_input))
        label_selected = np.concatenate((pos_samps_label, neg_samps_label))

        label_selected = label_selected.squeeze()


        ####

        X_train, X_test, y_train, y_test = train_test_split(feature_selected, label_selected, train_size=0.70, test_size=0.30,
                                                            random_state=seed_number)

        # unpack data
        #X_train = X_train.values
        #y_train = y_train.values.ravel()
        #X_test = np.array(X_test)
        #y_test = np.array(y_test)
        n_classes = 2


    elif dataset=='intrusion':

        print("dataset is", dataset)
        print(socket.gethostname())
        data, categorical_columns, ordinal_columns = load_dataset('intrusion')

        """ some specifics on this dataset """
        n_classes = 5

        """ some changes we make in the type of features for applying to our model """
        categorical_columns_binary = [6, 11, 13, 20]  # these are binary categorical columns
        the_rest_columns = list(set(np.arange(data[:, :-1].shape[1])) - set(categorical_columns_binary))

        num_numerical_inputs = len(the_rest_columns)  # 10. Separately from the numerical ones, we compute the length-scale for the rest columns
        num_categorical_inputs = len(categorical_columns_binary)  # 4.

        raw_labels = data[:, -1]
        raw_input_features = data[:, the_rest_columns + categorical_columns_binary]
        print(raw_input_features.shape)

        """ we take a pre-processing step such that the dataset is a bit more balanced """
        idx_negative_label = raw_labels == 0  # this is a dominant one about 80%, which we want to undersample
        idx_positive_label = raw_labels != 0

        pos_samps_input = raw_input_features[idx_positive_label, :]
        pos_samps_label = raw_labels[idx_positive_label]
        neg_samps_input = raw_input_features[idx_negative_label, :]
        neg_samps_label = raw_labels[idx_negative_label]

        # take random 40% of the negative labelled data
        in_keep = np.random.permutation(np.sum(idx_negative_label))
        under_sampling_rate = undersampled_rate#0.3
        in_keep = in_keep[0:np.int(np.sum(idx_negative_label) * under_sampling_rate)]

        neg_samps_input = neg_samps_input[in_keep, :]
        neg_samps_label = neg_samps_label[in_keep]

        feature_selected = np.concatenate((pos_samps_input, neg_samps_input))
        label_selected = np.concatenate((pos_samps_label, neg_samps_label))

        X_train, X_test, y_train, y_test = train_test_split(feature_selected, label_selected, train_size=0.70,
                                                            test_size=0.30,
                                                            random_state=seed_number)

    elif dataset=='covtype':

        print("dataset is", dataset)
        print(socket.gethostname())
        if 'g0' not in socket.gethostname() and 'p0' not in socket.gethostname():
            train_data = np.load("../data/real/covtype/train.npy")
            test_data = np.load("../data/real/covtype/test.npy")
            # we put them together and make a new train/test split in the following
            data = np.concatenate((train_data, test_data))
        else:
            train_data = np.load(
                "/home/user/Dropbox_from/Current_research/privacy/DPDR/data/real/covtype/train.npy")
            test_data = np.load(
                "/home/user/Dropbox_from/Current_research/privacy/DPDR/data/real/covtype/test.npy")
            data = np.concatenate((train_data, test_data))

        """ some specifics on this dataset """
        numerical_columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        ordinal_columns = []
        categorical_columns = list(set(np.arange(data.shape[1])) - set(numerical_columns + ordinal_columns))
        # Note: in this dataset, the categorical variables are all binary
        n_classes = 7

        print('data shape is', data.shape)
        print('indices for numerical columns are', numerical_columns)
        print('indices for categorical columns are', categorical_columns)
        print('indices for ordinal columns are', ordinal_columns)

        # sorting the data based on the type of features.
        data = data[:, numerical_columns + ordinal_columns + categorical_columns]
        # data = data[0:150000, numerical_columns + ordinal_columns + categorical_columns] # for fast testing the results

        num_numerical_inputs = len(numerical_columns)
        num_categorical_inputs = len(categorical_columns + ordinal_columns) - 1

        inputs = data[:20000, :-1]
        target = data[:20000, -1]


        ##################3

        raw_labels=target
        raw_input_features=inputs

        """ we take a pre-processing step such that the dataset is a bit more balanced """
        idx_negative_label = raw_labels == 1  # 1 and 0 are dominant but 1 has more labels
        idx_positive_label = raw_labels != 1

        pos_samps_input = raw_input_features[idx_positive_label, :]
        pos_samps_label = raw_labels[idx_positive_label]
        neg_samps_input = raw_input_features[idx_negative_label, :]
        neg_samps_label = raw_labels[idx_negative_label]

        # take random 40% of the negative labelled data
        in_keep = np.random.permutation(np.sum(idx_negative_label))
        under_sampling_rate = undersampled_rate  # 0.3
        in_keep = in_keep[0:np.int(np.sum(idx_negative_label) * under_sampling_rate)]

        neg_samps_input = neg_samps_input[in_keep, :]
        neg_samps_label = neg_samps_label[in_keep]

        feature_selected = np.concatenate((pos_samps_input, neg_samps_input))
        label_selected = np.concatenate((pos_samps_label, neg_samps_label))


        ###############3

        X_train, X_test, y_train, y_test = train_test_split(feature_selected, label_selected, train_size=0.70, test_size=0.30,
                                                            random_state=seed_number)  # 60% training and 40% test


    # specify heterogeneous dataset or not
    heterogeneous_datasets = ['cervical', 'adult', 'census', 'intrusion', 'covtype']
    homogeneous_datasets = ['epileptic','credit','isolet']

    ########################################################################################
    ##########################################################################################3



    def test_models(X_tr, y_tr, X_te, y_te, datasettype):

        print("\n",  datasettype, "data\n")

        roc_arr=[]
        prc_arr=[]
        f1_arr=[]

        aa=np.array([1,3])

        models=np.array([LogisticRegression(solver='lbfgs', max_iter=2000), GaussianNB(), BernoulliNB(alpha=0.02), LinearSVC(), DecisionTreeClassifier(), LinearDiscriminantAnalysis(), AdaBoostClassifier(), BaggingClassifier(), RandomForestClassifier(), GradientBoostingClassifier(), MLPClassifier(), xgboost.XGBClassifier()])
        models_to_test=models[np.array(arguments.classifiers)]

        # check
        for model in models_to_test:
        #for model in [LogisticRegression(solver='lbfgs', max_iter=1000), GaussianNB(), BernoulliNB(alpha=0.02)]:

            print('\n', type(model))
            model.fit(X_tr, y_tr)
            pred = model.predict(X_te)  # test on real data

        #LR_model = LogisticRegression(solver='lbfgs', max_iter=1000)
        # LR_model.fit(X_train, y_train)  # training on synthetic data
        # pred = LR_model.predict(X_test)  # test on real data

            if n_classes>2:

                f1score = f1_score(y_te, pred, average='weighted')

                print("F1-score on test %s data is %.3f" % (datasettype, f1score))
                # 0.6742486709433465 for covtype data, 0.9677751506935462 for intrusion data
                f1_arr.append(f1score)

            else:

                roc = roc_auc_score(y_te, pred)
                prc = average_precision_score(y_te, pred)

                print("ROC on test %s data is %.3f" % (datasettype, roc))
                print("PRC on test %s data is %.3f" % (datasettype, prc))

                roc_arr.append(roc)
                prc_arr.append(prc)

        if n_classes > 2:

            res1 = np.mean(f1_arr); res1_arr=f1_arr
            print("------\nf1 mean across methods is %.3f\n" % res1)
            res2 = 0; res2_arr=0  # dummy
        else:

            res1=np.mean(roc_arr); res1_arr=roc_arr
            res2=np.mean(prc_arr); res2_arr=prc_arr
            print("-"*40)
            print("roc mean across methods is %.3f" % res1)
            print("prc mean across methods is %.3f\n" % res2)

        return res1_arr, res2_arr
        #return res1, res2


    #######################################################
    # TEST REAL DATA


    # As a reference, we first test logistic regression on the real data
    # check
    if arguments.data_type=='real' or arguments.data_type=='both':
        if dataset in heterogeneous_datasets:
            roc_real, prc_real = test_models(X_train, y_train, X_test, y_test, "real")
            roc_return, prc_return=roc_real, prc_real
        else:
            f1_real = test_models(X_train, y_train, X_test, y_test, "real")
            f1_return = f1_real



    ###########################################################################
    # PREPARING GENERATOR

    if arguments.data_type=='generated' or arguments.data_type=='both':

        # one-hot encoding of labels.
        n, input_dim = X_train.shape
        onehot_encoder = OneHotEncoder(sparse=False)
        y_train = np.expand_dims(y_train, 1)
        true_labels = onehot_encoder.fit_transform(y_train)


        ######################################
        # MODEL

        # model specifics
        mini_batch_size = np.int(np.round(mini_batch_size_arg * n))
        print("minibatch: ", mini_batch_size)
        input_size = 10 + 1
        hidden_size_1 = 4 * input_dim
        hidden_size_2 = 2 * input_dim
        output_size = input_dim

        if dataset in homogeneous_datasets:

            model = Generative_Model_homogeneous_data(input_size=input_size, hidden_size_1=hidden_size_1,
                                                          hidden_size_2=hidden_size_2,
                                                          output_size=output_size, dataset=dataset).to(device)

        elif dataset in heterogeneous_datasets:

            model = Generative_Model_heterogeneous_data(input_size=input_size, hidden_size_1=hidden_size_1,
                                                            hidden_size_2=hidden_size_2,
                                                            output_size=output_size,
                                                            num_categorical_inputs=num_categorical_inputs,
                                                            num_numerical_inputs=num_numerical_inputs).to(device)
        else:
            print('sorry, please enter the name of your dataset either in homogeneous_dataset or heterogeneous_dataset list ')

        # define details for training
        optimizer = optim.Adam(model.parameters(), lr=1e-2)
        how_many_epochs = how_many_epochs_arg
        how_many_iter = 1 #np.int(n / mini_batch_size)
        training_loss_per_epoch = np.zeros(how_many_epochs)


        ##########################################################################


        """ specifying random fourier features """

        idx_rp = np.random.permutation(n)

        if dataset=='census': # some columns of census data have many zeros, so we need more datapoints to get meaningful length scales
            num_data_pt_to_discard = 100
        else:
            num_data_pt_to_discard = 10

        idx_to_discard = idx_rp[0:num_data_pt_to_discard]
        idx_to_keep = idx_rp[num_data_pt_to_discard:]

        if dataset=='census':

            sigma_array = np.zeros(num_numerical_inputs)
            for i in np.arange(0, num_numerical_inputs):
                med = util.meddistance(np.expand_dims(X_train[idx_to_discard, i], 1))
                sigma_array[i] = med


            print('we will use separate frequencies for each column of numerical features')
            sigma2 = sigma_array**2
            sigma2[sigma2==0] = 1.0
            # sigma2[sigma2>500] = 500
            #print('sigma values are ', sigma2)
            # sigma2 = np.mean(sigma2)

        elif dataset=='credit':

            # large value at the last column

            med = util.meddistance(X_train[idx_to_discard, 0:-1])
            med_last = util.meddistance(np.expand_dims(X_train[idx_to_discard, -1],1))
            sigma_array = np.concatenate((med*np.ones(input_dim-1), [med_last]))

            sigma2 = sigma_array**2
            sigma2[sigma2==0] = 1.0

            #print('sigma values are ', sigma2)

        else:

            if dataset in heterogeneous_datasets:
                med = util.meddistance(X_train[idx_to_discard, 0:num_numerical_inputs])
            else:
                med = util.meddistance(X_train[idx_to_discard, ])

            sigma2 = med ** 2

        X_train = X_train[idx_to_keep,:]
        true_labels = true_labels[idx_to_keep,:]
        n = X_train.shape[0]
        print('total number of datapoints in the training data is', n)

        # random Fourier features
        n_features = n_features_arg
        draws = n_features // 2

        # random fourier features for numerical inputs only
        if dataset in heterogeneous_datasets:
            W_freq = np.random.randn(draws, num_numerical_inputs) / np.sqrt(sigma2)
        else:
            W_freq = np.random.randn(draws, input_dim) / np.sqrt(sigma2)

        """ specifying ratios of data to generate depending on the class lables """
        unnormalized_weights = np.sum(true_labels,0)
        weights = unnormalized_weights/np.sum(unnormalized_weights)
        print('\nweights with no privatization are', weights, '\n'  )

        ####################################################
        # Privatising quantities if necessary

        """ privatizing weights """
        if is_private:
            print("private")
            # desired privacy level
            epsilon = 1.0
            delta = 1e-5
            # k = n_classes + 1   # this dp analysis has been updated
            k = 2
            privacy_param = privacy_calibrator.gaussian_mech(epsilon, delta, k=k)
            print(f'eps,delta = ({epsilon},{delta}) ==> Noise level sigma=', privacy_param['sigma'])

            sensitivity_for_weights = np.sqrt(2)/n  # double check if this is sqrt(2) or 2
            noise_std_for_weights = privacy_param['sigma'] * sensitivity_for_weights
            weights = weights + np.random.randn(weights.shape[0])*noise_std_for_weights
            weights[weights < 0] = 1e-3 # post-processing so that we don't have negative weights.
            print('weights after privatization are', weights)

        """ computing mean embedding of subsampled true data """
        if dataset in homogeneous_datasets:

            emb1_input_features = RFF_Gauss(n_features, torch.Tensor(X_train), W_freq)
            emb1_labels = Feature_labels(torch.Tensor(true_labels), weights)
            outer_emb1 = torch.einsum('ki,kj->kij', [emb1_input_features, emb1_labels])
            mean_emb1 = torch.mean(outer_emb1, 0)

        else:  # heterogeneous data

            numerical_input_data = X_train[:, 0:num_numerical_inputs]
            emb1_numerical = (RFF_Gauss(n_features, torch.Tensor(numerical_input_data), W_freq)).to(device)

            categorical_input_data = X_train[:, num_numerical_inputs:]

            emb1_categorical = (torch.Tensor(categorical_input_data) / np.sqrt(num_categorical_inputs)).to(device)

            emb1_input_features = torch.cat((emb1_numerical, emb1_categorical), 1)

            emb1_labels = Feature_labels(torch.Tensor(true_labels), weights)
            outer_emb1 = torch.einsum('ki,kj->kij', [emb1_input_features, emb1_labels])
            mean_emb1 = torch.mean(outer_emb1, 0)


        """ privatizing each column of mean embedding """
        if is_private:
            if dataset in heterogeneous_datasets:
                sensitivity = 2*np.sqrt(2) / n
            else:
                sensitivity = 2 / n
            noise_std_for_privacy = privacy_param['sigma'] * sensitivity

            # make sure add noise after rescaling
            weights_torch = torch.Tensor(weights)
            weights_torch = weights_torch.to(device)

            rescaled_mean_emb = weights_torch*mean_emb1
            noise = noise_std_for_privacy * torch.randn(mean_emb1.size())
            noise = noise.to(device)

            rescaled_mean_emb = rescaled_mean_emb + noise

            mean_emb1 = rescaled_mean_emb/weights_torch # rescaling back\

        # End of Privatising quantities if necessary
        ####################################################

        ##################################################################################################################
        # TRAINING THE GENERATOR

        print('Starting Training')

        for epoch in range(how_many_epochs):  # loop over the dataset multiple times

            running_loss = 0.0

            for i in range(how_many_iter):

                """ computing mean embedding of generated data """
                # zero the parameter gradients
                optimizer.zero_grad()

                if dataset in homogeneous_datasets: # In our case, if a dataset is homogeneous, then it is a binary dataset.

                    label_input = (1 * (torch.rand((mini_batch_size)) < weights[1])).type(torch.FloatTensor)
                    label_input = label_input.to(device)
                    feature_input = torch.randn((mini_batch_size, input_size-1)).to(device)
                    input_to_model = torch.cat((feature_input, label_input[:,None]), 1)

                    #we feed noise + label (like in cond-gan) as input
                    outputs = model(input_to_model)

                    """ computing mean embedding of generated samples """
                    emb2_input_features = RFF_Gauss(n_features, outputs, W_freq)

                    label_input_t = torch.zeros((mini_batch_size, n_classes))
                    idx_1 = (label_input == 1.).nonzero()[:,0]
                    idx_0 = (label_input == 0.).nonzero()[:,0]
                    label_input_t[idx_1, 1] = 1.
                    label_input_t[idx_0, 0] = 1.

                    emb2_labels = Feature_labels(label_input_t, weights)
                    outer_emb2 = torch.einsum('ki,kj->kij', [emb2_input_features, emb2_labels])
                    mean_emb2 = torch.mean(outer_emb2, 0)

                else:  # heterogeneous data

                    # (1) generate labels
                    label_input = torch.multinomial(torch.Tensor([weights]), mini_batch_size, replacement=True).type(torch.FloatTensor)
                    label_input=torch.cat((label_input, torch.arange(len(weights), out=torch.FloatTensor()).unsqueeze(0)),1) #to avoid no labels
                    label_input = label_input.transpose_(0,1)
                    label_input = label_input.to(device)

                    # (2) generate corresponding features
                    feature_input = torch.randn((mini_batch_size+len(weights), input_size-1)).to(device)
                    input_to_model = torch.cat((feature_input, label_input), 1)
                    outputs = model(input_to_model)

                    # (3) compute the embeddings of those
                    numerical_samps = outputs[:, 0:num_numerical_inputs] #[4553,6]
                    emb2_numerical = RFF_Gauss(n_features, numerical_samps, W_freq) #W_freq [n_features/2,6], n_features=10000

                    categorical_samps = outputs[:, num_numerical_inputs:] #[4553,8]

                    emb2_categorical = categorical_samps /(torch.sqrt(torch.Tensor([num_categorical_inputs]))).to(device) # 8

                    emb2_input_features = torch.cat((emb2_numerical, emb2_categorical), 1)

                    generated_labels = onehot_encoder.fit_transform(label_input.cpu().detach().numpy()) #[1008]
                    emb2_labels = Feature_labels(torch.Tensor(generated_labels), weights)
                    outer_emb2 = torch.einsum('ki,kj->kij', [emb2_input_features, emb2_labels])
                    mean_emb2 = torch.mean(outer_emb2, 0)

                loss = torch.norm(mean_emb1 - mean_emb2, p=2) ** 2

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            if epoch % 100 == 0:
                print('epoch # and running loss are ', [epoch, running_loss])
                training_loss_per_epoch[epoch] = running_loss



        #######################################################################33
        if dataset in heterogeneous_datasets:

            """ draw final data samples """

            label_input = torch.multinomial(torch.Tensor([weights]), n, replacement=True).type(torch.FloatTensor)
            label_input = label_input.transpose_(0, 1)
            label_input = label_input.to(device)

            # (2) generate corresponding features
            feature_input = torch.randn((n, input_size - 1)).to(device)
            input_to_model = torch.cat((feature_input, label_input), 1)
            outputs = model(input_to_model)


            # (3) round the categorial features
            output_numerical = outputs[:, 0:num_numerical_inputs]
            output_categorical = outputs[:, num_numerical_inputs:]
            output_categorical = torch.round(output_categorical)

            output_combined = torch.cat((output_numerical, output_categorical), 1)

            generated_input_features_final = output_combined.cpu().detach().numpy()
            generated_labels_final = label_input.cpu().detach().numpy()

            roc, prc= test_models(generated_input_features_final, generated_labels_final, X_test, y_test, "generated")
            roc_return, prc_return=roc, prc
            #roc, prc=-1, -1

            # LR_model_ours = LogisticRegression(solver='lbfgs', max_iter=1000)
            # LR_model_ours.fit(generated_input_features_final, generated_labels_final)  # training on synthetic data
            # pred_ours = LR_model_ours.predict(X_test)  # test on real data

            #return roc_real, prc_real
            #return roc, prc

        else: # homogeneous datasets

            """ now generate samples from the trained network """

            # weights[1] represents the fraction of the positive labels in the dataset,
            #and we would like to generate a similar fraction of the postive/negative datapoints
            label_input = (1 * (torch.rand((n)) < weights[1])).type(torch.FloatTensor)
            label_input = label_input.to(device)

            feature_input = torch.randn((n, input_size - 1)).to(device)
            input_to_model = torch.cat((feature_input, label_input[:, None]), 1)
            outputs = model(input_to_model)

            samp_input_features = outputs

            label_input_t = torch.zeros((n, n_classes))
            idx_1 = (label_input == 1.).nonzero()[:, 0]
            idx_0 = (label_input == 0.).nonzero()[:, 0]
            label_input_t[idx_1, 1] = 1.
            label_input_t[idx_0, 0] = 1.

            samp_labels = label_input_t

            generated_input_features_final = samp_input_features.cpu().detach().numpy()
            generated_labels_final = samp_labels.cpu().detach().numpy()
            generated_labels=np.argmax(generated_labels_final, axis=1)

            f1 = test_models(generated_input_features_final, generated_labels, X_test, y_test, "generated")
            f1_return=f1
            #f1=-1

            #return f1_real
            #return f1

    # returnt he values, if dataset is real, then real f1/roc/prc are returned, if dataset is generated or both geenrated f1/roc/prc are returned

    if dataset in heterogeneous_datasets:
        return roc_return, prc_return
    else:
        return f1_return


        #LR_model_ours = LogisticRegression(solver='lbfgs', max_iter=1000)
        #LR_model_ours.fit(generated_input_features_final,
         #                 np.argmax(generated_labels_final, axis=1))  # training on synthetic data
        #pred_ours = LR_model_ours.predict(X_test)  # test on real data



    # if n_classes>2:
    #
    #     f1score = f1_score(y_test, pred_ours, average='weighted')
    #     print('F1-score (ours) is ', f1score)
    #
    #     return f1score
    #
    # else:
    #
    #     roc = roc_auc_score(y_test, pred_ours)
    #     prc = average_precision_score(y_test, pred_ours)
    #
    #     print('ROC ours is', roc)
    #     print('PRC ours is', prc)
    #     print('n_features are ', n_features)
    #
    #     return roc, prc


###################################################################################################
###################################################################################################
###################################################################################################

if __name__ == '__main__':

    #epileptic, credit, census, cervical, adult, isolet

    #dataset = "cervical"

    ### this is setup I was testing for Credit data.
    ### Do not remove this please
    # is_priv_arg = False
    # single_run = False
    #
    # dataset = 'credit'
    #
    # how_many_epochs_arg = [4000]
    # n_features_arg = [500]
    # mini_batch_arg = [1.0]
    #
    # grid = ParameterGrid({"n_features_arg": n_features_arg, "mini_batch_arg": mini_batch_arg,
    #                               "how_many_epochs_arg": how_many_epochs_arg})
    # for elem in grid:
    #     print(elem, "\n")
    #     prc_arr = []; roc_arr = []
    #     repetitions = 5
    #     for ii in range(repetitions):
    #
    #         roc, prc  = main(dataset, elem["n_features_arg"], elem["mini_batch_arg"], elem["how_many_epochs_arg"], is_priv_arg, seed_number=ii)
    #         roc_arr.append(roc)
    #         prc_arr.append(prc)
    #
    #     print("Average ROC: ", np.mean(roc_arr)); print("Avergae PRC: ", np.mean(prc_arr))
    #     print("Std ROC: ", np.std(roc_arr)); print("Variance PRC: ", np.std(prc_arr), "\n")



    is_priv_arg = arguments.private  # check
    single_run = True  # check

    repetitions = arguments.repeat  # check

    # check
    #for dataset in ["credit", "epileptic", "census", "cervical", "adult", "isolet", "covtype", "intrusion"]:
    for dataset in [arguments.dataset]:
    #for dataset in ["intrusion"]:
        print("\n\n")
        print('is private?', is_priv_arg)

        if single_run == True:
            how_many_epochs_arg = list_epochs#[200]
            # n_features_arg = [100000]#, 5000, 10000, 50000, 80000]
            n_features_arg = list_num_features#[500]
            mini_batch_arg = [arguments.batch]#[0.3]
            undersampling_rates = [arguments.undersample] #[.8]  # dummy
        else:
            how_many_epochs_arg = [8000, 6000, 2000, 1000, 4000]
            n_features_arg = [500, 1000, 2000, 5000, 10000, 50000, 80000, 100000]
            # n_features_arg = [5000, 10000, 50000, 80000, 100000]
            # n_features_arg = [50000, 80000, 100000]
            mini_batch_arg = [0.5]

            if dataset=='adult':
                mini_batch_arg=[0.1]
                n_features_arg = [500, 1000, 2000, 5000, 10000, 50000]
                undersampling_rates = [.4]#[.8, .6, .4] #dummy
            elif dataset=='census':
                mini_batch_arg=[0.1]
                n_features_arg = [500, 1000, 2000, 5000, 10000, 50000, 80000]
                undersampling_rates = [0.4]#[0.2, 0.3, 0.4]
            elif dataset=='covtype':
                how_many_epochs_arg = [10000, 7000, 4000, 2000, 1000]
                n_features_arg = [500, 1000, 2000, 5000, 10000, 50000, 80000]
                mini_batch_arg=[0.03]
                repetitions=3
                undersampling_rates = [1.] #dummy
            elif dataset == 'intrusion':
                how_many_epochs_arg = [10000, 7000, 4000, 2000, 1000]
                n_features_arg = [500, 1000, 2000, 5000, 10000, 50000, 80000]
                mini_batch_arg = [0.03]
                repetitions=3
                undersampling_rates=[0.3]#[0.1, 0.2, 0.3]
            elif dataset=='credit':
                undersampling_rates = [0.005]#[0.01, 0.005, 0.02]
            elif dataset=='cervical':
                undersampling_rates = [1.]#[0.1, 0.3, 0.5, 0.7, 1.0]
            elif dataset=='isolet':
                undersampling_rates = [1.]#[0.8, 0.6, 0.4, 1.] #dummy
            elif dataset=='epileptic':
                undersampling_rates = [1.]#[0.8, 0.6, 0.4, 1.] #dummy



        grid = ParameterGrid({"undersampling_rates": undersampling_rates, "n_features_arg": n_features_arg, "mini_batch_arg": mini_batch_arg,
                              "how_many_epochs_arg": how_many_epochs_arg})





        if dataset in ["credit", "census", "cervical", "adult", "isolet", "epileptic"]:

            max_aver_roc, max_aver_prc, max_roc, max_prc, max_aver_rocprc, max_elem=0, 0, 0, 0, [0,0], 0

            for elem in grid:
                print(elem, "\n")
                prc_arr_all = []; roc_arr_all = []
                for ii in range(repetitions):
                    print("\nRepetition: ",ii)

                    roc, prc  = main(dataset, elem["undersampling_rates"], elem["n_features_arg"], elem["mini_batch_arg"], elem["how_many_epochs_arg"], is_priv_arg, seed_number=ii)

                    # print("Variables: ")
                    # for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
                    #                          key=lambda x: -x[1])[:10]:
                    #     print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
                    # print("\n")

                    roc_arr_all.append(roc)
                    prc_arr_all.append(prc)


                roc_each_method_avr=np.mean(roc_arr_all, 0)
                prc_each_method_avr=np.mean(prc_arr_all, 0)
                roc_each_method_std = np.std(roc_arr_all, 0)
                prc_each_method_std = np.std(prc_arr_all, 0)
                roc_arr=np.mean(roc_arr_all, 1)
                prc_arr = np.mean(prc_arr_all, 1)

                print("\n", "-" * 40, "\n\n")
                print("For each of the methods")
                print("Average ROC each method: ", roc_each_method_avr);
                print("Average PRC each method: ", prc_each_method_avr);
                print("Std ROC each method: ", roc_each_method_std);
                print("Std PRC each method: ", prc_each_method_std)


                print("Average over repetitions across all methods")
                print("Average ROC: ", np.mean(roc_arr)); print("Average PRC: ", np.mean(prc_arr))
                print("Std ROC: ", np.std(roc_arr)); print("Variance PRC: ", np.std(prc_arr), "\n")
                print("\n", "-" * 80, "\n\n\n")

                if np.mean(roc_arr)>max_aver_roc:
                    max_aver_roc=np.mean(roc_arr)

                if np.mean(prc_arr)>max_aver_prc:
                    max_aver_prc=np.mean(prc_arr)

                if np.mean(roc_arr) + np.mean(prc_arr)> max_aver_rocprc[0]+max_aver_rocprc[1]:
                    max_aver_rocprc = [np.mean(roc_arr), np.mean(prc_arr)]
                    max_elem = elem

            print("\n\n", "*"*30, )
            print(dataset)
            print("Max ROC! ", max_aver_rocprc[0])
            print("Max PRC! ", max_aver_rocprc[1])
            print("Setup: ", max_elem)
            print('*'*100)





        elif dataset in ["covtype", "intrusion"]: # multi-class classification problems.

            max_f1, max_aver_f1, max_elem=0, 0, 0

            for elem in grid:
                print(elem, "\n")
                f1score_arr_all = []
                for ii in range(repetitions):
                    print("\nRepetition: ",ii)

                    f1scr = main(dataset, elem["undersampling_rates"], elem["n_features_arg"], elem["mini_batch_arg"], elem["how_many_epochs_arg"], is_priv_arg, seed_number=ii)
                    f1score_arr_all.append(f1scr[0])

                f1score_each_method_avr = np.mean(f1score_arr_all, 0)
                f1score_each_method_std = np.std(f1score_arr_all, 0)
                f1score_arr = np.mean(f1score_arr_all, 1)

                print("\n", "-" * 40, "\n\n")
                print("For each of the methods")
                print("Average F1: ", f1score_each_method_avr)
                print("Std F1: ", f1score_each_method_std)


                print("Average over repetitions across all methods")
                print("Average f1 score: ", np.mean(f1score_arr))
                print("Std F1: ", np.std(f1score_arr))
                print("\n","-" * 80, "\n\n\n")

                if np.mean(f1score_arr)>max_aver_f1:
                    max_aver_f1=np.mean(f1score_arr)
                    max_elem = elem

            print("\n\n", "Max F1! ", max_aver_f1, "*"*20)
            print("Setup: ", max_elem)
            print('*' * 30)





