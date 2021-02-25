import socket
import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
import math

if sys.version_info[0] > 2:
    import sdgym
import xgboost



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
#import xgboost

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterGrid
#from autodp import privacy_calibrator
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score


#################################


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



####################################
def load_cervical():
    print("dataset is cervical")  # this is heterogenous
    seed_number=0
    print(socket.gethostname())
    if 'g0' not in socket.gethostname() and 'p0' not in socket.gethostname():
        df = pd.read_csv("/home/anon_k/Desktop/Dropbox/Current_research/privacy/DPDR/data/Cervical/kag_risk_factors_cervical_cancer.csv")
    else:
        df = pd.read_csv(
            "/home/anon_k/Dropbox_from/Current_research/privacy/DPDR/data/Cervical/kag_risk_factors_cervical_cancer.csv")
        print("Loaded Cervical")

    # df.head()

    df_nan = df.replace("?", np.float64(np.nan))

    #counting number of missing data samples
    # ind=0
    # nparray=np.array(df_nan)
    # for i in range(len(nparray)):
    #     print(nparray[i])
    #     had_qmark=False
    #     for j in range(len(nparray[0])):
    #         if math.isnan(float(nparray[i][j])) and had_qmark==False:
    #             ind+=1
    #             had_qmark=True

    #print("mising:", str(ind))




    df_nan.head()

    # df1 = df_nan.convert_objects(convert_numeric=True)
    df1 = df.apply(pd.to_numeric, errors="coerce")

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
        # df[feature] = df[feature].convert_objects(convert_numeric=True).fillna(0.0)
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
    under_sampling_rate = 0.5 #undersampled_rate  # 0.5
    in_keep = in_keep[0:np.int(np.sum(idx_negative_label) * under_sampling_rate)]

    neg_samps_input = neg_samps_input[in_keep, :]
    neg_samps_label = neg_samps_label[in_keep]

    feature_selected = np.concatenate((pos_samps_input, neg_samps_input))
    label_selected = np.concatenate((pos_samps_label, neg_samps_label))

    X_train, X_test, y_train, y_test = train_test_split(feature_selected, label_selected, train_size=0.80,
                                                        test_size=0.20, random_state=seed_number)

    return X_train, y_train, X_test, y_test


def load_isolet():

    seed_number=0

    print("isolet dataset")
    print(socket.gethostname())
    if 'g0' not in socket.gethostname() and 'p0' not in socket.gethostname():
        data_features_npy = np.load('/home/anon_k/Dropbox/Current_research/privacy/DPDR/data/Isolet/isolet_data.npy')
        data_target_npy = np.load('/home/anon_k/Dropbox/Current_research/privacy/DPDR/data/Isolet/isolet_labels.npy')
    else:
        # (1) load data
        data_features_npy = np.load(
            '/home/anon_k/Dropbox_from/Current_research/privacy/DPDR/data/Isolet/isolet_data.npy')
        data_target_npy = np.load(
            '/home/anon_k/Dropbox_from/Current_research/privacy/DPDR//data/Isolet/isolet_labels.npy')

    print(data_features_npy.shape)
    # dtype = [('Col1', 'int32'), ('Col2', 'float32'), ('Col3', 'float32')]
    values = data_features_npy
    index = ['Row' + str(i) for i in range(1, len(values) + 1)]

    values_l = data_target_npy
    index_l = ['Row' + str(i) for i in range(1, len(values) + 1)]

    data_features = pd.DataFrame(values, index=index)
    data_target = pd.DataFrame(values_l, index=index_l)

    X_train, X_test, y_train, y_test = train_test_split(data_features, data_target, train_size=0.70, test_size=0.30,
                                                        random_state=seed_number)

    # unpack data
    X_train = X_train.values
    y_train = y_train.values.ravel()
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    n_classes = 2

    data_features=np.array(data_features)
    data_target=np.array(data_target).squeeze()

    return X_train, y_train, X_test, y_test.squeeze()

def load_credit():

    seed_number=0


    print("Creditcard fraud detection dataset") # this is homogeneous

    if 'g0' not in socket.gethostname() and 'p0' not in socket.gethostname():
        data = pd.read_csv("/home/anon_k/Desktop/Dropbox/Current_research/privacy/DPDR/data/Kaggle_Credit/creditcard.csv")
        #data = pd.read_csv(
        #    "../data/Kaggle_Credit/creditcard.csv")
    else:
        # (1) load data
        data = pd.read_csv(
            '/home/anon_k/Dropbox_from/Current_research/privacy/DPDR/data/Kaggle_Credit/creditcard.csv')

    feature_names = data.iloc[:, 1:30].columns
    target = data.iloc[:1, 30:].columns

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
    under_sampling_rate = 0.01# undersampled_rate #0.01
    # under_sampling_rate = 0.3
    in_keep = in_keep[0:np.int(np.sum(idx_negative_label) * under_sampling_rate)]

    neg_samps_input = neg_samps_input[in_keep, :]
    neg_samps_label = neg_samps_label[in_keep]

    feature_selected = np.concatenate((pos_samps_input, neg_samps_input))
    label_selected = np.concatenate((pos_samps_label, neg_samps_label))

    X_train, X_test, y_train, y_test = train_test_split(feature_selected, label_selected, train_size=0.80,
                                                        test_size=0.20, random_state=seed_number)
    n_classes = 2

    return X_train, y_train, X_test, y_test.squeeze()


def load_epileptic():
    print("epileptic seizure recognition dataset") # this is homogeneous

    seed_number=0

    if 'g0' not in socket.gethostname() and 'p0' not in socket.gethostname():
        data = pd.read_csv("/home/anon_k/Desktop/Dropbox/Current_research/privacy/DPDR/data/Epileptic/data.csv")
    else:
        # (1) load data
        data = pd.read_csv('/home/anon_k/Dropbox_from/Current_research/privacy/DPDR/data/Epileptic/data.csv')

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
    under_sampling_rate = 1. #undersampled_rate  # 1.
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

    return X_train, y_train, X_test, y_test


def load_census():

    seed_number=0

    print("census dataset") # this is heterogenous

    print(socket.gethostname())
    if 'g0' not in socket.gethostname() and 'p0' not in socket.gethostname():
        data = np.load("/home/anon_k/Desktop/Dropbox/Current_research/privacy/DPDR/data/real/census/train.npy")
    else:
        data = np.load(
            "/home/anon_k/Dropbox_from/Current_research/privacy/DPDR/data/real/census/train.npy")

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
    under_sampling_rate = 0.4#undersampled_rate #0.4
    in_keep = in_keep[0:np.int(np.sum(idx_negative_label) * under_sampling_rate)]

    neg_samps_input = neg_samps_input[in_keep, :]
    neg_samps_label = neg_samps_label[in_keep]

    feature_selected = np.concatenate((pos_samps_input, neg_samps_input))
    label_selected = np.concatenate((pos_samps_label, neg_samps_label))

    X_train, X_test, y_train, y_test = train_test_split(feature_selected, label_selected, train_size=0.80,
                                                        test_size=0.20, random_state=seed_number)


    return X_train, y_train, X_test, y_test





def load_intrusion():

    seed_number=0

    print("dataset is intrusion")
    print(socket.gethostname())
    data, categorical_columns, ordinal_columns = sdgym.load_dataset('intrusion')

    """ some specifics on this dataset """
    n_classes = 5 #removed to 5

    """ some changes we make in the type of features for applying to our model """
    categorical_columns_binary = [6, 11, 13, 20]  # these are binary categorical columns
    the_rest_columns = list(set(np.arange(data[:, :-1].shape[1])) - set(categorical_columns_binary))

    num_numerical_inputs = len(the_rest_columns)  # 10. Separately from the numerical ones, we compute the length-scale for the rest columns
    num_categorical_inputs = len(categorical_columns_binary)  # 4.

    raw_labels = data[:, -1]
    raw_input_features = data[:, the_rest_columns + categorical_columns_binary]
    print(raw_input_features.shape)

    #we remove the least label
    non4_tokeep=np.where(raw_labels!=4)[0]
    raw_labels=raw_labels[non4_tokeep]
    raw_input_features=raw_input_features[non4_tokeep]

    """ we take a pre-processing step such that the dataset is a bit more balanced """
    idx_negative_label = raw_labels == 0  # this is a dominant one about 80%, which we want to undersample
    idx_positive_label = raw_labels != 0

    pos_samps_input = raw_input_features[idx_positive_label, :]
    pos_samps_label = raw_labels[idx_positive_label]
    neg_samps_input = raw_input_features[idx_negative_label, :]
    neg_samps_label = raw_labels[idx_negative_label]

    # take random 40% of the negative labelled data
    in_keep = np.random.permutation(np.sum(idx_negative_label))
    under_sampling_rate = 0.3#undersampled_rate#0.3
    in_keep = in_keep[0:np.int(np.sum(idx_negative_label) * under_sampling_rate)]

    neg_samps_input = neg_samps_input[in_keep, :]
    neg_samps_label = neg_samps_label[in_keep]

    feature_selected = np.concatenate((pos_samps_input, neg_samps_input))
    label_selected = np.concatenate((pos_samps_label, neg_samps_label))

    X_train, X_test, y_train, y_test = train_test_split(feature_selected, label_selected, train_size=0.70,
                                                        test_size=0.30,
                                                        random_state=seed_number)

    return X_train, y_train, X_test, y_test

def load_adult():
    seed_number=0
    print("dataset is adult") # this is heterogenous
    print(socket.gethostname())
    #if 'g0' not in socket.gethostname():
    data, categorical_columns, ordinal_columns = sdgym.load_dataset('adult')
    # else:

    """ some specifics on this dataset """
    numerical_columns = list(set(np.arange(data[:, :-1].shape[1])) - set(categorical_columns + ordinal_columns))
    n_classes = 2

    data = data[:, numerical_columns + ordinal_columns + categorical_columns]

    num_numerical_inputs = len(numerical_columns)
    num_categorical_inputs = len(categorical_columns + ordinal_columns) - 1

    inputs = data[:, :-1]
    target = data[:, -1]

    inputs, target=undersample(inputs, target, 0.4)

    X_train, X_test, y_train, y_test = train_test_split(inputs, target, train_size=0.90, test_size=0.10,
                                                        random_state=seed_number)

    return X_train, y_train, X_test, y_test




def load_covtype():

    seed_number=0

    print("dataset is covtype")
    print(socket.gethostname())
    if 'g0' not in socket.gethostname() and 'p0' not in socket.gethostname():
        train_data = np.load("/home/anon_k/Desktop/Dropbox/Current_research/privacy/DPDR/data/real/covtype/train.npy")
        test_data = np.load("/home/anon_k/Desktop/Dropbox/Current_research/privacy/DPDR/data/real/covtype/test.npy")
        # we put them together and make a new train/test split in the following
        data = np.concatenate((train_data, test_data))
    else:
        train_data = np.load(
            "/home/anon_k/Dropbox_from/Current_research/privacy/DPDR/data/real/covtype/train.npy")
        test_data = np.load(
            "/home/anon_k/Dropbox_from/Current_research/privacy/DPDR/data/real/covtype/test.npy")
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
    under_sampling_rate = 0.3#undersampled_rate  # 0.3
    in_keep = in_keep[0:np.int(np.sum(idx_negative_label) * under_sampling_rate)]

    neg_samps_input = neg_samps_input[in_keep, :]
    neg_samps_label = neg_samps_label[in_keep]

    feature_selected = np.concatenate((pos_samps_input, neg_samps_input))
    label_selected = np.concatenate((pos_samps_label, neg_samps_label))


    ###############3

    X_train, X_test, y_train, y_test = train_test_split(feature_selected, label_selected, train_size=0.70, test_size=0.30,
                                                        random_state=seed_number)  # 60% training and 40% test

    return X_train, y_train, X_test, y_test



#################33

n_classes=2

def test_models(X_tr, y_tr, X_te, y_te, datasettype, n_classes=2):

    roc_arr=[]
    prc_arr=[]
    f1_arr=[]

    for model in [xgboost.XGBClassifier(), LogisticRegression(solver='lbfgs', max_iter=1000), GaussianNB(), BernoulliNB(alpha=0.02), LinearSVC(), DecisionTreeClassifier(), LinearDiscriminantAnalysis(), AdaBoostClassifier(), BaggingClassifier(), RandomForestClassifier(), GradientBoostingClassifier(), MLPClassifier()]:
    #for model in [LogisticRegression(solver='lbfgs', max_iter=1000), GaussianNB(), BernoulliNB(alpha=0.02), LinearSVC(), DecisionTreeClassifier(), LinearDiscriminantAnalysis(), AdaBoostClassifier(), BaggingClassifier(), RandomForestClassifier(), GradientBoostingClassifier(), MLPClassifier()]:

    #for model in [LogisticRegression(solver='lbfgs', max_iter=1000), BernoulliNB(alpha=0.02)]:

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

        res1 = np.mean(f1_arr)
        print("f1 mean across methods is %.3f\n" % res1)
        res2 = 0  # dummy
    else:

        res1=np.mean(roc_arr)
        res2=np.mean(prc_arr)
        print("-----\nroc mean across methods is %.3f" % res1)
        print("prc mean across methods is %.3f\n" % res2)


    return res1, res2


#load_epileptic()
#load_adult()_
load_covtype()