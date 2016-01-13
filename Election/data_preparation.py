# coding=utf-8
__author__ = 'ORI'

import pandas as pd
import numpy as np
from sklearn import preprocessing
import os
from sklearn.cross_validation import StratifiedShuffleSplit, LabelKFold, StratifiedKFold


class DataForClassification(object):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels


def convert_categorical(data, feature):
    data[feature] = data[feature].astype("category")
    #     print pd.get_dummies(data[feature]).as_matrix().shape
    data[feature + "Int"] = data[feature].cat.rename_categories(range(data[feature].nunique())).astype(int)
    data.loc[data[feature].isnull(), feature + "Int"] = np.nan  # fix NaN conversion
    return dict(zip(range(data[feature].nunique()), list(data[feature].cat.categories)))


def fill_null_by_mode(data, feature):
    data.loc[data[feature].isnull(), feature] = data[feature].dropna().mode().iloc[0]


def fill_null_by_media(data, feature):
    median = data[feature].dropna().median()
    data.loc[data[feature].isnull(), feature] = median


def fill_outliers_with_nan(data, feature):
    data.loc[((data[feature] - data[feature].mean()) / data[feature].std()).abs() >= 2.5, feature] = np.nan

def prepare_the_data(data_file_name, output_directory):

    original_data_with_labels = pd.read_csv(data_file_name)
    list_of_features = ['Yearly_ExpensesK', 'Yearly_IncomeK', 'Overall_happiness_score', 'Most_Important_Issue',
                        'Avg_Residancy_Altitude', 'Will_vote_only_large_party', 'Financial_agenda_matters', 'Vote']

    original_data = original_data_with_labels[list_of_features].copy()


    # convert to categorical

    ObjFeat = original_data.keys()[original_data.dtypes.map(lambda x: x == 'object')]
    ObjNonCategorical = original_data.keys()[original_data.dtypes.map(lambda x: x != 'object')]


    # extarct 'vote' names
    categorical_dict = dict()

    number_to_party_dictionary=dict([(i, party) for i, party in enumerate(original_data['Vote'].astype("category").cat.categories)])

    for f in ObjFeat:
        feature_categorical_dictionary = convert_categorical(original_data, f)
        categorical_dict[f] = feature_categorical_dictionary
        original_data.drop(f, axis=1, inplace=True)

    # now, fill the null values. start with categorical:
    for f in ObjFeat:
        fill_null_by_mode(original_data, f + "Int")

    # now, fill the null values. fill with median
    for f in ObjNonCategorical:
        fill_null_by_media(original_data, f)




    res = StratifiedKFold(original_data['VoteInt'], 4)
    train_indexes = list()
    [train_indexes.extend(idx[1])  for idx in list(res)[:2]]
    validation_indexes = list(res)[2][1]
    test_indexes = list(res)[3][1]


    train = original_data.iloc[train_indexes, :]




    validation = original_data.iloc[validation_indexes, :]
    test = original_data.iloc[test_indexes, :]

    train.to_csv(os.path.join(output_directory, r"ElectionsData_train.csv"), index=False)
    validation.to_csv(os.path.join(output_directory, r"ElectionsData_validation.csv"), index=False)
    test.to_csv(os.path.join(output_directory, r"ElectionsData_test.csv"), index=False)


    X_train, X_valid, X_test = [data_set.drop('VoteInt', 1).as_matrix() for data_set in [train, validation, test]]
    y_train, y_valid, y_test = [data_set['VoteInt'].as_matrix() for data_set in [train, validation, test]]

    train_data = DataForClassification(data=X_train, labels=y_train)
    validation_data = DataForClassification(data=X_valid, labels=y_valid)
    test_data = DataForClassification(data=X_test, labels=y_test)
    return train_data, validation_data, test_data, categorical_dict, train_indexes, test_indexes, number_to_party_dictionary


if __name__ == "__main__":
    working_direcotry = r"C:\Users\ORI\Documents\IDC-non-sync\ML_Course\Election\Data\\"
    file_name = os.path.join(working_direcotry, r'ElectionsData.csv')

    prepare_the_data(file_name, working_direcotry)

