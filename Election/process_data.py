import pandas as pd
import numpy as np


def convert_categorical(data, feature):
    print feature
    data[feature] = original_data[feature].astype("category")
#     print pd.get_dummies(data[feature]).as_matrix().shape
    data[feature+"Int"] = data[feature].cat.rename_categories(range(data[feature].nunique())).astype(int)
    data.loc[data[feature].isnull(), feature+"Int"] = np.nan #fix NaN conversion


def fill_null_by_mode(data, feature):
    data.loc[data[feature].isnull(), feature] = data[feature].dropna().mode().iloc[0]

def fill_null_by_media(data, feature):
    median = data[feature].dropna().median()
    data.loc[data[feature].isnull(), feature] = median

def fill_outliers_with_nan(data, feature):
    data.loc[((data[feature] - data[feature].mean()) / data[feature].std()).abs() >= 2.5, feature] = np.nan





if __name__ == "__main__":
    working_dir = r"C:\Users\ORI\Documents\IDC-non-sync\ML_Course\Election\Data\\"
    original_data = pd.read_csv(working_dir +  r'ElectionsData.csv')

    ObjFeat=original_data.keys()[original_data.dtypes.map(lambda x: x=='object')]
    ObjNonCategorical=original_data.keys()[original_data.dtypes.map(lambda x: x <>'object')]
    # print "non categorical = " + ObjNonCategorical
    for f in ObjFeat:
        convert_categorical(original_data, f)

    # now, fill the null values. start with categorical:






    for f in ['AVG_lottary_expanses', 'Avg_monthly_expense_when_under_age_21', 'Avg_Residancy_Altitude']:
        original_data.loc[original_data[f] < 0, f] = np.nan





    for f in ObjFeat:
        fill_null_by_mode(original_data, f+"Int")

    from sklearn import preprocessing


    for f in ObjNonCategorical:
        fill_null_by_media(original_data, f)

    #     X_scaled = preprocessing.scale(original_data[f].as_matrix())
    #     original_data[f] = X_scaled



    # and remove the old categories
    for f in ObjFeat:
        original_data.drop(f, axis=1, inplace=True)

    original_data.info()