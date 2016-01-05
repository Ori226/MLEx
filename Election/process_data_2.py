# import string
#
# try:
#     maketrans = ''.maketrans
# except AttributeError:
#     # fallback for Python 2
#     from string import maketrans

import numpy as np
import pandas as pd
import pylab as P


if __name__ == "__main__":


    working_dir = r"C:\Users\ORI\Documents\IDC-non-sync\ML_Course\Election\Data\\"

    original_data = pd.read_csv(working_dir +  r'ElectionsData.csv')
    # print original_data.info()



    # now, create a copy of the data:
    original_data.to_csv(working_dir+r"output\copy_ElectionsData.csv", index=False)

    # Identify which of the orginal features are objects
    ObjFeat=original_data.keys()[original_data.dtypes.map(lambda x: x=='object')]
    # print ObjFeat
    # Transform the original features to categorical
    # Creat new 'int' features, resp.

    categorical_fields = []

    for f in ObjFeat:
        original_data[f] = original_data[f].astype("category")
        temp = original_data[f].cat.rename_categories(range(original_data[f].nunique())).astype(int)
        original_data[f+"Int"] = original_data[f].cat.rename_categories(range(original_data[f].nunique())).astype(int)
        original_data.loc[original_data[f].isnull(), f+"Int"] = np.nan #fix NaN conversion
        original_data[f] = original_data[f+"Int"]
    #     original_data[f] = original_data[f].astype("category")
        categorical_fields.append(f)
        original_data.drop(f+"Int",inplace=True,axis=1)



    # fill missing value


    # print original_data['Occupation_Satisfaction'][30:35]
    # print original_data['Occupation_Satisfaction'].index[original_data['Occupation_Satisfaction'].apply(np.isnan)]

    # index = original_data.index[original_data.apply(np.isnan)]

    # %matplotlib inline
    # print np.where(original_data['Occupation_Satisfaction'].isnull() == True)[0]
    original_data['Occupation_Satisfaction'].hist(alpha=0.5)
    P.show()








    for f in categorical_fields:
        catgory_mode = original_data[f].dropna().mode()[0]
        median_value = catgory_mode
    #     print "{0}:{1}".format(f, median_value)
    #     original_data[f].loc[original_data[f].isnull()==True] = int(median_value)
        original_data[f] = original_data[f].fillna(original_data[f].mode().iloc[0])

    print np.where(original_data['Vote'].isnull)
    for f in original_data.keys():
        if f in categorical_fields:
            continue
        else:
            original_data[f] = original_data[f].fillna(original_data[f].dropna().median())
    #         catgory_median = original_data[f].dropna().median()

    #         print f + "xxx"
    #         original_data[f] = original_data[f].fillna(3)

    print "yyy "+ str(original_data['Occupation_Satisfaction'].dropna().median())
    # original_data['Occupation_Satisfaction'] = original_data['Occupation_Satisfaction'].fillna(original_data['Occupation_Satisfaction'].dropna().median())

    # original_data['Occupation_Satisfaction'].loc[original_data['Occupation_Satisfaction'].isnull()==True] = int(6)

    # for f in categorical_fields:
    #     catgory_mode = original_data[f].dropna().mode()[0]
    #     median_value = catgory_mode
    # #     print "{0}:{1}".format(f, median_value)
    #     original_data.loc[original_data[f].isnull()] = int(median_value)


    # for f in original_data.keys():
    #     if f in categorical_fields:
    #         continue
    #     else:
    #         catgory_median = original_data[f].dropna().median()
    #         original_data.loc[original_data[f].isnull()] = median_value
    print original_data['Occupation_Satisfaction'][30:35]
    original_data['Occupation_Satisfaction'].hist(alpha=0.5)

    P.show()