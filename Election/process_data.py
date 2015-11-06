import pandas as pd
import numpy as np



if __name__ == "__main__":

    df = pd.read_csv(r'C:\Users\ORI\Documents\IDC-non-sync\ML_Course\Election\Data\ElectionsData.csv')

    # import matplotlib
    # import matplotlib.pyplot as plt
    for col in df.columns:
        if df[col].dtype == "object":
            df[col]= df[col].astype("category")

    # df["Vote"] = df["Vote"].astype("category")
    print df.Vote.cat.categories
    df_processed = df.copy() #    df.Vote.cat.categories
    df_processed["Vote"][0] = pd.Categorical(["Greens"], categories=df_processed["Vote"].cat.categories)
    print df_processed["Vote"][0]
    print df["Vote"][0]
    print df_processed['Occupation_Satisfaction'][33:35]
    df_processed['Occupation_Satisfaction'] = df_processed['Occupation_Satisfaction'].fillna(0)

    print df_processed['Occupation_Satisfaction'][33:35]
    print "mode is {0}".format(df["Vote"].mode())
    print pd.isnull(df_processed['Occupation_Satisfaction'])
    # inds = pd.isnull(df['Occupation_Satisfaction']).any(1).nonzero()[0]
    inds =  pd.isnull(df['Occupation_Satisfaction'][33:35]).select(lambda x: x  , axis=0)
    print "inds: {0}".format(inds)

    print "inds2: {0}".format(df['Will_vote_only_large_party'].index[pd.isnull(df['Will_vote_only_large_party']) ])
    #df_processed['Will_vote_only_large_party'] = \
    df_processed['Will_vote_only_large_party'] = df_processed['Will_vote_only_large_party'].fillna(df["Will_vote_only_large_party"].mode()[0])
    print "inds2: {0}".format(df_processed['Will_vote_only_large_party'].index[pd.isnull(df_processed['Will_vote_only_large_party']) ])


    # print pd.isnull(df['Vote'])[:]

    # print "inds: {0}".format(range(len(inds)))


    # fill up missing data
    # for each categorical, find the mode
    # for others find the average




