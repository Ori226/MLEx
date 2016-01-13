__author__ = 'ORI'



# Authors: Gael Varoquaux
# License: BSD 3 clause (C) INRIA 2014

print(__doc__)
from time import time

import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt

from sklearn import manifold, datasets
import sys
sys.path.append(r'C:\Users\ORI\Documents\IDC-non-sync\ML_Course\Election')
from sklearn.cluster import AgglomerativeClustering
from data_preparation import prepare_the_data
import os

def find_steady_coalition():

    working_direcotry = r"C:\Users\ORI\Documents\IDC-non-sync\ML_Course\Election\Data\\"
    file_name = os.path.join(working_direcotry, r'ElectionsData.csv')
    train, validation, test, feature_categorical_dictionary, train_idx, test_idx, number_to_party_dictionary = prepare_the_data(file_name,

                                                                                                        working_direcotry)

    good_colation_found = False
    for n_clusters in [5,4,3]:
        print ("---------------")
        linkage = 'ward'
        X = train.data
        clusters = AgglomerativeClustering(linkage=linkage, n_clusters=n_clusters)
        clusters.fit_predict(X)
        bin_count_of_kmeans_clusters = np.bincount(clusters.labels_)
        normalized_bin_count_of_kmeans_clusters = bin_count_of_kmeans_clusters/np.sum(bin_count_of_kmeans_clusters).astype('float32')
        #is there any cluster with more than 50% of the votes?
        coalition_exists = np.any(normalized_bin_count_of_kmeans_clusters > 0.5)
        print "number_of_clustes {0}".format(n_clusters)
        print "coalition_exists: {0} ".format(coalition_exists)

        # find all the parties belong to the cluster
        biggest_cluster = np.argmax(normalized_bin_count_of_kmeans_clusters)
        biggest_cluster_voters = np.bincount(train.labels[clusters.labels_ == biggest_cluster].astype('int64'))

        #normalize the votes by the size of their parties:
        votes_out_of_party =  biggest_cluster_voters/np.bincount( train.labels.astype('int32')).astype('float32')
        #commited_to_coalition_parties = partyw with majority of the  votes in the cluster
        commited_to_coalition_parties = votes_out_of_party > 0.5

        percentage_of_voters_in_commited_coalition = np.sum(biggest_cluster_voters[votes_out_of_party > 0.5])*1.0/len(train.labels)*1.0

        print percentage_of_voters_in_commited_coalition
        if percentage_of_voters_in_commited_coalition> 0.5:
            print "coalition found"
            parties_in_coalition = number_to_party_dictionary.keys()
            print "parties in coalition:{0}".format([number_to_party_dictionary[k] for k in  np.array(number_to_party_dictionary.keys())[votes_out_of_party > 0.5]])

            break
        print ("---------------")
if __name__ == "__main__":
    find_steady_coalition()

