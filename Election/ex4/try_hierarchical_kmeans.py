__author__ = 'ORI'

# Authors: Gael Varoquaux
# License: BSD 3 clause (C) INRIA 2014

print(__doc__)
from time import time

import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt

from sklearn import manifold, datasets

digits = datasets.load_digits(n_class=10)
X = digits.data
y = digits.target
n_samples, n_features = X.shape

np.random.seed(0)

def nudge_images(X, y):
    # Having a larger dataset shows more clearly the behavior of the
    # methods, but we multiply the size of the dataset only by 2, as the
    # cost of the hierarchical clustering methods are strongly
    # super-linear in n_samples
    shift = lambda x: ndimage.shift(x.reshape((8, 8)),
                                  .3 * np.random.normal(size=2),
                                  mode='constant',
                                  ).ravel()
    X = np.concatenate([X, np.apply_along_axis(shift, 1, X)])
    Y = np.concatenate([y, y], axis=0)
    return X, Y


X, y = nudge_images(X, y)


#----------------------------------------------------------------------
# Visualize the clustering
def plot_clustering(X_red, X, labels, title=None):
    x_min, x_max = np.min(X_red, axis=0), np.max(X_red, axis=0)
    X_red = (X_red - x_min) / (x_max - x_min)

    plt.figure(figsize=(6, 4))
    for i in range(X_red.shape[0]):
        plt.text(X_red[i, 0], X_red[i, 1], str(y[i]),
                 color=plt.cm.spectral(labels[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title, size=17)
    plt.axis('off')
    plt.tight_layout()

#----------------------------------------------------------------------
# 2D embedding of the digits dataset
print("Computing embedding")

print("Done.")

from sklearn.cluster import AgglomerativeClustering
from data_preparation import prepare_the_data
import os

# for linkage in ('ward', 'average', 'complete'):
linkage = 'ward'

working_direcotry = r"C:\Users\ORI\Documents\IDC-non-sync\ML_Course\Election\Data\\"
file_name = os.path.join(working_direcotry, r'ElectionsData.csv')
train, validation, test, feature_categorical_dictionary, train_idx, test_idx, number_to_party_dictionary = prepare_the_data(file_name,
                                                                                                    working_direcotry)
print "here"
X = train.data
X_red = manifold.SpectralEmbedding(n_components=2).fit_transform(X)

print "here 1"
clustering = AgglomerativeClustering(linkage=linkage, n_clusters=10)
t0 = time()
cluster_results = clustering.fit_predict(X)
print("%s : %.2fs" % (linkage, time() - t0))

print "here 2"


plot_clustering(X_red, X, clustering.labels_, "%s linkage" % linkage)


plt.show()
