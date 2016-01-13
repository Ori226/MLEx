__author__ = 'ORI'


# needed imports
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np


def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata
if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)  # suppress scientific float notation


    # generate two clusters: a with 100 points, b with 50:
    np.random.seed(4711)  # for repeatability of this tutorial
    a = np.random.multivariate_normal([10, 0], [[3, 1], [1, 4]], size=[100,])
    b = np.random.multivariate_normal([0, 20], [[3, 1], [1, 4]], size=[50,])
    X = np.concatenate((a, b),)
    import os
    from data_preparation import prepare_the_data
    working_direcotry = r"C:\Users\ORI\Documents\IDC-non-sync\ML_Course\Election\Data\\"
    file_name = os.path.join(working_direcotry, r'ElectionsData.csv')

    train, validation, test, feature_categorical_dictionary, train_idx, test_idx, number_to_party_dictionary = prepare_the_data(file_name,
                                                                                                    working_direcotry)
    X = train.data

    print X.shape  # 150 samples with 2 dimensions
    plt.scatter(X[:,0], X[:,1])
    plt.show()

    # generate the linkage matrix
    Z = linkage(X, 'ward')
    # calculate full dendrogram
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(
        Z,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
    )
    plt.show()


    from scipy.cluster.hierarchy import fcluster
    max_d = 100
    clusters = fcluster(Z, max_d, criterion='distance')
    plt.figure(figsize=(10, 8))
    plt.scatter(X[:,0], X[:,1], c=clusters, cmap='prism')  # plot points with cluster dependent colors
    plt.show()


    pass