import os
import numpy as np
from data_preparation import prepare_the_data
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
__author__ = 'ORI'



def identify_leading_features(data_to_eval):
    from sklearn.lda import LDA
    lda = LDA()
    _ = lda.fit(data_to_eval.data, data_to_eval.labels).predict(data_to_eval.data)
    leading_features = lda.coef_
    for_leading_featues = np.argsort(lda.coef_, axis=1)[:,0:4]
    print for_leading_featues


def identify_steady_colation(data_to_eval):
    from sklearn.lda import LDA
    lda = LDA()
    _ = lda.fit(data_to_eval.data, data_to_eval.labels).predict(data_to_eval.data)
    leading_features = lda.coef_
    for_leading_featues = np.argsort(lda.coef_, axis=1)[:,0:4]
    print for_leading_featues


if __name__ == "__main__":
    # load and prepare the data

    from sklearn import datasets
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


    working_direcotry = r"C:\Users\ORI\Documents\IDC-non-sync\ML_Course\Election\Data\\"
    file_name = os.path.join(working_direcotry, r'ElectionsData.csv')

    train, validation, test, feature_categorical_dictionary, train_idx, test_idx, number_to_party_dictionary = prepare_the_data(file_name,
                                                                                                    working_direcotry)

    identify_leading_features(train)
    # experiment 1: gmm for each party - easy to interpret:
    # I'll begin with one gaussain for each party:

    import colorsys

    def get_N_HexCol(N=5):

        HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in xrange(N)]
        hex_out = []
        for rgb in HSV_tuples:
            rgb = map(lambda x: int(x*255),colorsys.hsv_to_rgb(*rgb))
            # hex_out.append("".join(map(lambda x: chr(x).encode('hex'),rgb)))
            hex_out.append(rgb)
        return hex_out

    color_space = get_N_HexCol(N=5)
    random_numbers = np.random.rand(5,2)



    # cmap=plt.get_cmap(name)
    import matplotlib.cm as cmx
    jet = cm = plt.get_cmap('jet')
    import matplotlib.colors as colors
    # cNorm  = colors.Normalize(vmin=0, vmax=np.max(random_numbers))
    # scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    # colors = [ cm.jet(x) for x in np.linspace(0, 5, 5) ]
    # plt.scatter(random_numbers[:,0], random_numbers[:,1],
    #             c=np.arange(5),
    #             cmap=plt.get_cmap('jet'),
    #             s=40,marker='s',
    #             edgecolors='none', label=[str(x) for x in range(10)])
    # plt.show()

    from sklearn import mixture
    np.random.seed(1)
    model_per_class = dict()
    all_different_parties = np.unique(train.labels)
    for party_id in all_different_parties:
        model = mixture.GMM(n_components=1)
        only_this_party_data = train.data[train.labels == party_id]
        model.fit(only_this_party_data)
        model_per_class[party_id] = model



    #now using kmeans, find the best seperation of the data into two parties
    parties_architype = np.zeros((len(all_different_parties),train.data.shape[1] ))
    for i, class_model in enumerate(model_per_class):
        parties_architype[i, :] = model_per_class[class_model].means_
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_r = pca.fit(train.data).transform(train.data)



    import numpy as np
    import colorsys

    def _get_colors(num_colors):
        colors=[]
        for i in np.arange(0., 360., 360. / num_colors):
            hue = i/360.
            lightness = (50 + np.random.rand() * 10)/100.
            saturation = (90 + np.random.rand() * 10)/100.
            colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
        return colors

    # colors = _get_colors(len(all_different_parties)) #plt.get_cmap('jet')
    # for party_i, party_id in  enumerate(all_different_parties):
    #     plt.scatter(X_r[train.labels == party_id, 0], X_r[train.labels == party_id, 1], c=number_to_party_dictionary[party_id][0:-1],
    #                  s=40,  label=number_to_party_dictionary[party_id])
    #
    # # plt.scatter(X_r[:, 0], X_r[:, 1], c=train.labels, cmap=plt.get_cmap('jet'), s=40,marker='s' ,edgecolors='none')
    #
    # # plt.scatter(X_r[train.labels==1, 0], X_r[train.labels==1, 1], c=colors[1], cmap=plt.get_cmap('jet'), s=40,marker='s' ,edgecolors='none', label='1')
    # # plt.scatter(X_r[train.labels==2, 0], X_r[train.labels==2, 1], c=colors[2], cmap=plt.get_cmap('jet'), s=40,marker='s' ,edgecolors='none', label='2')
    # # plt.scatter(X_r[:, 0], X_r[:, 1], c=train.labels, cmap=plt.get_cmap('jet'), s=40,marker='s' ,edgecolors='none')
    # plt.legend()
    # plt.show()

    # train clustering model

    from sklearn.naive_bayes import GaussianNB
    from sklearn.lda import LDA
    gnb = GaussianNB()
    lda = LDA()
    y_pred = lda.fit(train.data, train.labels).predict(train.data)
    np.argsort(lda.coef_, axis=1)

    clustering_model = KMeans(init='k-means++', n_clusters=10, n_init=10)
    clustering_model.fit(train.data)




    prediction_results = clustering_model.predict(train.data)
    bin_res = np.bincount(prediction_results)
    plt.hist(prediction_results)
    plt.show()
    # measurement 1: histogram of the party in each cluster
    # measurement 2: number of voters
    # I want an area  with the most number of different parties
    # distance from other group






    print "dfs"
    # vote_int_to_name = feature_categorical_dictionary['Vote']
    #
    # # examine the two classifiers
    # models = [DecisionTreeClassifier(), LDA()]
    #
    # prediction_results = dict()
    # for model in models:
    #     evaluation = evaluate_model(model, train.data, test.data, train.labels, test.labels, vote_int_to_name)
    #     prediction_results[model.__class__.__name__] = evaluation
    #
    # # selected model - DecisionTreeClassifier:
    # with open(os.path.join(r'C:\Users\ORI\Documents\IDC-non-sync\ML_Course\Election\Data\output', 'prediction_results.csv'), 'wb') as csvfile:
    #     cvs_writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    #     for idx, vote in zip(test_idx, prediction_results["DecisionTreeClassifier"].test_prediction):
    #         cvs_writer.writerow([idx, vote_int_to_name[vote]])
    #
    # division_of_voters = predictio
