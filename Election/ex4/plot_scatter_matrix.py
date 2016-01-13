__author__ = 'ORI'
import os
from data_preparation import prepare_the_data
import numpy as np
import colorsys
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier


def get_winning_party(votes):
    votes_distribution = np.bincount(votes.astype('int8'))
    return np.argmax(votes_distribution)

if __name__ == "__main__":

    # load and prepare the data

    from sklearn import datasets
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


    working_direcotry = r"C:\Users\ORI\Documents\IDC-non-sync\ML_Course\Election\Data\\"
    file_name = os.path.join(working_direcotry, r'ElectionsData.csv')

    train, validation, test, feature_categorical_dictionary, train_idx, test_idx, number_to_party_dictionary = prepare_the_data(file_name,
                                                                                                    working_direcotry)


    def _get_colors(num_colors):
        colors=[]
        for i in np.arange(0., 360., 360. / num_colors):
            hue = i/360.
            lightness = (50 + np.random.rand() * 10)/100.
            saturation = (90 + np.random.rand() * 10)/100.
            colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
        return colors
    all_different_parties = np.unique(train.labels)
    colors = _get_colors(len(all_different_parties)) #plt.get_cmap('jet')

    counter = 1
    headers = ['Yearly_ExpensesK', 'Yearly_IncomeK', 'Overall_happiness_score', 'Avg_Residancy_Altitude', 'Most_Important_IssueInt', 'Will_vote_only_large_partyInt', 'Financial_agenda_mattersInt']

    for feature_i in range(7):
        for feature_j in range(feature_i,7):
            X_r = train.data[:, feature_j]
            Y_r = train.data[:, feature_i]
            # plt.subplot(7,7,feature_i*7 + feature_j +1)
            counter += 1
            for party_i, party_id in  enumerate(all_different_parties):

                plt.scatter(X_r[train.labels == party_id], Y_r[train.labels == party_id], c=number_to_party_dictionary[party_id][0:-1],
                             s=10,  label=number_to_party_dictionary[party_id], edgecolors='none')

                # plt.title(headers[feature_j])
                # plt.xlabel(headers[feature_j])
                # plt.ylabel(headers[feature_i],fontsize=10)
            plt.show()
            pass

    plt.subplots_adjust(hspace=0.2, wspace=0.2)
    plt.show()
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=1.0)
    train.data
    _model = DecisionTreeClassifier()
    _model.fit(train.data, train.labels)

    original_test_data = test.data
    manipulated_test_data = original_test_data.copy()
    manipulated_test_data[:, 3] = 600

    prediction_results = _model.predict(original_test_data)

    winning_without_mainpulation =  get_winning_party(prediction_results)

    prediction_results_after_manipulation = _model.predict(manipulated_test_data)
    winning_with_mainpulation =  get_winning_party(prediction_results_after_manipulation)


    pass
