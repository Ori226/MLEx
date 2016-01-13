# coding=utf-8

import csv
import os

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from data_preparation import prepare_the_data

__author__ = 'ORI'


class ClassifierEvaluation(object):
    pass


def get_winning_party(votes):
    votes_distribution = np.bincount(votes.astype('int8'))
    return np.argmax(votes_distribution)


def plot_confusion_matrix(cm, category_labels, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(category_labels))
    plt.xticks(tick_marks, category_labels, rotation=45)
    plt.yticks(tick_marks, category_labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def evaluate_model(_model, train_data, test_data, train_data_labels, test_data_labels, labels_name):
    # cross validation on the winning party using the training set

    number_of_k_folds = 10
    kf = StratifiedKFold(train_data_labels, number_of_k_folds)
    number_of_success_predcition = 0
    cv_distances = []
    model_name = _model.__class__.__name__
    print "********{0}***********".format(model_name)
    for train, test in kf:
        _model.fit(train_data[train], train_data_labels[train])
        prediction_results = _model.predict(train_data[test])
        predicted_winning_party = get_winning_party(prediction_results)
        actual_winning_party = get_winning_party(train_data_labels[test])
        if predicted_winning_party == actual_winning_party:
            number_of_success_predcition += 1

        cv_precited_votes_distribution = np.bincount(prediction_results.astype('int8'), minlength=10)
        cv_actual_votes_distribution = np.bincount(train_data_labels[test].astype('int8'), minlength=10)
        cv_distances.append(np.sqrt(np.sum(np.power(cv_precited_votes_distribution - cv_actual_votes_distribution, 2))))

    cv_distances = np.asarray(cv_distances)
    cv_distribution_mean = cv_distances.mean()
    cv_distribution_std = cv_distances.std()

    _model.fit(train_data, train_data_labels)
    prediction_on_test_data = _model.predict(test_data)
    predicted_winning_party_on_test = get_winning_party(prediction_on_test_data)
    actual_winning_party_on_test = get_winning_party(test_data_labels)

    precited_votes_distribution = np.bincount(prediction_on_test_data.astype('int8'), minlength=10)
    actual_votes_distribution = np.bincount(test_data_labels.astype('int8'), minlength=10)
    distance = np.sqrt(np.sum(np.power(precited_votes_distribution - actual_votes_distribution, 2)))

    claasifier_evaluation = ClassifierEvaluation()
    claasifier_evaluation.model = _model

    claasifier_evaluation.predicted_winning_party = predicted_winning_party_on_test
    claasifier_evaluation.cv_mean = cv_distribution_mean
    claasifier_evaluation.cv_std = cv_distribution_std
    claasifier_evaluation.cv_percentage_of_correct_prediction = 1.0 * number_of_success_predcition / number_of_k_folds
    claasifier_evaluation.test_data_distribution_distance = distance
    claasifier_evaluation.test_data_good_prediction = actual_winning_party_on_test == predicted_winning_party_on_test

    claasifier_evaluation.total_number_of_error = 1.0 * np.sum(prediction_on_test_data != test_data_labels) / \
                                                  test_data_labels.shape[0]
    claasifier_evaluation.total_number_success = 1.0 * np.sum(prediction_on_test_data == test_data_labels) / \
                                                 test_data_labels.shape[0]
    for k in claasifier_evaluation.__dict__:
        print "{0}, {1}".format(k, claasifier_evaluation.__dict__[k])

    claasifier_evaluation.test_prediction = prediction_on_test_data
    claasifier_evaluation.precited_votes_distribution = 1.0 * precited_votes_distribution / np.sum(
        precited_votes_distribution)
    cm = confusion_matrix(test_data_labels.astype('int16'), prediction_on_test_data.astype('int16'))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plot_confusion_matrix(cm_normalized, labels_name.values(), title=model_name)
    plt.show()
    return claasifier_evaluation


if __name__ == "__main__":
    # load and prepare the data
    working_direcotry = r"C:\Users\ORI\Documents\IDC-non-sync\ML_Course\Election\Data\\"
    file_name = os.path.join(working_direcotry, r'ElectionsData.csv')

    train, validation, test, feature_categorical_dictionary, train_idx, test_idx, _ = prepare_the_data(file_name,
                                                                                                    working_direcotry)
    vote_int_to_name = feature_categorical_dictionary['Vote']

    # examine the two classifiers
    models = [DecisionTreeClassifier(), LDA()]

    prediction_results = dict()
    for model in models:
        evaluation = evaluate_model(model, train.data, test.data, train.labels, test.labels, vote_int_to_name)
        prediction_results[model.__class__.__name__] = evaluation

    # selected model - DecisionTreeClassifier:
    with open(os.path.join(r'C:\Users\ORI\Documents\IDC-non-sync\ML_Course\Election\Data\output', 'prediction_results.csv'), 'wb') as csvfile:
        cvs_writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for idx, vote in zip(test_idx, prediction_results["DecisionTreeClassifier"].test_prediction):
            cvs_writer.writerow([idx, vote_int_to_name[vote]])

    division_of_voters = prediction_results["DecisionTreeClassifier"].precited_votes_distribution
    print ["{0} {1:0.2f}".format(vote_int_to_name[i], p) for i, p in enumerate(division_of_voters)]
    # plot the division of voters on the test data
    plt.clf()
    import matplotlib.pyplot as plt

    labels = ["{0}: {1}%".format(vote_int_to_name[i], percentage * 100) for i, percentage in
              enumerate(division_of_voters)]

    patches, texts = plt.pie(division_of_voters, startangle=90)
    plt.legend(patches, labels, loc="best")

    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    pass
