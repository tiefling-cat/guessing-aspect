#! /usr/bin/python3

import logging, warnings, sys, os
from optparse import OptionParser
from time import time
import numpy as np
import matplotlib.pyplot as plt
from ngrams import aspect_dataset, load_test_data

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.utils.extmath import density
from sklearn.cross_validation import train_test_split, KFold
from sklearn import metrics

classifiers_n = 13
folds_count = 10
xdf = 0.5
ndf = 1
tokpat = '[^ ]+'
sw = None
voc = None

dfname = 'aspect.csv'

class ZeroR(BaseEstimator, ClassifierMixin):
    """
    Majority class of the training data.
    """
    def __init__(self):
        pass

    def fit(self, X, y):
        self.majority_ = np.argmax(np.bincount(y))
        return self

    def predict(self, X):
        return np.repeat(self.majority_, X.shape[0])

class L1LinearSVC(LinearSVC):
    def fit(self, X, y):
        # The smaller C, the stronger the regularization.
        # The more regularization, the more sparsity.
        self.transformer_ = LinearSVC(penalty="l1", dual=False, tol=1e-3)
        X = self.transformer_.fit_transform(X, y)
        return LinearSVC.fit(self, X, y)

    def predict(self, X):
        X = self.transformer_.transform(X)
        return LinearSVC.predict(self, X)

classifier_list = [
    "ZeroR",
    "Ridge",
    "Perceptron",
    "Passive-Aggressive",
    "kNN",
    "LinearSVC1",
    "LinearSVC2",
    "SGD1",
    "SGD2",
    "SGDen",
    "NearestCentroid",
    "L1LinearSVC"
]

classifiers = {
    "ZeroR":(ZeroR(), "Zero Rule"),
    "Ridge":(RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
    "Perceptron":(Perceptron(n_iter=50), "Perceptron"),
    "Passive-Aggressive":(PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
    "kNN":(KNeighborsClassifier(n_neighbors=10), "kNN"),
    "LinearSVC1":(LinearSVC(loss='l2', penalty="l1", dual=False, tol=1e-3), "LinearSVC with l1 penalty"),
    "LinearSVC2":(LinearSVC(loss='l2', penalty="l2", dual=False, tol=1e-3), "LinearSVC with l2 penalty"),
    "SGD1":(SGDClassifier(alpha=.0001, n_iter=50, penalty="l1"), "SGD with l1 penalty"),
    "SGD2":(SGDClassifier(alpha=.0001, n_iter=50, penalty="l2"), "SGD with l2 penalty"),
    "SGDen":(SGDClassifier(alpha=.0001, n_iter=50, penalty="elasticnet"), "SGD with elasticnet penalty"),
    "NearestCentroid":(NearestCentroid(), "Nearest Centroid"),
    "L1LinearSVC":(L1LinearSVC(), "L1-based LinearSVC")
}

def get_options():
    """
    Parse commandline arguments.
    """
    usage = "USAGE: ./%prog [OPTIONS] [INPUT_FILE]"
    op = OptionParser(usage=usage)
    op.add_option("-o", "--output", type=str,
                action="store", dest="ofname", default=None, metavar="OUTPUT_FILE",
                help="Output results to OUTPUT_FILE. Default is INPUT_FILE_classified_by_CLASSIFIER_NAME.csv")
    op.add_option("-c", "--classifier", type=str,
                action="store", dest="classifier_name", default="LinearSVC1",
                metavar="CLASSIFIER_NAME", help="Specify classifier (LinearSVC1 is default)")
    op.add_option("-l", "--list-classifiers", dest="list_classifiers",
                action="store_true", help="List available classifiers")
    op.add_option("-b", "--benchmark", dest="benchmark",
                action="store_true", help="Run classifier benchmarks on training data")
    op.add_option("-t", "--top10",
                action="store_true", dest="print_top10",
                help="Print ten most discriminative n-grams per class"
                    " for every classifier")
    op.add_option("-s", "--silent",
                action="store_true",
                help="Supress output")
    op.add_option("-m", "--benchmark-log", type=str,
                action="store", dest="lfname", default=None, metavar="LOG_FILE",
                help="Output all benchmark info to LOG_FILE")
    op.add_option("-p", "--plot-output", type=str,
                action="store", dest="pfname", default=None, metavar="GRAPH_OUTPUT_FILE",
                help="Save benchmark diagram to GRAPH_OUTPUT_FILE. File extension must be either png or pdf")

    (opts, args) = op.parse_args()

    if opts.list_classifiers:
        for classifier in classifier_list:
            print(classifier)
        sys.exit(0)

    if opts.classifier_name not in classifier_list:
        op.error("unknown classifier.")
        op.print_help()
        sys.exit(1)

    if not opts.benchmark:
        if len(args) == 0:
            op.error("missing input file name.")
            op.print_help()
            sys.exit(1)
        if len(args) > 1:
            op.error("too many arguments.")
            op.print_help()
            sys.exit(1)
        ifname = args[0]
        if opts.ofname is None:
            opts.ofname = '{}_classified_by_{}.csv'.format(os.path.splitext(ifname)[0], opts.classifier_name)
    else:
        if len(args) > 0:
            op.error("too many arguments.")
            op.print_help()
            sys.exit(1)
        ifname = None

    if opts.pfname is not None and os.path.splitext(opts.pfname)[-1].lower() not in ['.png', '.pdf']:
        opts.pfname = opts.pfname + '.png'

    return opts, ifname

def extract_features(X_train, X_test, silent, print_top10):
    """
    Extract features from the dataset.

    Note that entry for each verb is treated like a text
    with n-grams being treated like words in it.
    We then vectorize those texts, tfidf and all.
    """
    if not silent:
        print("Extracting features from the training dataset using a sparse vectorizer")
    t0 = time()
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=xdf, min_df=ndf, token_pattern=tokpat, stop_words=sw, vocabulary=voc)
    X_tfidf_train = vectorizer.fit_transform(X_train)
    duration = time() - t0
    if not silent:
        print("done in %fs" % duration)
        print("n_samples: %d, n_features: %d" % X_tfidf_train.shape)
        print()
        print("Extracting features from the test dataset using the same vectorizer")

    t0 = time()
    X_tfidf_test = vectorizer.transform(X_test)
    duration = time() - t0
    if not silent:
        print("done in %fs" % duration)
        print("n_samples: %d, n_features: %d" % X_tfidf_test.shape)
        print()

    # mapping from integer feature name to original token string
    if not print_top10:
        feature_names = None
    else:
        feature_names = np.asarray(vectorizer.get_feature_names())

    return X_tfidf_train, X_tfidf_test, feature_names

def benchmark(clf, clf_descr, X_train, X_test, y_train, y_test, feature_names, categories, silent, print_top10):
    """
    Benchmark a classifier.
    """
    if not silent:
        print('_' * 80)
        print("Training: ")
    if (not silent) or print_top10:
        print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    if not silent:
        print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    if not silent:
        print("test time:  %0.3fs" % test_time)

    #score = metrics.f1_score(y_test, pred)
    score = np.mean(pred == y_test)
    if not silent:
        print("accuracy:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        if not silent:
            print("dimensionality: %d" % clf.coef_.shape[1])
            print("density: %f" % density(clf.coef_))

        if print_top10 and feature_names is not None:
            print("top 10 keywords per class:")
            #print(categories)
            if len(categories) > 2: # multi-class
                for i, category in enumerate(categories):
                    top10 = np.argsort(clf.coef_[i])[-10:]
                    print("%s: %s" % (category, " ".join(feature_names[top10])))
            else: # binary
                top10 = np.argsort(clf.coef_[0])[-10:]
                print("%s" % (" ".join(feature_names[top10])))
            print()

    #clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time, pred

def benchmark_all(X_train, X_test, y_train, y_test, feature_names, categories, silent, print_top10):
    """
    Benchmark all classifiers.
    """
    results=[]

    # Train ZeroR
    if not silent:
        print('=' * 80)
        print("Zero Rule")
    results.append(benchmark(ZeroR(), "ZeroR", X_train, X_test, y_train, y_test, feature_names, categories, silent, print_top10))

    for clf, name in (
            (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
            (Perceptron(n_iter=50), "Perceptron"),
            (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
            (KNeighborsClassifier(n_neighbors=10), "kNN")):
        if not opts.silent:
            print('=' * 80)
            print(name)
        results.append(benchmark(clf, name, X_train, X_test, y_train, y_test, feature_names, categories, silent, print_top10))

    for penalty in ["l2", "l1"]:
        if not opts.silent:
            print('=' * 80)
            print("%s penalty" % penalty.upper())
        # Train Liblinear model
        results.append(benchmark(LinearSVC(loss='l2', penalty=penalty, dual=False, tol=1e-3), "LinearSVC {} penalty".format(penalty), X_train, X_test, y_train, y_test, feature_names, categories, silent, print_top10))

        # Train SGD model
        results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50, penalty=penalty), "SGD {} penalty".format(penalty), X_train, X_test, y_train, y_test, feature_names, categories, silent, print_top10))

    # Train SGD with Elastic Net penalty
    if not silent:
        print('=' * 80)
        print("Elastic-Net penalty")
    results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50, penalty="elasticnet"), "SGD elasticnet penalty", X_train, X_test, y_train, y_test, feature_names, categories, silent, print_top10))

    # Train NearestCentroid without threshold
    if not silent:
        print('=' * 80)
        print("NearestCentroid (aka Rocchio classifier)")
    results.append(benchmark(NearestCentroid(), "NearestCentroid", X_train, X_test, y_train, y_test, feature_names, categories, silent, print_top10))

    # Train sparse Naive Bayes classifiers
    if not silent:
        print('=' * 80)
        print("Naive Bayes")
    results.append(benchmark(MultinomialNB(alpha=.01), "MultinomialNB", X_train, X_test, y_train, y_test, feature_names, categories, silent, print_top10))
    results.append(benchmark(BernoulliNB(alpha=.01), "BernoulliNB", X_train, X_test, y_train, y_test, feature_names, categories, silent, print_top10))

    if not silent:
        print('=' * 80)
        print("LinearSVC with L1-based feature selection")
    results.append(benchmark(L1LinearSVC(), "L1-based LinearSVC", X_train, X_test, y_train, y_test, feature_names, categories, silent, print_top10))
    return results

def plot_all(results, indices, pfname):
    """
    Plot all results.
    """
    clf_names, score, training_time, test_time = results
    training_time = np.array(training_time) / np.max(training_time) * 100.
    test_time = np.array(test_time) / np.max(test_time) * 100.
    score = np.array(score) * 100.

    font = {'family' : 'serif',
            #'weight' : 'bold',
            'size'   : 22}

    plt.rc('font', **font)
    plt.rcParams['axes.facecolor'] = 'ffffe6'
    plt.rcParams['savefig.facecolor'] = 'ffffe6'

    plt.figure(figsize=(20, 10))
    plt.title("%")
    plt.barh(indices, score, .2, label="accuracy", color='m')
    plt.barh(indices + .3, training_time, .2, label="training time", color='c')
    #plt.barh(indices + .6, test_time, .2, label="test time", color='y')
    plt.yticks(())
    xlabels = ['0'] + list(np.hstack(tuple([[''] * 4 + [str((i + 1) * 5)] for i in range(20)])))
    plt.xticks(range(101), xlabels)
    plt.legend(loc='best')
    plt.subplots_adjust(left=.25)
    plt.subplots_adjust(top=.95)
    plt.subplots_adjust(bottom=.05)
    plt.grid()
    for i, c in zip(indices, clf_names):
        plt.text(-30, i + .15, c)
    if pfname is None:
        plt.show()
    else:
        plt.savefig(pfname, bbox_inches='tight')

def cross_validate(X, y, categories, count, silent, print_top10):
    """
    Make cross-validation benchmarks for all classifiers.
    """
    folds = KFold(X.size, n_folds=count, shuffle=True)  
    i, total_results = 1, []
    for train_index, test_index in folds:
        print("Cross-validation fold {}".format(i))
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        X_tfidf_train, X_tfidf_test, feature_names = extract_features(X_train, X_test, silent, print_top10)
        results = benchmark_all(X_tfidf_train, X_tfidf_test, y_train, y_test, feature_names, categories, silent, print_top10)
        total_results.extend(results)
        i += 1
    classifier_dict = {}
    for bench in total_results:
        classifier = bench[0]
        classifier_dict[classifier] = classifier_dict.get(classifier, np.array([0.] * 3)) + np.array(bench[1:-1])
    for classifier in classifier_dict:
        classifier_dict[classifier] = classifier_dict[classifier] / count
    mean_results = [[classifier] + list(classifier_dict[classifier]) for classifier in classifier_dict]
    mean_results.sort(key=lambda x: x[0])
    return mean_results

def aspect_detection(ifname, ofname, clf_name, dfname, silent=True, print_top10=False):
    """
    Get the data and detect aspects.
    """
    clf, clf_descr = classifiers[clf_name]
    verbs, X_train, y_train, categories = aspect_dataset(dfname)
    test_verbs, X_test, y_test = load_test_data(ifname)
    X_tfidf_train, X_tfidf_test, feature_names = extract_features(X_train, X_test, silent, print_top10)
    results = benchmark(clf, clf_descr, X_tfidf_train, X_tfidf_test, y_train, y_test, feature_names, categories, silent, print_top10)
    with open(ofname, 'w', encoding='utf-8') as ofile:
        for verb, asp in zip(test_verbs, results[-1]):
            ofile.write(','.join([verb, categories[asp]]) + '\n')

def aspect_training(pfname, silent=False, print_top10=False):
    """
    Get the dataset, do the benchmarks, plot them.    
    """
    verbs, X, y, categories = aspect_dataset() # this one comes from ngrams.py
    print("{} entities in dataset".format(len(X)))
    mean_results = cross_validate(X, y, categories, folds_count, silent, print_top10)
    indices = np.arange(len(mean_results))
    plot_all([[x[i] for x in mean_results] for i in range(4)], indices, pfname)

#------------------------------------------------------------------------------
# 'Ere we go ------------------------------------------------------------------
#------------------------------------------------------------------------------
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    opts, ifname = get_options()
    if opts.lfname is not None:
        sys.stdout = open(opts.ofname, 'w', encoding='utf-8')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        warnings.warn('deprecated', DeprecationWarning)
        warnings.warn('deprecated', UserWarning)
        if opts.benchmark:
            aspect_training(opts.pfname, silent=opts.silent, print_top10=opts.print_top10)
        else:
            aspect_detection(ifname, opts.ofname, opts.classifier_name, dfname, silent=opts.silent, print_top10=opts.print_top10)
            print("Done. See {}".format(opts.ofname))
