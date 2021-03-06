import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.learning_curve import learning_curve
from sklearn import preprocessing
from numpy import loadtxt
import scipy


### data preprocessing and matrix formation 



train_data1 = np.genfromtxt("train_data.txt")
dev_labels = np.genfromtxt("bc3_act_gold_standard.tsv")
dev_labels = dev_labels[:,1]
labels = np.genfromtxt("test_labels.txt")
labels = labels[:,1]
train_labels = labels[0:4000 :,]
test_labels = labels[4000:6000 :,] 
 

n_train_data1= preprocessing.normalize(train_data1)
n_train_data1 = scipy.delete(n_train_data1,227,1)
n_train_data1 = scipy.delete(n_train_data1,242,1)
n_train_data1 = scipy.delete(n_train_data1,312,1)
n_train_data1 = scipy.delete(n_train_data1,135,1)
                                                       #n_train_data1 has 6000 samples and 1408 features 
train_data = n_train_data1[0:4000 :,]                  #  training data with 4000 pts 
test_data = n_train_data1[4000:6000 :,]                # test_data has 2000 pts  



dev_data = np.genfromtxt("test_data.txt")
n_dev_data = preprocessing.normalize(dev_data)  
n_dev_data = scipy.delete(n_dev_data,227,1)
n_dev_data = scipy.delete(n_dev_data,242,1)  
n_dev_data = scipy.delete(n_dev_data,312,1)
dev_data = scipy.delete(n_dev_data,135,1)               #  development data has 2280 pts 




def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(0.725,1,5)):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  


    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """







    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")

    return plt


'''
title = "Learning Curves (Naive Bayes)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = cross_validation.ShuffleSplit(digits.data.shape[0], n_iter=100,
                                   test_size=0.2, random_state=0)

estimator = GaussianNB()
plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

'''



title = "Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
# SVC is more expensive so we do a lower number of CV iterations:

estimator = SVC(C =1,gamma=1)
plot_learning_curve(estimator, "SVM learning curves" ,train_data, train_labels, cv=3, n_jobs=1)

plt.show()