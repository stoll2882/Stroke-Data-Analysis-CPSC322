import myutils as myutils
import numpy as np
import copy
import math
import random

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets (sublists) based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before splitting

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)
    
    Note:
        Loosely based on sklearn's train_test_split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    
    X_data = copy.deepcopy(X)
    y_data = copy.deepcopy(y)
    
    if random_state is not None:
       # TODO: seed your random number generator
       # you can use the math module or use numpy for your generator
       # choose one and consistently use that generator throughout your code
       np.random.seed(random_state)
    
    if shuffle: 
        # TODO: shuffle the rows in X and y before splitting
        # be sure to maintain the parallel order of X and y!!
        # note: the unit test for train_test_split() does not test
        # your use of random_state or shuffle, but you should still 
        # implement this and check your work yourself
        randomize_in_place(X_data, y_data)
        
    num_instances = len(X_data)
    if isinstance(test_size, float):
        test_size = math.ceil(num_instances * test_size)  # ceil(8 * 0.33)
    split_index = num_instances - test_size  # 8 - 2 = 6

    return X[:split_index], X[split_index:], y[:split_index], y[split_index:]
    
def randomize_in_place(alist, parallel_list=None):
    for i in range(len(alist)):
        # generate a random index to swap the element at i with
        rand_index = random.randrange(0, len(alist))  # [0, len(alist)]
        alist[i], alist[rand_index] = alist[rand_index], alist[i]
        if parallel_list is not None:
            parallel_list[i], parallel_list[rand_index] = parallel_list[rand_index], parallel_list[i]

def kfold_cross_validation(X, n_splits=5):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.

    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold
        X_test_folds(list of list of int): The list of testing set indices for each fold

    Notes: 
        The first n_samples % n_splits folds have size n_samples // n_splits + 1, 
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    folds = []
    indexs = []
    X_train_folds = []
    X_test_folds = []
    for s in range(n_splits):
        folds.append([])
        indexs.append([])
    for i, val in enumerate(X):
        folds[i % n_splits].append(val)
        indexs[i % n_splits].append(i)
    for i, fold in enumerate(folds):
        training = []
        testing = []
        for j, fold in enumerate(folds):
            if j != i:
                for k, val in enumerate(fold):
                    training.append(indexs[j][k])
            else:
                X_test_folds.append(indexs[j])
        X_train_folds.append(training)

    return X_train_folds, X_test_folds  # TODO: fix this

def stratified_kfold_cross_validation(X, y, n_splits=5):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples). 
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X). 
            The shape of y is n_samples
        n_splits(int): Number of folds.

    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold.
        X_test_folds(list of list of int): The list of testing set indices for each fold.

    Notes: 
        Loosely based on sklearn's StratifiedKFold split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    class_labels = []
    class_folds = []
    folds = []

    indexs = []
    X_train_folds = []
    X_test_folds = []

    for label in y:
        if label not in class_labels:
            class_labels.append(label)
            class_folds.append([])
    for _ in range(n_splits):
        folds.append([])
        indexs.append([])
    for i, val in enumerate(X):
        for j, label in enumerate(class_labels):
            if y[i] == label:
                class_folds[j].append(val)

    ind = 0
    for i, fold in enumerate(class_folds):
        j = 0
        while (j < len(fold)):
            folds[ind % n_splits].append(fold[j])
            indexs[ind % n_splits].append(X.index(fold[j]))
            j += 1
            ind += 1

    X_train_folds = []
    X_test_folds = []

    for i, fold in enumerate(folds):
        training = []
        testing = []
        for j, fold in enumerate(folds):
            if j != i:
                for k, val in enumerate(fold):
                    training.append(indexs[j][k])
            else:
                X_test_folds.append(indexs[j])
        X_train_folds.append(training)

    return X_train_folds, X_test_folds  # TODO: fix this

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry 
            indicates the number of samples with true label being i-th class 
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix(): https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    matrix = []

    for _ in labels:
        new = []
        for _ in labels:
            new.append(0)
        matrix.append(new)

    for i, val in enumerate(y_true):
        label_index = labels.index(val)
        label_index2 = labels.index(y_pred[i])
        matrix[label_index][label_index2] += 1

    return matrix  # TODO: fix this