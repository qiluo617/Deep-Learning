# Qi Luo
# A02274095

import numpy as np
from sklearn.neighbors import KDTree
from scipy.stats import mode
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

DATA = np.genfromtxt(r'C:\Users\Qi\Desktop\Deep Learning\assignment2\assignment1\data_seed.dat')


def prob2_2(data):
    K = [1, 5, 10, 15]
    # 5-Folds cross-validator
    # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html#sklearn.neighbors.KDTree
    KFold_err = []

    for k in K:
        index = np.split(np.arange(len(data)), 5)
        count = 0
        total = 0
        for i in range(0, len(index)):
            test_index = index[i]
            train_index = np.concatenate(np.delete(index, i, axis=0), axis=None)
            tree = KDTree(data[train_index])
            for d in data[test_index]:
                dist, ind = tree.query(d.reshape(1, 8), k=k)
                vote = data[ind.reshape(-1, ), 7]
                # https://stackoverflow.com/questions/6252280/find-the-most-frequent-number-in-a-numpy-vector
                a = mode(vote)[0][0]
                if d[7] != a:
                    count = count + 1
            total = total + len(test_index)
        KFold_err.append(float(count) / total)

    print(KFold_err)

    # Note: LeaveOneOut() is equivalent to KFold(n_splits=n) and LeavePOut(p=1) where n is the number of samples.
    loo_err = []

    for k in K:
        index = np.arange(len(data))
        count = 0
        total = 0
        for i in range(0, len(index)):
            test_index = i
            train_index = np.delete(index, i, axis=0)
            tree = KDTree(data[train_index])
            dist, ind = tree.query(data[test_index].reshape(1, 8), k=k)
            vote = data[ind.reshape(-1, ), 7]
            # https://stackoverflow.com/questions/6252280/find-the-most-frequent-number-in-a-numpy-vector
            a = mode(vote)[0][0]
            # print(vote)
            # print(a)
            if data[test_index][7] != a:
                count = count + 1
            total += 1
        loo_err.append(float(count) / total)

    print(loo_err)

    plt.plot(KFold_err, label='5-fold cross validation')
    plt.plot(loo_err, label='leave-one-out validation ')
    plt.ylabel('Test Error')
    plt.xlabel('k-nn classifier')
    plt.xticks(np.arange(4), K)
    plt.legend()
    plt.show()


def prob2_3(data):
    # LogisticRegression and SVM
    train_err_lr = 0
    test_err_lr = 0
    train_err_svm = 0
    test_err_svm = 0
    lr_tr = []
    svm_tr = []

    lr_te = []
    svm_te = []

    index = np.split(np.arange(len(data)), 5)
    for i in range(0, len(index)):
        test_index = index[i]
        train_index = np.concatenate(np.delete(index, i, axis=0), axis=None)

        model = LogisticRegression()
        model.fit(data[train_index][:, 0:6], data[train_index][:, [7]])
        # predictions = model.predict(data[test_index][:, 0:6])
        # print(classification_report(data[test_index][:, [7]], predictions))
        test_score = 1.0 - model.score(data[test_index][:, 0:6], data[test_index][:, [7]])
        test_err_lr += test_score
        lr_te.append(test_score)
        train_score = 1.0 - model.score(data[train_index][:, 0:6], data[train_index][:, [7]])
        train_err_lr += train_score
        lr_tr.append(train_score)

        clf = SVC(gamma='auto')
        clf.fit(data[train_index][:, 0:6], data[train_index][:, [7]])
        train_err_svm += 1.0 - clf.score(data[train_index][:, 0:6], data[train_index][:, [7]])
        svm_tr.append(1.0 - clf.score(data[train_index][:, 0:6], data[train_index][:, [7]]))
        test_err_svm += 1.0 - clf.score(data[test_index][:, 0:6], data[test_index][:, [7]])
        svm_te.append(1.0 - clf.score(data[test_index][:, 0:6], data[test_index][:, [7]]))
    print('LogisticRegression')
    print('train:', float(train_err_lr)/5, lr_tr)
    print('test:', float(test_err_lr)/5, lr_te)
    print('SVM')
    print('train:', float(train_err_svm)/5, svm_tr)
    print('test:', float(test_err_svm)/5, svm_te)

    plt.figure(0)
    plt.plot(lr_tr, label='lr train')
    plt.plot(lr_te, label='lr test')
    plt.ylim([0, 1])
    plt.legend()
    plt.xticks(np.arange(5), [1, 2, 3, 4, 5])

    plt.figure(1)
    plt.plot(svm_tr, label='svm train')
    plt.plot(svm_te, label='svm test')
    plt.ylim([0, 1])
    plt.legend()
    plt.xticks(np.arange(5), [1, 2, 3, 4, 5])
    plt.show()


prob2_2(DATA)
prob2_3(DATA)
