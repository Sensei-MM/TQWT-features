'''
前9训练集，后6测试集，GridSearchCV优化分类
'''

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,recall_score
import numpy as np
import pandas as pd
import math
import scipy.io as sio
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score,cross_val_predict,cross_validate
from sklearn.metrics import plot_roc_curve,confusion_matrix, classification_report,r2_score
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from pandas import DataFrame
from sklearn.metrics import precision_recall_fscore_support,accuracy_score,precision_score,recall_score,f1_score
import h5py
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

def band_accuracy(data):
    X = data[:, :-1]
    print(X.shape)
    y = data[:, -1]
    y = y.reshape(y.size, 1)
    # 3. 数据标准化
    X_scale = preprocessing.scale(X)

    data_scale = np.concatenate((X_scale, y), axis=1)

    train_data = data_scale[:3980, :]  # 0.5s
    np.random.shuffle(train_data)
    X_train = train_data[:, :-1]
    y_train = train_data[:, -1]

    test_data = data_scale[3980:]
    np.random.shuffle(test_data)
    X_test = test_data[:, :-1]
    y_test = test_data[:, -1]
    rf_clf = RandomForestClassifier()
    tuned_parameters = {'n_estimators': [5, 10, 50, 100],  # 树的数量
                        'criterion': ['gini', 'entropy'],  # 标准采用计算信息增益gini的方法
                        'max_features': ['auto', 'sqrt'],
                        'max_depth': [10, 20, 50]}
    scores = [ 'accuracy']
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()
        clf = GridSearchCV(rf_clf, tuned_parameters)
        clf.fit(X_train, y_train)
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']

        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))

        y_true, y_pred = y_test, clf.predict(X_test)
        acc = np.round(accuracy_score(y_true, y_pred),4)
    return acc

people_name = ['1_20131027',#1
               '2_20140404',
               '3_20140603',
               '4_20140621',
               '5_20140411',
               '6_20130712',
               '7_20131027',
               '8_20140511',
               '9_20140620',
               '10_20131130',
               '11_20140618',
               '12_20131127',
               '13_20140527',
               '14_20140601',
               '15_20130709']

if __name__ == '__main__':
    # 1 DE
    file_path = 'G:\Paper-EEG Emotion\SEED-tqwt2-8band-q1r3j7-2\Analysis-Trditional\DE0.5s-day1\\'
    band = 8
    svm_linear_acc = np.zeros((len(people_name), band))
    for sub in range(len(people_name)):
        file_name = file_path + people_name[sub]
        read_data = sio.loadmat(file_name)
        # read_data = h5py.File(file_name+'.mat',mode='r')
        temp_data = read_data.get('eeg_label')
        sub_data = np.array(temp_data)
        print(temp_data.shape)
        for b in range(band):
            print('processing:{},band:{},feature:{}'.format(people_name[sub],b,file_path[-14:-10]))
            band_data = sub_data[:, :, b]
            band_acc = band_accuracy(band_data)
            svm_linear_acc[sub,b] = band_acc
    dataframe = DataFrame(svm_linear_acc)
    with pd.ExcelWriter('G:\Paper-EEG Emotion\SEED-tqwt2-8band-q1r3j7-2\Analysis-Trditional\\rf-de0.5-grid.xlsx') as writer:
        dataframe.to_excel(writer, sheet_name='de-q1r3j7-8band-rf-day1')
