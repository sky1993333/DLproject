import csv
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import svm
from os.path import getsize
from sklearn.cross_validation import KFold
from sklearn import datasets
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

# load data  in .csv format
trainingData = pd.read_csv('traindata.csv', sep=',',header=None)
trainingLabel = pd.read_csv('trainlabel.csv', sep=',',header=None)
testingData = pd.read_csv('testdata.csv', sep=',',header=None)

# get the mean and std of the original data
print trainingData.mean(axis=0)
print trainingData.std(axis=0)

# normalize raw data
trainingData = preprocessing.scale(trainingData)
testingData = preprocessing.scale(testingData)
trainingData.mean(axis=0)
trainingData.std(axis=0)
# feature selection
# pca = PCA(n_components=40)
# pca.fit(trainingData)
# pca.fit(testingData)
# trainingData = pca.transform(trainingData)
# testingData = pca.transform(testingData)

# SVMModel
def svmClassify(x,y,z):
    clf = svm.SVC(kernel='rbf')
    scores = []
    # use 10-fold cross validation
    kf = KFold(len(y),n_folds = 10)
    for train,test in kf:
        clf.fit(x[train],y[train])
        testLabel = clf.predict(z)
        print testLabel.astype(int)
        #get the cross validation accuracy rate
        scores.append(clf.score(x[test],y[test]))
    np.savetxt("project1_20404460.csv", testLabel.astype(int),fmt='%10.5f',delimiter=",")
    print np.mean(scores)


# logisticRegressionModel
def classify(x,y,z):
    clf = LogisticRegression(random_state = 12)
    scores = []
    # use 10-fold cross validation
    kf = KFold(len(y),n_folds = 10)
    for train,test in kf:
        clf.fit(x[train],y[train])
        testLabel = clf.predict(z)
        print testLabel.astype(int)
        #get the cross validation accuracy rate
        scores.append(clf.score(x[test],y[test]))
    np.savetxt("project1_20404460.csv", testLabel.astype(int),fmt='%10.5f',delimiter=",")
    print np.mean(scores)

# transform dataframe to numpy array data
# trainingData = trainingData.values
trainingLabel = trainingLabel.values
# type(trainingData)

# run classify models
svmClassify(trainingData,trainingLabel,testingData)
classify(trainingData,trainingLabel,testingData)
