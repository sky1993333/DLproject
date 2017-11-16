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
print "the mean of traing data is:", trainingData.mean(axis=0)
print "the std of training data is:", trainingData.std(axis=0)

# normalize raw data
trainingData = preprocessing.scale(trainingData)
testingData = preprocessing.scale(testingData)
print "the mean of normalized traing data is:",trainingData.mean(axis=0)
print "the std of normalized traing data is:",trainingData.std(axis=0)
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
        #get the cross validation accuracy rate
        scores.append(clf.score(x[test],y[test]))
    np.savetxt("project1_20404460.csv", testLabel.astype(int),fmt='%10.5f',delimiter=",")
    print "the score of SVM model is", np.mean(scores)
    return np.mean(scores)


# logisticRegressionModel
def classifylr(x,y,z):
    clf = LogisticRegression(random_state = 12)
    scores = []
    # use 10-fold cross validation
    kf = KFold(len(y),n_folds = 10)
    for train,test in kf:
        clf.fit(x[train],y[train])
        testLabel = clf.predict(z)
        #get the cross validation accuracy rate
        scores.append(clf.score(x[test],y[test]))
    np.savetxt("project1_20404460.csv", testLabel.astype(int),fmt='%10.5f',delimiter=",")
    print "the score of Logistic Regression model is", np.mean(scores)
    return np.mean(scores)

# linear SVC
from sklearn.svm import LinearSVC

def lsvcClassifier(x,y,z):
    lsvc = LinearSVC(penalty='l2',  dual=False, tol=0.0001,
                     C=17, fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000)
        
    scores = []
    kf = KFold(len(y),n_folds = 10)
    for train,test in kf:
        lsvc.fit(x[train],y[train])
        testLabel = lsvc.predict(z)
        # print testLabel.astype(int)
        scores.append(lsvc.score(x[test],y[test]))
    np.savetxt("project1_20404460.csv", testLabel.astype(int),fmt='%10.5f',delimiter=",")
    print "the score of Linear SVC model is", np.mean(scores)
    return np.mean(scores)

# transform dataframe to numpy array data
# trainingData = trainingData.values
trainingLabel = trainingLabel.values
# type(trainingData)

# run classify models
svms = svmClassify(trainingData,trainingLabel,testingData)
lrs = classifylr(trainingData,trainingLabel,testingData)
lsvcs = lsvcClassifier(trainingData,trainingLabel,testingData)
dict = {"svms":svms,"lrs": lrs, "lsvcs":lsvcs}
# print dict['']
maxModel = max(dict)
if maxModel =='svms':
    svmClassify(trainingData,trainingLabel,testingData)
elif maxModel == 'lrs':
    classifylr(trainingData,trainingLabel,testingData)
else:
    lsvcClassifier(trainingData,trainingLabel,testingData)
