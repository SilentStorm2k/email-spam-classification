# -*- coding: utf-8 -*-
'''
# @author : SilentStorm2k
# Modified version of code from Kevin S. Xu
'''
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # Or another interactive backend
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SequentialFeatureSelector
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import random
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, AdaBoostClassifier, StackingClassifier
def aucCV(features,labels):
    #model = GaussianNB()
    #model = KNeighborsClassifier(n_neighbors=9,
    #                           p=2,
    #                           metric='minkowski')
    model = SVC()
    scores = cross_val_score(model,features,labels,cv=10,scoring='roc_auc')
    return scores

def predictTest(trainFeatures,trainLabels,testFeatures):
    sc = StandardScaler()
    sc.fit(trainFeatures)
    trainFeatures = sc.transform(trainFeatures)
    testFeatures = sc.transform(testFeatures)
    model1 = GaussianNB()
    model2 = SVC(gamma=0.001, C=1000, probability=True, class_weight='balanced', random_state=1)
    model3 = LogisticRegression(C=0.01,solver='lbfgs', random_state=1, class_weight='balanced', n_jobs=-1)
    model4 = RandomForestClassifier(n_estimators=200, max_depth=100, random_state=1, class_weight='balanced', n_jobs=-1)
    model5 = KNeighborsClassifier(n_neighbors=15, p=1, metric='minkowski', n_jobs=-1)
    model6 = DecisionTreeClassifier(max_depth=None, random_state=1)
    model7 = AdaBoostClassifier(random_state=1)
    lr = LogisticRegression(random_state=1)
    #model = StackingClassifier(estimators=[('svm', model2), ('lr', model3), ('rf',model4), ('knn', model5), ('dtc', model6), ('ada', model7)
    #                                       ], final_estimator=model3, cv=10, n_jobs=-1, stack_method='predict_proba')
    model = VotingClassifier(estimators=[('svm', model2), ('lr', model3), ('rf', model4), ('knn', model5), ('dtc', model6), ('ada', model7)
                                         ], voting='soft', n_jobs=-1)
    #params = {'svm__C'  : [0.1, 1, 10, 1000], 'svm__gamma':[1, 0.1, 0.001],
    #          'lr__C' : [0.01], 'rf__n_estimators' : [200,  600, 1000],
    #          'rf__max_depth' : [20, 60, 100, None],
    #         'knn__n_neighbors': [5, 7 ,9 ,11 ,13 ,15],
    #          'dtc__max_depth' :[20, 60, 100, None]}
    # {'svm__gamma': 1, 'svm__C': 1, 'rf__n_estimators': 1400, 'rf__max_depth': 10,'lr__C': 0.01, 'knn__n_neighbors': 10, 'dtc__max_depth': 90}
    # {'svm__gamma': 1, 'svm__C': 0.1, 'rf__n_estimators': 1000, 'rf__max_depth': 100, 'lr__C': 0.01, 'knn__n_neighbors': 7, 'dtc__max_depth': 20}
    # {'svm__gamma': 0.001, 'svm__C': 1000, 'rf__n_estimators': 200, 'rf__max_depth': 100, 'lr__C': 0.01, 'knn__n_neighbors': 15, 'dtc__max_depth': None}
    #model = RandomizedSearchCV(clf, param_distributions=params, n_iter=100, n_jobs=-1)
    #sfs = SequentialFeatureSelector(model, direction='backward', n_features_to_select=23, cv=10, n_jobs=-1)
    #sfs = SFS(model5, k_features=21, forward=False, floating=True, verbose=1, cv=10, n_jobs=-1)
    #sfs.fit(trainFeatures, trainLabels)
    #print(sfs.get_params())
    #print(sfs.get_metric_dict())
    #print(sfs.get_feature_names_out())
    #print(sfs.get_support())
    #trainFeatures = sfs.transform(trainFeatures)
    #testFeatures = sfs.transform(testFeatures)
    selectedFeatures = [ True, False , True , True,  True, False,  True, False,  True , True,  True, False,
                         False,  True,  True, False,  True,  True,  True , True,  True,  True,  True,  True,
                         True, False,  True,  True , True,  True]
    trainFeatures = trainFeatures[:, selectedFeatures]
    testFeatures = testFeatures[:, selectedFeatures]
    model.fit(trainFeatures,trainLabels)
    # see best parameters
    #print(model.best_params_)
    
    # Use predict_proba() rather than predict() to use probabilities rather
    # than estimated class labels as outputs
    testOutputs = model.predict_proba(testFeatures)[:,1]
    
    return testOutputs
    
# Run this code only if being used as a script, not being imported
if __name__ == "__main__":
    data = np.loadtxt('data/spamTrain1.csv', delimiter=',')
    # Randomly shuffle rows of data set then separate labels (last column)
    shuffleIndex = np.arange(np.shape(data)[0])
    np.random.shuffle(shuffleIndex)
    data = data[shuffleIndex,:]
    features = data[:,:-1]
    labels = data[:,-1]
    
    # Evaluating classifier accuracy using 10-fold cross-validation
    print("10-fold cross-validation mean AUC: ",
          np.mean(aucCV(features,labels)))
    
    # Arbitrarily choose all odd samples as train set and all even as test set
    # then compute test set AUC for model trained only on fixed train set
    trainFeatures = features[0::2,:]
    trainLabels = labels[0::2]
    testFeatures = features[1::2,:]
    testLabels = labels[1::2]
    testOutputs = predictTest(trainFeatures,trainLabels,testFeatures)
    print("Test set AUC: ", roc_auc_score(testLabels,testOutputs))
    
    # Examine outputs compared to labels
    sortIndex = np.argsort(testLabels)
    nTestExamples = testLabels.size
    plt.subplot(2,1,1)
    plt.plot(np.arange(nTestExamples),testLabels[sortIndex],'b.')
    plt.xlabel('Sorted example number')
    plt.ylabel('Target')
    plt.subplot(2,1,2)
    plt.plot(np.arange(nTestExamples),testOutputs[sortIndex],'r.')
    plt.xlabel('Sorted example number')
    plt.ylabel('Output (predicted target)')
    plt.show()