'''
Authors - Andrew Vo & Anant Natekar
'''
import nltk
import os
import re
import random  
import string
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors  import KNeighborsClassifier
from sklearn.svm  import SVC
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_svmlight_file

path = "/Users/nat2147/Anant/code/python/" #defining the path to the training data files

##################################################################
# Multinomial Naive Bayes
##################################################################

#f1 macro
clf = MultinomialNB()
feature_vectors, targets = load_svmlight_file(path + "trainingdatafile.txt")
scores = cross_val_score(clf, feature_vectors, targets, cv=5, scoring='f1_macro')
print("MultinomialNB(f1 macro) Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(),scores.std() * 2))

#precision macro
clf = MultinomialNB()
feature_vectors, targets = load_svmlight_file(path + "trainingdatafileTFIDF.txt")
scores = cross_val_score(clf, feature_vectors, targets, cv=5,scoring='precision_macro')
print("MultinomialNB(precision macro) Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#recall macro
clf = MultinomialNB()
feature_vectors, targets = load_svmlight_file(path + "trainingdatafileTFIDF.txt")
scores = cross_val_score(clf, feature_vectors, targets, cv=5,scoring='recall_macro')
print("MultinomialNB(recall macro) Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(),scores.std() * 2))

##################################################################
# Bernoulli Naive Bayes
##################################################################

#f1 macro
clf2 = BernoulliNB()
feature_vectors2, targets2 = load_svmlight_file(path + "trainingdatafileTFIDF.txt")
scores = cross_val_score(clf2, feature_vectors2, targets2, cv=5,scoring='f1_macro')
print("BernoulliNB(f1 macro) Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(),scores.std() * 2))

#precision macro
clf2 = BernoulliNB()
feature_vectors2, targets2 = load_svmlight_file(path + "trainingdatafileTFIDF.txt")
scores = cross_val_score(clf2, feature_vectors2, targets2, cv=5,scoring='precision_macro')
print("BernoulliNB(precision_macro) Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(),scores.std() * 2))

#recall macro
clf2 = BernoulliNB()
feature_vectors2, targets2 = load_svmlight_file(path + "trainingdatafileTFIDF.txt")
scores = cross_val_score(clf2, feature_vectors2, targets2, cv=5,scoring='recall_macro')
print("BernoulliNB(recall macro) Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(),scores.std() * 2))

##################################################################
# K Nearest Neighbor
##################################################################

#f1 macro
clf3 = KNeighborsClassifier()
feature_vectors3, targets3 = load_svmlight_file(path + "trainingdatafileTFIDF.txt")
scores = cross_val_score(clf3, feature_vectors3, targets3, cv=5,scoring='f1_macro')
print("KNeighborsClassifier(f1 macro) Accuracy: %0.2f (+/- %0.2f)" %(scores.mean(), scores.std() * 2))

#precision macro
clf3 = KNeighborsClassifier()
feature_vectors3, targets3 = load_svmlight_file(path + "trainingdatafileTFIDF.txt")
scores = cross_val_score(clf3, feature_vectors3, targets3, cv=5,scoring='precision_macro')
print("KNeighborsClassifier(precision macro) Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#recall macro
clf3 = KNeighborsClassifier()
feature_vectors3, targets3 = load_svmlight_file(path + "trainingdatafileTFIDF.txt")
scores = cross_val_score(clf3, feature_vectors3, targets3, cv=5,scoring='recall_macro')
print("KNeighborsClassifier(recall macro) Accuracy: %0.2f (+/- %0.2f)" %(scores.mean(), scores.std() * 2))

##################################################################
# SVC
##################################################################

#f1 macro
clf4 = SVC(gamma='auto')
feature_vectors4, targets4 = load_svmlight_file(path + "trainingdatafileTFIDF.txt")
scores = cross_val_score(clf4, feature_vectors4, targets4, cv=5,scoring='f1_macro')
print("SVC (f1 macro) Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() *2))

#precision macro
clf4 = SVC(gamma='auto')
feature_vectors4, targets4 = load_svmlight_file(path + "trainingdatafileTFIDF.txt")
scores = cross_val_score(clf4, feature_vectors4, targets4, cv=5,scoring='precision_macro')
print("SVC (precision macro) Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(),scores.std() * 2))

#recall macro
clf4 = SVC(gamma='auto')
feature_vectors4, targets4 = load_svmlight_file(path + "trainingdatafileTFIDF.txt")
scores = cross_val_score(clf4, feature_vectors4, targets4, cv=5,scoring='recall_macro')
print("SVC (recall macro) Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(),scores.std() * 2))