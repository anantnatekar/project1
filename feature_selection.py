'''
Authors - Andrew Vo & Anant Natekar
'''
import string
import os
import matplotlib.pyplot as plt
from sklearn import svm,datasets
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.feature_selection import f_classif
from sklearn.datasets import load_svmlight_file
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors  import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm  import SVC

#path = "/Users/nat2147/Anant/code/python/" #defining the path to the training data files
#feature_vectors, targets = load_svmlight_file(path + "trainingdatafile.txt")
iris = datasets.load_iris()
feature_vectors = iris['data']
targets = iris['target']

X = feature_vectors
y = targets
X_new1 = SelectKBest(chi2, k=2).fit_transform(X, y)
X_new2 = SelectKBest(mutual_info_classif, k=2).fit_transform(X, y)

##################################################################
# Different classifiers
##################################################################

clf = MultinomialNB()
#clf = BernoulliNB()
#clf = KNeighborsClassifier()
#clf = SVC()

#f1 macro for chi2
feature_vectors = X_new1
scores_chi2 = cross_val_score(clf, feature_vectors, targets, cv=5, scoring='f1_macro')
print("%s (f1 macro) Accuracy: %0.2f (+/- %0.2f)" % (clf,scores_chi2.mean(),scores_chi2.std() * 2))

#f1 macro for mutual_info_classif
feature_vectors = X_new2
scores_mic = cross_val_score(clf, feature_vectors, targets, cv=5, scoring='f1_macro')
print("%s (f1 macro) Accuracy: %0.2f (+/- %0.2f)" % (clf,scores_mic.mean(),scores_mic.std() * 2))

plt.title(clf, fontsize=15)
plt.plot(scores_chi2, color='b', linestyle='-', linewidth=2, label = "Chi-squared distribution")
plt.plot(scores_mic, color='g', linestyle='-', linewidth=2, label = "Mutual info classifier")
plt.xlabel("Number of Clusters",fontsize=13)
plt.ylabel("F1 Scores",fontsize=13)
plt.legend(loc = 1)
#plt.show(block=False)
plt.show()