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
#clf = MultinomialNB()
#clf = BernoulliNB()
clf = KNeighborsClassifier()
#clf = SVC()

scores = cross_val_score(clf, feature_vectors, targets, cv=5, scoring='f1_macro')
print(scores)
X_new1 = SelectKBest(chi2, k=3).fit_transform(X, y)
X_new2 = SelectKBest(mutual_info_classif, k=3).fit_transform(X, y)

print (X_new1.shape)
print (X_new2.shape)

plt.title("Feature selection", fontsize=15)
plt.plot(scores, color='b', linestyle='-', linewidth=2)
plt.xlabel("Number of Clusters",fontsize=13)
plt.ylabel("F1 Scores",fontsize=13)
#plt.legend(loc = 1)
#plt.show(block=False)
plt.show()

# load data
'''
filename = '.csv'
names = [names]
dataframe = read_csv(filename, names=names)
array = dataframe.values

X = array[:,0:8]
Y = array[:,8]

# feature extraction
test = SelectKBest(score_func=f_classif, k=4)
fit = test.fit(X, Y)

# summarize scores
set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)

# summarize selected features
print(features[0:5,:])
'''