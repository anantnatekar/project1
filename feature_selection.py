'''
Authors - Andrew Vo & Anant Natekar
'''
import string
#import pandas as pd
#import read_csv
from sklearn import svm,datasets    
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.feature_selection import f_classif

X = feature_vectors
y = targets
X_new1 = SelectKBest(chi2, k=100).fit_transform(X, y)
X_new2 = SelectKBest(mutual_info_classif, k=100).fit_transform(X, y)

# load data

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
