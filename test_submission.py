#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 03:47:12 2018

@author: junshuaizhang
"""

import helper
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import submission as submission
test_data='./test_data.txt'
modified_data='./modified_data.txt'
strategy_instance = submission.fool_classifier(test_data)
parameters={'gamma':'auto',"C":0.05,
                "degree":3, "kernel":"linear",
                "coef0":0}
########
#
# Testing Script.......
#
#
########
X_train_class0, X_test_class0,y_train_class0, y_test_class0 = \
train_test_split(strategy_instance.class0,\
                  [0]*len(strategy_instance.class0),\
                  test_size=0.5)

X = X_train_class0+ strategy_instance.class1
for i in range(len(X)):
#    X[i] = np.array(X[i])
    X[i] = " ".join(X[i])

    
Y= y_train_class0 + [1]*len(strategy_instance.class1)

#    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.00001)
X_train = X
y_train = Y

count_vect = CountVectorizer(max_df=0.3, min_df=0.02, binary = True)
                                #stop_words='english')
#vectorizer = TfidfVectorizer(max_df=0.5,
#                                         min_df=2,stop_words="english", max_features=10000)
#vectors_train = vectorizer.fit_transform(X_train)
vectors_train = count_vect.fit_transform(X_train)
#test data
#vectors_test = count_vect.transform(X_test)
#vectors_test = vectorizer.transform(X_test)
#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler(with_mean = False)
#vectors_train = scaler.fit_transform(vectors_train)
#vectors_test = scaler.fit_transform(vectors_test)


clf = strategy_instance.train_svm(parameters, vectors_train,y_train)
with open(modified_data,'r') as file:
    modified_test_data=[line.strip() for line in file]
    modified_test_vectors = count_vect.transform(modified_test_data)
    y_test2 = [1]*len(modified_test_data)
    print(clf.score(modified_test_vectors,y_test2))
with open(test_data,'r') as file:
    test_data=[line.strip() for line in file]
    test_vectors = count_vect.transform(test_data)
    y_test1 = [1]*len(test_data)
    print(clf.score(test_vectors,y_test1))
    
result = (clf.score(test_vectors,y_test1)-clf.score(modified_test_vectors,y_test2))*100
print('Success %-age = {}-%'.format(result))