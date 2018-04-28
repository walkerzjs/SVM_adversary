#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 03:27:03 2018

@author: junshuaizhang
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
import helper
#import numpy as np
#def fool_classifier(test_data): ## Please do not change the function defination...
    ## Read the test data file, i.e., 'test_data.txt' from Present Working Directory...
    
    
    ## You are supposed to use pre-defined class: 'strategy()' in the file `helper.py` for model training (if any),
    #  and modifications limit checking
strategy_instance=helper.strategy() 
parameters={'gamma':'auto',"C":0.1,
            "degree":10, "kernel":"linear",
            "coef0":-100}


##..................................#
#
#
#
## Your implementation goes here....#
#
#
#
##..................................#


## Write out the modified file, i.e., 'modified_data.txt' in Present Working Directory...


## You can check that the modified text is within the modification limits.
modified_data='./modified_data.txt'
#assert strategy_instance.check_data(test_data, modified_data)
#return strategy_instance ## NOTE: You are required to return the instance of this class.

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_train_class0, X_test_class0,y_train_class0, y_test_class0 = \
train_test_split(strategy_instance.class0,\
                  [0]*len(strategy_instance.class0),\
                  test_size=0)

X = X_train_class0+ strategy_instance.class1
for i in range(len(X)):
#    X[i] = np.array(X[i])
    X[i] = " ".join(X[i])

    
Y= y_train_class0 + [1]*len(strategy_instance.class1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0)


vectorizer = CountVectorizer(max_df=1.0, min_df=0.01, binary = False)
                               
#vectorizer = TfidfVectorizer(max_df=0.5,
#                                         min_df=2,max_features=10000,stop_words="english",ngram_range=(2, 2), binary=True, norm="l1")
vectors_train = vectorizer.fit_transform(X_train)
#vectors_train = vectorizer.fit_transform(X_train)
#test data
#vectors_test = vectorizer.transform(X_test)
vectors_test = vectorizer.transform(X_test)
#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler(with_mean = False)
#vectors_train = scaler.fit_transform(vectors_train)
#vectors_test = scaler.fit_transform(vectors_test)


clf = strategy_instance.train_svm(parameters, vectors_train,y_train)
#clf = LinearSVC(random_state=0, C = 100).fit(vectors_train,y_train)
print("train accuracy")
print(clf.score(vectors_train,y_train))
#print("train_test accuracy")
#print(clf.score(vectors_test,y_test))

#y_test_p = clf.predict(vectors_test)
#print("train_test confusion matrix")
#print(confusion_matrix(y_test, y_test_p))
#cm = confusion_matrix(y_test, y_test_p)
#rate_true_positive = cm[1][1]/(cm[1][1]+cm[0][1])
#print("rate_true_positive: {}".format(rate_true_positive))
#print("rate_1 vs all: {}".format(1-len(y_train_class0)/len(Y)))
#get test data
test_data='./test_data.txt'
test_data_splitted = []
with open(test_data,'r') as file:
    test_data_splitted=[line.strip().split(' ') for line in file]

test_data_text=[]
for i in range(len(test_data_splitted)):
#    X[i] = np.array(X[i])
    test_data_text.append(" ".join(test_data_splitted[i]))

y_test2 = [1]*len(test_data_text)
#vectors_test2 = vectorizer.transform(test_data_text)
vectors_test2 = vectorizer.transform(test_data_text)
print("test_data_accuracy")
print(clf.score(vectors_test2,y_test2))


##start to prepare modification
features = vectorizer.get_feature_names()
#features = vectorizer.get_feature_names()

coef = clf.coef_.toarray()
pairs = list(zip(features,coef.tolist()[0]))
pairs.sort(key = lambda x:x[1])

pos_features = []
neg_features = []
for pair in pairs:
    if pair[1]>0:
        pos_features.append(pair)
    else:
        neg_features.append(pair)
        
pos_features.sort(key = lambda x:x[1], reverse=True)
neg_features.sort(key = lambda x:x[1])
pairs_dict = dict(pairs)

for doc in test_data_splitted:
    for i in range(len(doc)):
        word = doc[i]
        if word in pairs_dict:
            
            doc[i] = (word,pairs_dict[word])
        else:
            doc[i] = (word,0)
   
# Generate modified data
num_to_delete = 10
neg_features_dict = dict(neg_features)
test_data_modified_splitted = []
for doc in test_data_splitted:
#    order = sorted(range(len(doc)), key=lambda k: doc[k][1],
#                   reverse=True)
#    idx=0
#    for i in order[:10]:
#        doc[i] = neg_features[idx]
#        idx+=1
    doc.sort(key = lambda x:x[1], reverse=True)
#    print(doc)
    doc_text = [pair[0] for pair in doc]
    
    word_set = set()
    last_count = 0
#    idx_now = 0
    origin_set = doc_text.copy()
    replaced_amount = 0
    deleted_amount = 0
    # Delete most positive words
    for i in range(len(pos_features)):
        replaced_amount = len((set(origin_set)-set(doc_text)) | (set(doc_text)-set(origin_set)))
#        print(replaced_amount)
        if replaced_amount==10:
#            print(i)
            break
        pos_word = pos_features[i][0]
        if doc_text.count(pos_word)>0:
            #The next line referred to https://stackoverflow.com/questions/1157106/remove-all-occurrences-of-a-value-from-a-list 
            #This function removes all occurances of "pos_word" in doc_text
            doc_text = list(filter(lambda x: x != pos_word, doc_text))
            deleted_amount+=1
#            print("Deleted word: {}".format(pos_word))
    # Add most negative words
    for i in range(len(neg_features)):
        replaced_amount = len((set(origin_set)-set(doc_text)) | (set(doc_text)-set(origin_set)))
#        print(replaced_amount)
        if replaced_amount==20:
#            print(i)
            break
        neg_word = neg_features[i][0]
        if doc_text.count(neg_word)>0:
            continue
        else:
            for i in range(3):
                doc_text.append(neg_word)
#                print("Added word: {}".format(neg_word))
        
    replaced_amount = len((set(origin_set)-set(doc_text)) | (set(doc_text)-set(origin_set)))
#        print(replaced_amount)
    if replaced_amount!=20:
        print("Something wrong!!! The replaced amount is : {}".format(replaced_amount))
        break
#            print(i)         
            
            
#    for i in range(len(neg_features)):
##    i=0
##    while i < len(doc_text):
#        word_replace = neg_features[i][0]
#        if doc_text.count(word_replace)>0:
#            continue
#        
#        word= doc_text[last_count]
#        #The next line referred to https://stackoverflow.com/questions/1157106/remove-all-occurrences-of-a-value-from-a-list 
#        if word not in neg_features_dict:
#            doc_text = list(filter(lambda x: x != word, doc_text))
#        doc_text.append(neg_features[i][0])
#        last_count+=1
#        #word_set.add(word)
#        replaced_amount = len((set(origin_set)-set(doc_text)) | (set(doc_text)-set(origin_set)))
##        print(replaced_amount)
#        if replaced_amount==20:
##            print(i)
#            break
#        if replaced_amount> last_count:
            

#        word_set.add(neg_features[len(word_set)][0])
#        idx_now+=1
        #i+=1
   
    #print("aaa: "+str(replaced_amount))
    test_data_modified_splitted.append(doc_text)
    
#    doc_text = [pair[0] for pair in doc]
#    test_data.append(doc_text)
    

modified_test_data=[]
for i in range(len(test_data_modified_splitted)):
#    X[i] = np.array(X[i])
    modified_test_data.append(" ".join(test_data_modified_splitted[i]))
    
vectors_test_modified = vectorizer.transform(modified_test_data)

file = open(modified_data,"w")
for doc in modified_test_data:
    file.write(doc+" \n")


with open(modified_data,'r') as file:
    modified_test_data2=[line.strip() for line in file]

#order = sorted(range(len(doc)), key=lambda k: doc[k][1])
#for i in order:
#    print(doc[i])
print("modified_data_accuracy")
print(clf.score(vectors_test_modified,y_test2))
print(strategy_instance.check_data("./test_data.txt","./modified_data.txt"))

#strategy_instance2=helper.strategy() 
#parameters2={'gamma':'auto',"C":10,
#            "degree":3, "kernel":"rbf",
#            "coef0":0}
#
#clf = strategy_instance2.train_svm(parameters2, vectors_train,y_train)
#
#print(clf.score(vectors_train,y_train))
#print(clf.score(vectors_test2,y_test2))
#print(clf.score(vectors_test_modified,y_test2))