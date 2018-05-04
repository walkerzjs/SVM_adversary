#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 03:27:03 2018

@author: junshuaizhang, monaithang
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
import helper
from collections import defaultdict
#import numpy as np
#def fool_classifier(test_data): ## Please do not change the function defination...
    ## Read the test data file, i.e., 'test_data.txt' from Present Working Directory...


    ## You are supposed to use pre-defined class: 'strategy()' in the file `helper.py` for model training (if any),
    #  and modifications limit checking
strategy_instance=helper.strategy()
parameters={'gamma':'auto',"C":0.1,
            "degree":3, "kernel":"linear",
            "coef0":1}


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


vectorizer = CountVectorizer(max_df=1.0, min_df=1, binary = True, ngram_range=(1,1), max_features=5720, lowercase=False)

#vectorizer = TfidfVectorizer(max_df=1.0,
#                                         min_df=1,max_features=5720,ngram_range=(3, 10), binary=False)
vectors_train = vectorizer.fit_transform(X_train)
#vectors_train = vectorizer.fit_transform(X_train)
#test data
#vectors_test = vectorizer.transform(X_test)
#vectors_test = vectorizer.transform(X_test)
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
test_data_splitted_backup = test_data_splitted.copy()
#print(test_data_splitted_backup[0])
## compute the difference of test data and training data
#similarities_all = []
#for test_doc in test_data_splitted:
#    words_test = set(test_doc)
#    similarities_one = []
#    for train_0_doc in strategy_instance.class0:
#        words_train = set(train_0_doc)
#        difference = len((words_test-words_train) | (words_train-words_test))
#        similarities_one.append((train_0_doc, difference))
#        similarities_one.sort(key = lambda x:x[1])
#    similarities_all.append(similarities_one)
    

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

test_data_splitted_new = []
for i in range(len(test_data_splitted)):
    doc = test_data_splitted[i]
    new_doc = []
    for i in range(len(doc)):
        word = doc[i]
        if word in pairs_dict:

            new_doc.append((word,pairs_dict[word]))
        else:
            new_doc.append((word,0))
    test_data_splitted_new.append(new_doc)


test_data_splitted_modified = []
for i in range(len(test_data_splitted)):
    doc = test_data_splitted[i]
    new_doc = doc.copy()
    add_candidates = []
    delete_candidates=[]
    for pos in pos_features:
        if doc.count(pos[0])>0:
            delete_candidates.append(pos)
    for neg in neg_features:
        if doc.count(neg[0])>0:
            continue
        add_candidates.append(neg)
        
    idx_pos = 0
    idx_neg = 0
    origin_set = set(doc)
    doc_modified = list(set(doc))
    diff =0
    print("begin modifying: ")
    added_num = 0
    deleted_num = 0
    while(diff!=20):
        
        neg = add_candidates[idx_neg]
        pos = delete_candidates[idx_pos]
        #this part can control the amount of added words
        if abs(neg[1])> abs(pos[1]) and added_num<10:
            doc_modified.append(neg[0])
            print("add: {}".format(neg[0]))
            idx_neg+=1
            added_num+=1
        else:
            doc_modified.remove(pos[0])
            print("delete: {}".format(pos[0]))
            idx_pos+=1
            deleted_num+=1
        diff =len((set(origin_set)-set(doc_modified)) | (set(doc_modified)-set(origin_set))) 
    
    doc_modified_separated = []
    for word in doc_modified:
        doc_modified_separated.append(word)
        doc_modified_separated.append(".")
    
    test_data_splitted_modified.append(doc_modified_separated)
        
            
modified_test_data=[]
for i in range(len(test_data_splitted_modified)):
#    X[i] = np.array(X[i])
    modified_test_data.append(" ".join(test_data_splitted_modified[i]))

vectors_test_modified = vectorizer.transform(modified_test_data)

file = open(modified_data,"w")
for doc in modified_test_data:
    file.write(doc+" \n")   
file.close()
#with open(modified_data,'r') as file:
#    modified_test_data2=[line.strip() for line in file]

print(strategy_instance.check_data("./test_data.txt","./modified_data.txt"))
  
#replaced_amount = len((set(origin_set)-set(doc_text)) | (set(doc_text)-set(origin_set))) 
    
    
    
train_0 = strategy_instance.class0.copy()
#for doc in train_0:
#    for i in range(len(doc)):
#        word = doc[i]
#        if word in pairs_dict:
#
#            doc[i] = (word,pairs_dict[word])
#        else:
#            doc[i] = (word,0)

train_1 = strategy_instance.class1.copy()
#for doc in train_1:
#    for i in range(len(doc)):
#        word = doc[i]
#        if word in pairs_dict:
#
#            doc[i] = (word,pairs_dict[word])
#        else:
#            doc[i] = (word,0)


train_0_list = []
for doc in train_0:
    train_0_list+=doc
train_0_set = set(train_0_list)

train_1_list = []
for doc in train_1:
    train_1_list+=doc
train_1_set = set(train_1_list)

#diff_0_1 = train_0_set-train_1_set
#
#diff_01_count = []
#for word in diff_0_1:
#    word_count = train_0_list.count(word)
#    if word in pairs_dict:
#        coef = pairs_dict[word]
#    else:
#        coef = 0
#    diff_01_count.append((word,word_count,coef))
#
#diff_01_count.sort(key = lambda x:x[1], reverse = True)
#
#diff_1_0 = train_1_set-train_0_set
#diff_10_count = []
#for word in diff_1_0:
#    word_count = train_1_list.count(word)
#    if word in pairs_dict:
#        coef = pairs_dict[word]
#    else:
#        coef = 0
#    diff_10_count.append((word,word_count,coef))
#diff_10_count.sort(key = lambda x:x[1], reverse = True)
#
#ratios_train_0 = []
#ratios_dict_0 = {}
#for word in train_0_set:
#    word_count_0 = train_0_list.count(word)
#    word_count_1 = train_1_list.count(word)
#    ratio_word_0 = word_count_0/len(train_0_list)
#    ratio_word_1 = (word_count_1/len(train_1_list))
#    if ratio_word_1==0:
##        ratio=1000000000
#        continue
#    else:
#        ratio = ratio_word_0/ratio_word_1
##    ratio = word_count_0/(word_count_0 + word_count_1)
#    if word in pairs_dict:
#        coef = pairs_dict[word]
#    else:
#        coef = 0
#    ratios_train_0.append((word,word_count_0,word_count_1,ratio,coef))
#    ratios_dict_0[word] = (word_count_0,word_count_1,ratio,coef)
#ratios_train_0.sort(key = lambda x:x[3], reverse = True)
#
#ratios_train_1 = []
#ratios_dict_1 = {}
#for word in train_1_set:
#    word_count_0 = train_0_list.count(word)
#    word_count_1 = train_1_list.count(word)
#    ratio_word_0 = word_count_0/len(train_0_list)
#    ratio_word_1 = (word_count_1/len(train_1_list))
#    if ratio_word_0==0:
##        ratio=1000000000
#        continue
#    else:
#        ratio = ratio_word_1/ratio_word_0
##    ratio = word_count_0/(word_count_0 + word_count_1)
#    if word in pairs_dict:
#        coef = pairs_dict[word]
#    else:
#        coef = 0
#    ratios_train_1.append((word,word_count_0,word_count_1,ratio,coef))
#    ratios_dict_1[word] = (word_count_0,word_count_1,ratio,coef)
#ratios_train_1.sort(key = lambda x:x[3], reverse = True)


def words_to_str(words_list):
    output = " ".join(words_list)
    return output

#output = words_to_str(train_0[0])
#print(output)

train_0_sentences = []
for doc in train_0:
    start=0
    sentences = []
    for i in range(len(doc)):
        word = doc[i]
        if word == ".":
            sentence = doc[start:i+1]
            sent_str = words_to_str(sentence)
            sent_str_vector = vectorizer.transform([sent_str])
            distance = clf.decision_function(sent_str_vector)[0]
            sentences.append([sentence, distance])
            start = i+1
    train_0_sentences.append(sentences)

train_1_sentences = []
for doc in train_1:
    start=0
    sentences = []
    for i in range(len(doc)):
        word = doc[i]
        if word == ".":
            sentence = doc[start:i+1]
            sent_str = words_to_str(sentence)
            sent_str_vector = vectorizer.transform([sent_str])
            distance = clf.decision_function(sent_str_vector)[0]
            sentences.append([sentence, distance])
            start = i+1
    train_1_sentences.append(sentences)

test_sentences = []
for doc in test_data_splitted:
    start=0
    sentences = []
    for i in range(len(doc)):
        word = doc[i]
        if word == ".":
            sentence = doc[start:i+1]
            sent_str = words_to_str(sentence)
            sent_str_vector = vectorizer.transform([sent_str])
            distance = clf.decision_function(sent_str_vector)[0]
            sentences.append([sentence, distance])
            start = i+1
    test_sentences.append(sentences)

#test_sentences = []
#for doc in test_data_splitted:
#    start=0
#    sentences = []
#    for i in range(len(doc)):
#        word = doc[i]
#        if word == ".":
#            sentences.append(doc[start:i+1])
#            start = i+1
#    test_sentences.append(sentences)


distances=[]
idx = 10
test_case = train_1_sentences[idx][3]

##print(test_case)
#for doc in train_0_sentences:
#    for sent in doc:
#        difference = len(set(sent)-set(train_1[idx]))
#        distances.append((sent,difference))
#distances.sort(key = lambda x:x[1])

all_refers=[]
for i in range(len(test_sentences)):
    
    doc = test_sentences[i]
    doc_text = []
    for sent in doc:
        doc_text+=sent[0]
    refer_list = {}
#    for j in range(len(doc)):
#        sent = doc[j]
#        refer_list[j]=[]
    refer_list[i]=[]
    for ii in range(len(train_0_sentences)):
        doc_0 = train_0_sentences[ii]
        for jj in range(len(doc_0)):
            sent_0 = doc_0[jj]
            
            difference = len(set(sent_0[0])-set(doc_text))
            if sent_0[1] < 0 and difference <= int(22/(len(doc))) and len(sent_0[0])>6:
                sent_0_new = sent_0+[difference]
                refer_list[i].append(sent_0_new)
    refer_list[i].sort(key=lambda x:x[1])
#    refer_list[i].sort(key=lambda x:x[2])
    all_refers.append(refer_list)


def compute_diff(text,origin_text):
    diff = len((set(origin_text)-set(text)) | (set(text)-set(origin_text)))
    return diff


modified_test = []
for i in range(len(test_sentences)):
    test_origin_splitted = test_data_splitted[i]
    options = all_refers[i][i].copy()
    origin_doc = test_sentences[i]
    origin_doc.sort(key=lambda x:x[1], reverse=True)
    added_sentences = []
    words_redundant = []
    for sent in origin_doc:
        option = options[0][0]
        if len(options)>1:
            options.pop(0)
        added_sentences.append(option)
#        extra_words = set(sent[0])- set(option)
#        words_redundant += list(extra_words)
    
    added_sentences_combined=[]
    for s in added_sentences:
        added_sentences_combined+=s
    words_redundant = list(set(test_origin_splitted)-set(added_sentences_combined))
    diff_add = compute_diff(added_sentences_combined,test_origin_splitted)
    diff_redn = compute_diff(words_redundant,test_origin_splitted)
    diff_whole = compute_diff(added_sentences_combined+words_redundant,test_origin_splitted)
    modified_test.append([added_sentences_combined,words_redundant,diff_add,diff_redn,diff_whole])
        
        
        
        
    
        

#
#one_input=[]
#idx = 1
#for i in range(len(test_sentences[idx])):
##    X[i] = np.array(X[i])
#    one_input.append(" ".join(test_sentences[idx][i]))
##one_input_vector = vectorizer.transform(X[400:500])
#    
#one_input_vector = vectorizer.transform(one_input)
#one_input_vector_whole = vectorizer.transform([test_data_text[idx]])
#print(clf.predict(one_input_vector_whole))
#print(clf.predict(one_input_vector))
#print(clf.decision_function(one_input_vector_whole))
#print(clf.decision_function(one_input_vector))



# Generate modified data
#num_to_delete = 10
#neg_features_dict = dict(neg_features)
#test_data_modified_splitted = []
#for doc in test_data_splitted_new:
##    order = sorted(range(len(doc)), key=lambda k: doc[k][1],
##                   reverse=True)
##    idx=0
##    for i in order[:10]:
##        doc[i] = neg_features[idx]
##        idx+=1
#    # Summarize tokens coefficient
#    
#    test_data_coefficient = defaultdict(float)
#    for word, coefficient in doc:
#        test_data_coefficient[word] += coefficient
#    # print(test_data_coefficient.items())
#    test_data_coefficient = list(test_data_coefficient.items())
#    test_data_coefficient.sort(key = lambda x:x[1], reverse=True)
#
#    doc.sort(key = lambda x:x[1], reverse=True)
##    doc_preserved = doc.copy()
##    print(doc)
#    doc_text = [pair[0] for pair in doc]
#
#    word_set = set()
#    last_count = 0
##    idx_now = 0
#    origin_set = doc_text.copy()
#    replaced_amount = 0
#    deleted_amount = 0
#    
#    neg_features_copy = neg_features.copy()
#    positive_words = [x for x in test_data_coefficient if x[1] > 0]
##    for i in range(len(positive_words)):
##        replaced_amount = len((set(origin_set)-set(doc_text)) | (set(doc_text)-set(origin_set)))
###        print(replaced_amount)
##        if replaced_amount==20:
###            print(i)
##            break
##        tup = test_data_coefficient[i]
##        print(tup)
##        print(tup[1])
##        weight = tup[1]
##        if weight>0 and abs(weight)>abs(neg_features_copy[0][1]):
##            doc_text = list(filter(lambda x: x != tup[0], doc_text))
##            print("Deleted word: {}".format(tup[0]))
##        if weight>0 and abs(weight)<abs(neg_features_copy[0][1]):
##            for j in range(3):
##                word_appended = neg_features_copy.pop(0)[0]
##                doc_text.append(word_appended)
##                print("Added word: {}".format(word_appended))
#    idx = 0
##    print("Begin")
#    while(idx<len(positive_words)):
#        replaced_amount = len((set(origin_set)-set(doc_text)) | (set(doc_text)-set(origin_set)))
##        print(replaced_amount)
#        if replaced_amount==20:
##            print(i)
#            break
#        tup = positive_words[idx]
#        
##        print(tup)
#        weight = tup[1]
#        if abs(weight)>abs(neg_features_copy[0][1]):
#            doc_text = list(filter(lambda x: x != tup[0], doc_text))
#            idx+=1
##            print("Deleted word: {}".format(tup[0]))
#        if abs(weight)<abs(neg_features_copy[0][1]):
#            tuple_appended = neg_features_copy.pop(0)
#            word_appended = tuple_appended[0]
#            coef = tuple_appended[1]
#            for j in range(2):
#                doc_text.append(word_appended)
##                print("Added word: {}, {}".format(word_appended, coef))
#        
#        
#        
##    for i in range(len(neg_features)):
##        
##        replaced_amount = len((set(origin_set)-set(doc_text)) | (set(doc_text)-set(origin_set)))
###        print(replaced_amount)
##        if replaced_amount==20:
###            print(i)
##            break
##        neg_word = neg_features[i][0]
###        if doc_text.count(neg_word)>0:
###            continue
###        else:
##        for i in range(1):
##            doc_text.append(neg_word)
#        
#    if replaced_amount!=20:
#        print("Something wrong!!! The replaced amount is : {}".format(replaced_amount))
#        break
##    # Delete most positive words
##    for i in range(len(pos_features)):
##        replaced_amount = len((set(origin_set)-set(doc_text)) | (set(doc_text)-set(origin_set)))
###        print(replaced_amount)
##        if replaced_amount==10:
###            print(i)
##            break
##        pos_word = pos_features[i][0]
##        if doc_text.count(pos_word)>0:
##            #The next line referred to https://stackoverflow.com/questions/1157106/remove-all-occurrences-of-a-value-from-a-list
##            #This function removes all occurances of "pos_word" in doc_text
##            doc_text = list(filter(lambda x: x != pos_word, doc_text))
##            deleted_amount+=1
###            print("Deleted word: {}".format(pos_word))
#    # Add most negative words
##    for i in range(len(neg_features)):
##        replaced_amount = len((set(origin_set)-set(doc_text)) | (set(doc_text)-set(origin_set)))
###        print(replaced_amount)
##        if replaced_amount==20:
###            print(i)
##            break
##        neg_word = neg_features[i][0]
##        if doc_text.count(neg_word)>0:
##            continue
##        else:
##            for i in range(3):
##                doc_text.append(neg_word)
###                print("Added word: {}".format(neg_word))
##
##    replaced_amount = len((set(origin_set)-set(doc_text)) | (set(doc_text)-set(origin_set)))
###        print(replaced_amount)
#
##            print(i)
#
#
##    for i in range(len(neg_features)):
###    i=0
###    while i < len(doc_text):
##        word_replace = neg_features[i][0]
##        if doc_text.count(word_replace)>0:
##            continue
##
##        word= doc_text[last_count]
##        #The next line referred to https://stackoverflow.com/questions/1157106/remove-all-occurrences-of-a-value-from-a-list
##        if word not in neg_features_dict:
##            doc_text = list(filter(lambda x: x != word, doc_text))
##        doc_text.append(neg_features[i][0])
##        last_count+=1
##        #word_set.add(word)
##        replaced_amount = len((set(origin_set)-set(doc_text)) | (set(doc_text)-set(origin_set)))
###        print(replaced_amount)
##        if replaced_amount==20:
###            print(i)
##            break
##        if replaced_amount> last_count:
#
#
##        word_set.add(neg_features[len(word_set)][0])
##        idx_now+=1
#        #i+=1
#
#    #print("aaa: "+str(replaced_amount))
#    test_data_modified_splitted.append(doc_text)
#
##    doc_text = [pair[0] for pair in doc]
##    test_data.append(doc_text)
#
#
#modified_test_data=[]
#for i in range(len(test_data_modified_splitted)):
##    X[i] = np.array(X[i])
#    modified_test_data.append(" ".join(test_data_modified_splitted[i]))
#
#vectors_test_modified = vectorizer.transform(modified_test_data)
#
#file = open(modified_data,"w")
#for doc in modified_test_data:
#    file.write(doc+" \n")
#
#
#with open(modified_data,'r') as file:
#    modified_test_data2=[line.strip() for line in file]
#
##order = sorted(range(len(doc)), key=lambda k: doc[k][1])
##for i in order:
##    print(doc[i])
#print("modified_data_accuracy")
#print(clf.score(vectors_test_modified,y_test2))
#print(strategy_instance.check_data("./test_data.txt","./modified_data.txt"))

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