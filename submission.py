import helper
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
def fool_classifier(test_data): ## Please do not change the function defination...
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
#    X_train_class0, X_test_class0,y_train_class0, y_test_class0 = \
#    train_test_split(strategy_instance.class0,\
#                      [0]*len(strategy_instance.class0),\
#                      test_size=0)
    X_train_class0 = strategy_instance.class0
    y_train_class0 = [0]*len(strategy_instance.class0)
    X = X_train_class0+ strategy_instance.class1
    for i in range(len(X)):
    #    X[i] = np.array(X[i])
        X[i] = " ".join(X[i])
    
    
    Y= y_train_class0 + [1]*len(strategy_instance.class1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0)
    
    
    vectorizer = CountVectorizer(max_df=1.0, min_df=1, binary = True, ngram_range=(1,1), max_features=5720, lowercase=False)
    
    
    vectors_train = vectorizer.fit_transform(X_train)
    
    
    clf = strategy_instance.train_svm(parameters, vectors_train,y_train)
    #clf = LinearSVC(random_state=0, C = 100).fit(vectors_train,y_train)
#    print("train accuracy")
#    print(clf.score(vectors_train,y_train))
    
#    test_data='./test_data.txt'
    test_data_splitted = []
    with open(test_data,'r') as file:
        test_data_splitted=[line.strip().split(' ') for line in file]
#    test_data_splitted_backup = test_data_splitted.copy()
    
    
    test_data_text=[]
    for i in range(len(test_data_splitted)):
    #    X[i] = np.array(X[i])
        test_data_text.append(" ".join(test_data_splitted[i]))
    
#    y_test2 = [1]*len(test_data_text)
    #vectors_test2 = vectorizer.transform(test_data_text)
#    vectors_test2 = vectorizer.transform(test_data_text)
#    print("test_data_accuracy")
#    print(clf.score(vectors_test2,y_test2))
    
    
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
    
    
        
    train_0 = strategy_instance.class0.copy()
    
    
    train_1 = strategy_instance.class1.copy()
    
    
    
    train_0_list = []
    for doc in train_0:
        train_0_list+=list(set(doc))
    train_0_set = set(train_0_list)
    
    train_1_list = []
    for doc in train_1:
        train_1_list+=list(set(doc))
    train_1_set = set(train_1_list)
    
    
    ratios_train_0 = []
    ratios_dict_0 = {}
    for word in train_0_set:
        word_count_0 = train_0_list.count(word)
        word_count_1 = train_1_list.count(word)
        ratio_word_0 = word_count_0/len(train_0_list)
        ratio_word_1 = (word_count_1/len(train_1_list))
        if ratio_word_1==0 and word_count_0>=8:
            ratio=1000000000
    #        continue
        elif ratio_word_1==0 and word_count_0<8:
            continue
        else:
            ratio = ratio_word_0/ratio_word_1
    #    ratio = word_count_0/(word_count_0 + word_count_1)
        if word in pairs_dict:
            coef = pairs_dict[word]
        else:
            coef = 0
    #    ratios_train_0.append((word,word_count_0,word_count_1,ratio,coef))
        ratios_train_0.append((word,ratio))
        ratios_dict_0[word] = (word_count_0,word_count_1,ratio,coef)
    ratios_train_0.sort(key = lambda x:x[1], reverse = True)
    
    ratios_train_1 = []
    ratios_dict_1 = {}
    for word in train_1_set:
        word_count_0 = train_0_list.count(word)
        word_count_1 = train_1_list.count(word)
        ratio_word_0 = word_count_0/len(train_0_list)
        ratio_word_1 = (word_count_1/len(train_1_list))
        if ratio_word_0==0 and word_count_1>=8:
            ratio=1000000000
    #        continue
        elif ratio_word_0==0 and word_count_1<8:
            continue
        else:
            ratio = ratio_word_1/ratio_word_0
    #    ratio = word_count_0/(word_count_0 + word_count_1)
        if word in pairs_dict:
            coef = pairs_dict[word]
        else:
            coef = 0
    #    ratios_train_1.append((word,word_count_0,word_count_1,ratio,coef))
        ratios_train_1.append((word,ratio))
        ratios_dict_1[word] = (word_count_0,word_count_1,ratio,coef)
    ratios_train_1.sort(key = lambda x:x[1], reverse = True)
    
    test_data_splitted_modified = []
    for i in range(len(test_data_splitted)):
        doc = test_data_splitted[i]
#        new_doc = doc.copy()
        add_candidates = []
        delete_candidates=[]
        for pos in ratios_train_1:
            if doc.count(pos[0])>0:
                delete_candidates.append(pos)
        for neg in ratios_train_0:
            if doc.count(neg[0])>0:
                continue
            add_candidates.append(neg)
            
        idx_pos = 0
        idx_neg = 0
        origin_set = set(doc)
        doc_modified = list(set(doc))
        diff =0
#        print("begin modifying: ")
        added_num = 0
        deleted_num = 0
        if len(delete_candidates)>1:    
            delete_flag = 1
        while(diff!=20):
            
            neg = add_candidates[idx_neg]
            pos = delete_candidates[idx_pos]
            #this part can control the amount of added words
            if abs(neg[1])> abs(pos[1]) and (added_num<3 or delete_flag==0):
                doc_modified.append(neg[0])
#                print("add: {}".format(neg[0]))
                idx_neg+=1
                added_num+=1
            else:
                doc_modified.remove(pos[0])
#                print("delete: {}".format(pos[0]))
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
    
#    vectors_test_modified = vectorizer.transform(modified_test_data)
    
    file = open(modified_data,"w")
    for doc in modified_test_data:
        file.write(doc+" \n")   
    file.close()
    #with open(modified_data,'r') as file:
    #    modified_test_data2=[line.strip() for line in file]
    
#    print(strategy_instance.check_data("./test_data.txt","./modified_data.txt"))
    assert strategy_instance.check_data(test_data, modified_data)
    return strategy_instance ## NOTE: You are required to return the instance of this class.
