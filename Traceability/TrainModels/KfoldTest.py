import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score
import random
import pickle
import json


#Seems that discardnoneSentences, and test_whole_dataset doesn't work well. I'll train the models with the best parameters using train/test splits and testing against the whole dataset, to then manually check some sentences from policies.

def RunKFoldTest(configs, indexToLabel, discardNone = False):
    #Create index to label dict (Not used otherwise the dict can always return different orderings
    #indexToLabel =  {key: value for (key, value) in enumerate(list(df['Permission'].unique()))}

    print(indexToLabel)
    labelToindex = {v:k for k,v in indexToLabel.items()}
    print(labelToindex)
    
    '''
    prepare dataset
    1. Jide Alexa's excel file
    '''
    path = 'data/Annotated_Policies_Alexa.csv'
    df = pd.read_csv(path)
    df = df[pd.notnull(df['Permission'])]
    df = df[pd.notnull(df['Sentence'])]
    df.drop('Similarities_Score',axis=1,inplace=True)


    for c in df['Permission'].unique():
        df.loc[ (df['Permission'] == c), 'Label'] = labelToindex[c]
    df['Label'] = np.int64(df['Label'])

    #discard None sentences, as they will be filtered with previous model
    if(discardNone is True):
        df = df.loc[df['Permission']!='None']
    
    
    
    #'''
    #APP350 category mapping (THIS DEPENDS ON THE LABEL DICTIONARY CREATED indexToLabel!)
    #2. Device Address --- contact postal address
    #3. Device country and postal code --- contact ct, contact zip
    #4. Email address --- contact email address
    #5. Location Services --- location gps
    #6. Mobile Number ---  contact phone number
    #'''
    ##this dict lets us map the different app350 categories to alexa cats
    #filenametolabel = {
    #    'data/App350/Contact_Postal_Address_3rdParty.txt' : 2,
    #    'data/App350/Contact_Postal_Address_1stParty.txt' : 2,
    #    'data/App350/Contact_ZIP_3rdParty.txt' : 3,
    #    'data/App350/Contact_ZIP_1stParty.txt' : 3,
    #    'data/App350/Contact_City_3rdParty.txt' : 3,
    #    'data/App350/Contact_City_1stParty.txt' : 3,
    #    'data/App350/Contact_E_Mail_Address_3rdParty.txt' : 4,
    #    'data/App350/Contact_E_Mail_Address_1stParty.txt' : 4,
    #    'data/App350/Location_GPS_3rdParty.txt' : 5,
    #    'data/App350/Location_GPS_1stParty.txt' : 5,
    #    'data/App350/Contact_Phone_Number_3rdParty.txt' : 6,
    #    'data/App350/Contact_Phone_Number_1stParty.txt' : 6,
    #}
#
    #for filepath in filenametolabel.keys():
    #    with open(filepath, encoding='utf8') as f:   #utf8      
    #        label = filenametolabel[filepath]
    #        permission = indexToLabel[label]
    #        text = f.readlines()
    #        text = [k.replace('\n', '').strip() for k in text]
    #        for t in text:
    #            df = df.append({'Permission':permission, 'Sentence':t, 'Label':label},  ignore_index=True)

    '''
    Prepare the full datset
    full dataset category distribution

    Amazon Pay -  166
    None -  6451
    Device Address -  633
    Device country and postal code -  294
    Email Address -  2205
    Location Services -  516
    Mobile Number -  934
    Name -  396
    Personal Information -  2367
    Skill Personisation -  57
    '''
    #create columns
    for index in indexToLabel.keys():
        df.loc[df['Label'] == index, 'Label'+str(index)] = 0
        df.loc[df['Label'] != index, 'Label'+str(index)] = 1


    '''
    One vs all, train and test appraoch
    min_ngrams = 1
    max_ngrams = 3
    SVMloss = [modified_huber, hinge]
    CountVectorizerBinary = [binary =True]
    SGDClassifier maxiter = [1000, 10000]
    alpha = [1e-3, 1e-5]
    #ksplits = depende de la clase
    #ngrams  = depende de la clase
    use_idf = True
    classproportion = 5 #max proportion difference between class0 and class1, to balance the dataset classes. 
    ksplits = 5         #k-fold k splits

    class = 0 #class to test
    '''
    #using SVM linear
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import SGDClassifier
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.model_selection import train_test_split


    '''
    load different configs and instantiate
    '''
    alloutputdata = []           #structure we will use to save and return all data
    for config in configs:
        print('********************')
        print('  PROCESSING CONFIG')
        print(config)
        print('********************')


        outputdata = {}                       #data for the actual test
        i = config['class_alexa_label_index'] #label of the class to test, 0..9
        classproportion = config['classproportion']
        ksplits = config['ksplits']
        ngram_min = config['ngram_min']
        ngram_max = config['ngram_max']
        use_tfidf = config['use_tfidf']
        count_binary = config['count_binary']
        svm_loss = config['svm_loss']
        svm_alpha = config['svm_alpha']
        svm_iter = config['svm_iter']
        test_whole_dataset = config['test_whole_dataset']


        #define the pipeline
        text_clf = Pipeline([
        ('vect', CountVectorizer(ngram_range=(ngram_min, ngram_max), binary=count_binary)),
        ('tfidf', TfidfTransformer(use_idf=use_tfidf)),
        ('clf', SGDClassifier(loss=svm_loss, penalty='l2',
                              alpha=svm_alpha, random_state=42,
                              max_iter=svm_iter, tol=None,  class_weight='balanced')),
        ])


        label = 'Label'+str(i)
        print('*Testing for ', label)
        print()


        #random sample to balance number of instances between classes
        X0 = df.loc[df[label] == 0]['Sentence'].tolist()
        xn0 = len(X0)
        X1 = df.loc[df[label] == 1]['Sentence'].tolist()
        xn1 = len(X1)
        if(len(X1)>len(X0)*classproportion):
            X1 = random.sample(X1, xn0*classproportion)
            xn1 = len(X1)
        elif(len(X0)>len(X1)*classproportion):
            X0 = random.sample(X0, xn1*classproportion)
            xn0 = len(X0)
            
        #save a collection of all sentences for 'test_whole_dataset' testing purposes
        Xall = df.loc[df[label] == 0]['Sentence'].tolist()+df.loc[df[label] == 1]['Sentence'].tolist()
        yall = len(df.loc[df[label] == 0]['Sentence'].tolist())*[0] + len(df.loc[df[label] == 1]['Sentence'].tolist())*[1]
        print('Original length of all comments ' , len(Xall) )
        
        #then check for repeated sentences belonging to both X0 and X1. If it belongs to X0, we remove it from X1.
        #This happens, sentences are repeated in the dataset as belonging to more than one privacy category, hence we clean it
        #before letting the classifier learn
        totalr = 0
        for s0 in X0:
            s0 = s0.strip()
            ir = [index for index, s in enumerate(X1) if s.strip()==s0]
            ir = sorted(ir, reverse = True) #need to sort in reverse so when deleting an index does not affect the res of indices
            for index in ir:
                del X1[index]
            totalr += len(ir)
        print('Removed', totalr, 'repeated sentences from X1')

        #create the input model lists
        X = X0+X1
        X = np.array(X)
        xn0 = len(X0);xn1 = len(X1)
        y = labels = np.array(xn0*[0] + xn1*[1])

        print('all data_balanced', len(X))
        print('data_balanced_class_0', len(X0), 'data_balanced_class_1', len(X1))


        kfold = StratifiedKFold(n_splits=ksplits, shuffle=True, random_state=1)
        # enumerate the splits and summarize the distributions
        acc_acc = 0                                #average accuracy
        acc_f1 = 0                                 #average f1
        cm_total = np.array([[0.,0.],[0.,0.]])     #agrgeagte the percantages in a CM
        cm_total_values = np.array([[0,0],[0,0]])  #agrgeagte the total hits of the CM
        for train_ix, test_ix in kfold.split(X, y):
            train_X, test_X = X[train_ix], X[test_ix]
            train_y, test_y = y[train_ix], y[test_ix]
            
            #variable to test with the whole rest of the dataset, this way we simulate a real-world situation
            #in which classes are not balanced for testing
            if(test_whole_dataset):                
                #first get the rest of the docuemnts not used to train
                test_X = [];test_y = []
                for ind, x in enumerate(Xall):                    
                    if(x not in train_X):
                        test_X.append(x)
                        test_y.append(yall[ind])
                test_y = np.array(test_y)
                test_X = np.array(test_X)
            
                        
            
            #get some data on train/test splits
            # summarize train and test composition
            train_0, train_1 = len(train_y[train_y==0]), len(train_y[train_y==1])
            test_0, test_1 = len(test_y[test_y==0]), len(test_y[test_y==1])
            print('\t>Train: 0=%d, 1=%d, Test: 0=%d, 1=%d' % (train_0, train_1, test_0, test_1))
            print('\tTotal train+test {}'.format( len(train_X)+len(test_X) ))

            
            #train and test
            text_clf.fit(train_X, train_y)          
            pred_y = text_clf.predict(test_X)
            #plot_confusion_matrix(clf, test_X, test_y)
            cm = confusion_matrix(test_y, pred_y)
            cm_total_values += np.array(cm)
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_total+=np.array(cm)
            f1 = f1_score(test_y, pred_y, average=None)[0]
            acc_f1 += f1
   
            acc = np.mean(pred_y == test_y) 
            print('*\t',indexToLabel[i], '(', i, '):', acc, 'f1', f1)
            acc_acc+= acc

            for i,sentence in enumerate(test_X):
                sentence = test_X[i]
                realy = test_y[i]
                testedy = pred_y[i]
                print(sentence, ' real:', realy,'/ predicted:', testedy)
                if(i == 100):
                    break
            
            return
            
        print('TOTALS:')
        print(cm_total_values)
        print('AVG Percentages:')
        print(cm_total/ksplits)
        print('**',indexToLabel[i], '(', i, ') Average precision:', acc_acc/ksplits)

        outputdata['test_whole_dataset'] = test_whole_dataset
        outputdata['config'] = config
        outputdata['cm_total_values'] = (cm_total_values).tolist()
        outputdata['cm_total_percantages_avg'] = (cm_total/ksplits).tolist()
        outputdata['avg_precision'] = acc_acc/ksplits
        outputdata['avg_f1'] = acc_f1/ksplits
        outputdata['class_name'] =  indexToLabel[i]
        outputdata['data_balanced_class_0'] = len(X0)
        outputdata['data_balanced_class_1'] = len(X1)
        outputdata['data_balanced_total'] = len(X)
        outputdata['train0'] = train_0
        outputdata['train1'] = train_1
        outputdata['test0'] = test_0
        outputdata['test1'] = test_1
        alloutputdata.append(outputdata)

    alloutputdata = sorted(alloutputdata, key=lambda x : x['avg_f1'], reverse=True)    
    ofile = 'outputs/'+str(label)+'.json'
    with open(ofile, 'w') as f:
        json.dump(alloutputdata, f)

    print('[DATA DUMPED] in ', ofile)    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    