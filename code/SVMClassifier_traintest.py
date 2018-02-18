# coding: utf8
import sys
import pandas as pd
import numpy as np
import scipy as sp
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import string
from nltk.tokenize import regexp_tokenize
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support as prfs_score
from sklearn import cross_validation
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.externals import joblib
import pickle
import json

import codecs
import csv
from sklearn.model_selection import KFold  # import KFold

reload(sys)
sys.setdefaultencoding('utf8')

stemmer = PorterStemmer()

data_json = json.load(codecs.open(
    'selected_lang_stopwords.json', 'r', 'utf-8'))

s = set(data_json['en'])
s.update(data_json['es'])
s.update(data_json['it'])
s.update(data_json['fr'])
s.update(data_json['pt'])
s.update(data_json['hi'])
s.update(data_json['nl'])
s.update(data_json['ru'])
s.update(data_json['de'])

fs = frozenset(s)


def tokenize_and_stem(text):
    # tokens = word_tokenize(text)
    # tokens = regexp_tokenize(text, pattern=r"\s|[\.,:;'()?!]", gaps=True)
    tokens = regexp_tokenize(text, pattern=r"\s|[\.,:;'()?!]", gaps=True)
    # strip out punctuation and make lowercase
    tokens = [token.lower().strip(string.punctuation)
              for token in tokens if token.isalnum()]

    # now stem the tokens
    tokens = [stemmer.stem(token) for token in tokens]

    return tokens


# folder = 'statistical_features'
# folder = 'stat_n_hypernym_en'
# folder = 'stat_n_hypernym_dbpedia_en'
folder = 'stat_n_dbpedia_en'

# folder = 'statistical_features'
# folder = 'stat_n_hypernym_en/filtered_hypernym_en'
# folder = 'stat_n_hypernym_dbpedia_en/filtered_hypernym_en_dbpedia_semantics'
# folder = 'stat_n_dbpedia_en/filtered_dbpedia_semantics'

# file_list = ['TPY', 'TYP', 'ALB', 'QFL', 'GAU', 'LAM', 'RUS', 'IT', 'ES', 'PT']
# file_list = ['FR', 'WTX', 'CFL', 'PHF', 'CWF']
# file_list = ['TPY', 'TYP', 'ALB', 'QFL', 'GAU', 'LAM', 'RUS',
#             'IT', 'ES', 'PT', 'FR', 'WTX', 'CFL', 'PHF', 'CWF']
# file_list = ['LAX', 'BOB', 'SPT', 'LAM']
# file_list = ['IT']
# file_list = ['LAM']
# file_list = ['TPY', 'TYP', 'ALB', 'QFL', 'CFL', 'PHF', 'SAR']
# file_list = ['GAU']
# file_list = ['PHF']
# file_list = ['ITL', 'BOL', 'COS', 'SAR']
# file_list = ['GAU', 'ITL', 'BOL', 'COS']
# file_list = ['SAR']
# file_list = ['SGR']
# file_list = ['TPY']
# file_list = ['it', 'en', 'es', 'fr']
file_list = ['it', 'en', 'es', 'pt']

for f_lst in file_list:

    # f_lst_temp = 'SPT'

    # train_file = 'train_' + f_lst + '_balanced_random_stat_hypernym_dbpedia_en_semantics.csv'
    # f_lst_temp = 'NOFLTY'
    # f_lst_temp = 'NOEQ'
    # f_lst_temp = 'pt'
    # train_file = 'train_' + f_lst_temp + '_balanced_random_stat_hypernym_dbpedia_en_semantics.csv'
    # test_file = 'test_' + f_lst + '_balanced_random_stat_hypernym_dbpedia_en_semantics.csv'

    # train_file = 'all_' + f_lst_temp + '_balanced_random_stat_dbpedia_en_semantics.csv'
    # test_file = 'all_' + f_lst + '_balanced_random_stat_dbpedia_en_semantics.csv'

    train_file = 'train_just_' + f_lst + '_balanced_random_stat_dbpedia_en_semantics.csv'
    test_file = 'test_just_' + f_lst + '_balanced_random_stat_dbpedia_en_semantics.csv'

    # train_file = 'train_ALB_balanced_stat_random.csv'
    # test_file = 'test_ALB_balanced_stat_random.csv'

    print '\n'

    print 'Starting new classification round:' + ' ' + folder

    print test_file[0:8]
    print test_file[0:12]

    data_csv = []
    data = ''
    data_csv_test = []
    data_test = ''

    # with open('../data/statistical_features/balanced_stat_random.csv') as fline:
    with open('../data/' + folder + '/' + train_file) as fline:
        data_csv = list(csv.reader(fline, delimiter="\t"))

        data = np.array(data_csv[0:])

    # For statistical_features file arrangement X = data[:, [1, 7, 8, 9, 10, 11, 12]]
    # X = data[:, [1, 7, 8, 9, 10, 11, 12]]
    # Y = data[:, 4].astype(np.float32)
    # X = data[:, 1]  # just for the tweets based features
    # Y = data[:, 4].astype(np.float32)
    # Y = data[:, 2].astype(np.float32)  # jusr for the dbpedia balanced tweets

    X = data[:, 1:8]
    Y = data[:, 8].astype(np.float32)

    t = ['Document', 'NumberOfNouns', 'NumberOfVerbs', 'NumberOfPronouns',
         'TweetLength', 'NumberOfWords', 'NumberOfHashTag']

    # t = ['Document']

    # str(nouns), str(verbs), str(pronouns), str(tweet_length), str(token_count), str(numHashTag)

    frm = pd.DataFrame(X, columns=t)
    # print type(frm)

    vectorizer = CountVectorizer(analyzer='word', tokenizer=tokenize_and_stem,
                                 stop_words=fs, lowercase=True, ngram_range=(1, 1), max_features=40000)
    doc_vectorize = vectorizer.fit_transform(frm.Document)
    tf_transform = TfidfTransformer()
    tf_vectorize = tf_transform.fit_transform(doc_vectorize)

    X_data = sp.sparse.hstack((tf_vectorize, frm[['NumberOfNouns', 'NumberOfVerbs', 'NumberOfPronouns', 'TweetLength',
                                                  'NumberOfWords', 'NumberOfHashTag']].values.astype(np.float32)), format='csr')

    # X_data = tf_vectorize

    # print type(X_data)
    # print type(Y)
    print X_data.shape
    # X_data.toarray().shape

    # create training and testing vars
    # X_train, X_test, y_train, y_test = train_test_split(X_data, Y, test_size=0.3)

    #############################
    print 'done1'
    unique, counts = np.unique(Y, return_counts=True)
    print dict(zip(unique, counts))

    svc = SVC(kernel='linear', degree=3, gamma='auto', tol=0.001)
    # print X_train.shape
    # print y_train.shape
    svc.fit(X_data, Y)
    print 'done2'
    # predictions = svc.predict(X_test)

    # print (metrics.f1_score(y_test, predictions))

    # scores = cross_validation.cross_val_score(svc, X_data, Y, cv=5)
    '''scores = cross_validation.cross_val_predict(svc, X_data, Y, cv=5)

    print scores.shape
    p, r, f, s = prfs_score(Y, scores.astype(np.float32))

    print('precision: {}'.format(p))
    print('recall: {}'.format(r))
    print('fscore: {}'.format(f))
    print('support: {}'.format(s))
    print('overall f-1 score: {}'.format(f1_score(Y, scores.astype(np.float32))))
    print('overall precision score: {}'.format(precision_score(Y, scores.astype(np.float32))))
    print('overall recall score: {}'.format(recall_score(Y, scores.astype(np.float32))))
    print('confusion matrix: {}'.format(confusion_matrix(Y, scores.astype(np.float32))))
    print ('************ Classification Report ************')
    print classification_report(Y, scores.astype(np.float32))'''

    # for test test validation
    with open('../data/' + folder + '/' + test_file) as fline2,\
            open('../data/statistical_features/balanced_stat_random.csv') as flineX:

        data_csv_balanced_full = list(csv.reader(flineX, delimiter="\t"))
        data_balanced_full = np.array(data_csv_balanced_full[0:])
        lang = dict()
        correct_classify = dict()
        incorrect_classify = dict()

        for rec in data_balanced_full:
            lang[rec[0]] = rec[5]

        data_csv_test = list(csv.reader(fline2, delimiter="\t"))

        data_test = np.array(data_csv_test[0:])

        # For statistical_features file arrangement X_test = data[:, [1, 7, 8, 9, 10, 11, 12]]
        # X_test = data_test[:, [1, 7, 8, 9, 10, 11, 12]]
        # Y_test = data_test[:, 4].astype(np.float32)
        # X_test = data_test[:, 1]  # just for the tweets based features
        # Y_test = data_test[:, 4].astype(np.float32)
        # Y_test = data_test[:, 2].astype(np.float32)  # just for the dbpedia balanced tweets

        X_test = data_test[:, 1:8]
        Y_test = data_test[:, 8].astype(np.float32)

        t_test = ['Document', 'NumberOfNouns', 'NumberOfVerbs', 'NumberOfPronouns',
                  'TweetLength', 'NumberOfWords', 'NumberOfHashTag']

        # t_test = ['Document']

        frm_test = pd.DataFrame(X_test, columns=t_test)

        test_vectorize = vectorizer.transform(frm_test.Document)

        test_tf_vectorize = tf_transform.transform(test_vectorize)

        X_data_test = sp.sparse.hstack((test_tf_vectorize, frm_test[['NumberOfNouns', 'NumberOfVerbs', 'NumberOfPronouns',
                                                                     'TweetLength', 'NumberOfWords', 'NumberOfHashTag']].values.astype(np.float32)), format='csr')

        # X_data_test = test_tf_vectorize

        print 'now shapes of test data'
        # print type(X_data_test)
        # print type(Y_test)
        print X_data_test.shape
        unique, counts = np.unique(Y_test, return_counts=True)
        print dict(zip(unique, counts))

        Y_test_predict = svc.predict(X_data_test)

        print Y_test_predict.shape
        p, r, f, s = prfs_score(Y_test, Y_test_predict.astype(np.float32))

        print('precision: {}'.format(p))
        print('recall: {}'.format(r))
        print('fscore: {}'.format(f))
        print('support: {}'.format(s))
        print('overall precision score: {}'.format(
            precision_score(Y_test, Y_test_predict.astype(np.float32), average='macro')))
        print('overall recall score: {}'.format(
            recall_score(Y_test, Y_test_predict.astype(np.float32), average='macro')))
        print('overall f-1 score: {}'.format(f1_score(Y_test,
                                                      Y_test_predict.astype(np.float32), average='macro')))
        print('confusion matrix: {}'.format(confusion_matrix(Y_test, Y_test_predict.astype(np.float32))))
        print ('************ Classification Report ************')
        print classification_report(Y_test, Y_test_predict.astype(np.float32))

        # print Y_test_predict
        for l in range(0, len(Y_test)):
            if Y_test[l].astype(np.float32) == Y_test_predict[l].astype(np.float32):
                # print 'c'

                # print data_test[l][0]
                # print lang[data_test[l][0]]

                if lang[data_test[l][0]] in correct_classify:
                    correct_classify[lang[data_test[l][0]]
                                     ] = correct_classify[lang[data_test[l][0]]] + 1
                else:
                    correct_classify[lang[data_test[l][0]]] = 1

            else:
                # print 'n'
                if lang[data_test[l][0]] in incorrect_classify:
                    incorrect_classify[lang[data_test[l][0]]
                                       ] = incorrect_classify[lang[data_test[l][0]]] + 1
                else:
                    incorrect_classify[lang[data_test[l][0]]] = 1

        print 'correct_classify:\n'
        print correct_classify
        print 'incorrect_classify:\n'
        print incorrect_classify
