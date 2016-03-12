import csv
import gzip
###
import numpy as np
###

from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import BaggingRegressor
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.utils import np_utils
from sklearn import metrics
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler
import pandas
import numpy
import sklearn
from sknn.mlp import *
from sklearn.feature_extraction import DictVectorizer
from dateutil.parser import parse
from nltk.stem.snowball import SnowballStemmer
import editdistance
from sklearn.metrics import mean_squared_error
import math
from sklearn.ensemble import RandomForestRegressor
from nltk.corpus import stopwords
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from nltk.corpus import words


stopwords = set([word for word in stopwords.words('english')])
stemmer = SnowballStemmer('english')

word_set = set([stemmer.stem(word) for word in words.words()])
ids_have_word = {}

def words_in_common(words1, words2):
    sum = 0
    dct = set()
    for word1 in words1:
        if word1 == 'in.':
            word1 = 'inch'
        for word2 in words2:
            if word1 == 'in.':
                word2 = 'inch'
            if word1 not in dct and word2 not in dct and word2.find(word1) >= 0:
                dct.add(word1)
                dct.add(word2)
                sum += 1
    return sum

def edit_dist_less_two(words1, words2):
    sum = 0
    for word1 in words1:
        for word2 in words2:
            if len(word1) > 5 and len(word2) > 5 and editdistance.eval(word1, word2) <= 2:
                sum += 1
                break
    return sum

def weighted_count(words1, words2):
    dct = {}
    for word in words2:
        if word in dct:
            dct[word] += 1
        else:
            dct[word] = 1
    sm = 0
    for word in words1:
        if word in dct:
            sm += math.log(dct[word])
    return sm


def generate_features(raw_train):
    X = []
    # Create features
    for it, row in raw_train.iterrows():
        data_point = []
        pd_id = row['product_uid']
        data_point.append(len(row['search_term']))
        data_point.append(len(row['product_title']))
        search_stemmed = [''.join(e for e in stemmer.stem(word) if e.isalnum()) for word in row['search_term'].split() if stemmer.stem(word)]
        search_stemmed = ['in.' if w == 'inch' else w for w in search_stemmed]
        search_stemmed = ['ft.' if w == 'feet' else w for w in search_stemmed]
        title_stemmed = [''.join(e for e in stemmer.stem(word) if e.isalnum()) for word in row['product_title'].split() if stemmer.stem(word)]
        data_point.append(words_in_common(search_stemmed, title_stemmed))
        data_point.append(words_in_common(search_stemmed, descr_dict[pd_id]))
        data_point.append(edit_dist_less_two(search_stemmed, title_stemmed))
        data_point.append(edit_dist_less_two(search_stemmed, descr_dict[pd_id]))
        data_point.append(1 if search_stemmed[-1] in title_stemmed else 0)
        data_point.append(weighted_count(search_stemmed, descr_dict[pd_id]))
        search_joined = "".join([word for word in search_stemmed])
        title_joined = "".join([word for word in title_stemmed])
        data_point.append(len(search_stemmed) if (title_joined.find(search_joined) >= 0 and len(search_stemmed) > 1) else 0)
        X.append(data_point)
    X = numpy.array(X)
    return X

def add_to_dict(raw_data):
    for it, row in raw_data.iterrows():
        stemmed_words = [stemmer.stem(word) for word in row['product_title'].split()if stemmer.stem(word) not in stopwords]
        pd_id = row['product_uid']
        for word in stemmed_words:
            if word in ids_have_word:
                ids_have_word[word][pd_id] = True
            else:
                ids_have_word[word] = {pd_id: True}




raw_train = pandas.read_csv("/Users/patrickhess/Documents/kaggle/home_depot/train.csv", encoding="ISO-8859-1")
descriptions_raw = pandas.read_csv("/Users/patrickhess/Documents/kaggle/home_depot/product_descriptions.csv", encoding="ISO-8859-1")
raw_test = pandas.read_csv("/Users/patrickhess/Documents/kaggle/home_depot/test.csv", encoding="ISO-8859-1")
add_to_dict(raw_test)
add_to_dict(raw_train)


descr_dict = {}
descr_len = {}
# Stem descriptions
for it, row in descriptions_raw.iterrows():
    descr_dict[row['product_uid']] = [''.join(e for e in stemmer.stem(word) if e.isalnum()) for word in row['product_description'].split()]

X = generate_features(raw_train)
Xtest = generate_features(raw_test)

Y = []
for it, row in raw_train.iterrows():
    Y.append(row['relevance'])

Y = numpy.array(Y)

def build_model(input_dim, hn=64, dp=0.5, layers=1):
    model = Sequential()
    model.add(Dense(input_dim=input_dim, output_dim=hn, init='he_uniform'))
    model.add(PReLU(input_shape=(hn,)))
    model.add(Dropout(dp))
    for i in range(layers):
        model.add(Dense(input_dim=hn, output_dim=hn, init='he_uniform'))
        model.add(PReLU(input_shape=(hn,)))
        model.add(BatchNormalization(input_shape=(hn,)))
        model.add(Dropout(dp))
    model.add(Dense(input_dim=hn, output_dim=1))
    model.compile(loss='mse', optimizer='adam')
    return model


model = build_model(len(X[0]), 48, 0.35, 1)
model.fit(X, Y, nb_epoch=32, batch_size=84, verbose=2)
#rfc = RandomForestRegressor(n_estimators=100, min_samples_split=10)
#rfc.fit(X, Y)


# Classifier
model2 = xgb.XGBRegressor()
params = {
        'n_estimators': [100, 150, 250],
        'learning_rate': [0.05, 0.1, 0.3],
        'max_depth': [3, 6, 9, 12],
        'subsample': [0.25, 0.35, 0.5, 0.7],
        'colsample_bytree': [0.3, 0.5, 0.75, 0.9]}

clf1 = GridSearchCV(model2, params, verbose=1)
clf1.fit(X, Y)
print(clf1.best_score_)
print(clf1.best_params_)

num_boost_round = 112
dtrain = xgb.DMatrix(X, Y)
#clf1 = xgb.cv(params=params, dtrain=dtrain, num_boost_round=num_boost_round, early_stopping_rounds=10, nfold=5)
clf1 = xgb.train(params=params, dtrain=dtrain, num_boost_round=num_boost_round)
#clf1.predict(xgb.DMatrix(X))
math.sqrt(mean_squared_error(Y[60000:], clf1.predict(xgb.DMatrix(X[60000:]))))

Ypred = [max(1, min(3, y)) for y in clf1.predict(xgb.DMatrix(Xtest))]
Ypred = numpy.add(Ypred, [y[0] for y in model.predict(Xtest)])
Ypred /= 2
id_test = raw_test['id']
pandas.DataFrame({"id": id_test, "relevance": Ypred}).to_csv('submission.csv',index=False)

