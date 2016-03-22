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
documents_its_in = {}
num_documents = 0

def words_in_common(words1, words2):
    sum = 0
    dct = set()
    for word1 in words1:
        for word2 in words2:
            if word1 not in dct and word2 not in dct and word2.find(word1) >= 0:
                dct.add(word1)
                dct.add(word2)
                sum += math.log(num_documents / documents_its_in[word1])
    return sum

def edit_dist_less_two(words1, words2):
    sum = 0
    for word1 in words1:
        for word2 in words2:
            if len(word1) > 5 and len(word2) > 5 and editdistance.eval(word1, word2) <= 2 and word1[0] == word2[0]:
                sum += 1
                break
    return sum

def generate_features(raw_train):
    X = []
    # Create features
    for it, row in raw_train.iterrows():
        data_point = []
        pd_id = row['product_uid']
        data_point.append(len(row['search_term']))
        data_point.append(len(row['product_title']))
        search_stemmed = [''.join(e for e in stemmer.stem(word) if e.isalnum()) for word in row['search_term'].split() if stemmer.stem(word)]
        search_stemmed = ['inch' if w == 'in.' else w for w in search_stemmed]
        search_stemmed = ['ft' if (w == 'feet' or w == 'foot') else w for w in search_stemmed]
        search_stemmed = ['lb' if (w == 'pound' or w == 'pounds') else w for w in search_stemmed]
        title_stemmed = [''.join(e for e in stemmer.stem(word) if e.isalnum()) for word in row['product_title'].split() if stemmer.stem(word)]
        title_stemmed = ['inch' if w == 'in.' else w for w in title_stemmed]
        title_stemmed = ['ft' if (w == 'feet' or w == 'foot') else w for w in title_stemmed]
        title_stemmed = ['lb' if (w == 'pound' or w == 'pounds') else w for w in title_stemmed]
        data_point.append(words_in_common(search_stemmed, title_stemmed))
        data_point.append(words_in_common(search_stemmed, descr_dict[pd_id]))
        data_point.append(edit_dist_less_two(search_stemmed, title_stemmed))
        data_point.append(edit_dist_less_two(search_stemmed, descr_dict[pd_id]))
        data_point.append(1 if search_stemmed[-1] in title_stemmed else 0)
        search_joined = "".join([word for word in search_stemmed])
        title_joined = "".join([word for word in title_stemmed])
        data_point.append(len(search_stemmed) if (title_joined.find(search_joined) >= 0 and len(search_stemmed) > 1) else 0)
        X.append(data_point)
    X = numpy.array(X)
    return X


raw_train = pandas.read_csv("/Users/patrickhess/Documents/kaggle/home_depot/train.csv", encoding="ISO-8859-1")
descriptions_raw = pandas.read_csv("/Users/patrickhess/Documents/kaggle/home_depot/product_descriptions.csv", encoding="ISO-8859-1")
raw_test = pandas.read_csv("/Users/patrickhess/Documents/kaggle/home_depot/test.csv", encoding="ISO-8859-1")

descr_dict = {}

# Stem descriptions
for it, row in descriptions_raw.iterrows():
    words = [''.join(e for e in stemmer.stem(word) if e.isalnum()) for word in row['product_description'].split()]
    descr_dict[row['product_uid']] = words
    x = set()
    for word in words:
        word = 'inch' if word == 'in.' else word
        word = 'ft' if (word == 'foot' or word == 'feet.') else word
        word = 'lb' if  (word == 'pounds' or word == 'pound') else word
        if word not in x:
            if word in documents_its_in:
                documents_its_in[word] += 1
            else:
                documents_its_in[word] = 1
        x.add(word)

    num_documents += 1

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


model = build_model(len(X[0]), 64, 0.35, 1)
model.fit(X, Y, nb_epoch=32, batch_size=84, verbose=2)
#rfc = RandomForestRegressor(n_estimators=100, min_samples_split=10)
#rfc.fit(X, Y)


# Classifier
model2 = xgb.XGBRegressor()
params = {
        'n_estimators': [145],
        'learning_rate': [0.06],
        'max_depth': [13],
        'subsample': [0.88],
        'colsample_bylevel': [0.9],
        'colsample_bytree': [0.75],
         'min_child_weight': [3] }

clf1 = GridSearchCV(model2, params, verbose=1, n_jobs=8, cv=7)
clf1.fit(X, Y)
print(clf1.best_score_)
print(clf1.best_params_)

# best actual {'colsample_bytree': 0.78, 'colsample_bylevel': 0.9, 'learning_rate': 0.05, 'min_child_weight': 3, 'n_estimators': 136, 'subsample': 0.88, 'max_depth': 9}
best_params = {'colsample_bytree': 0.78, 'colsample_bylevel': 0.9, 'learning_rate': 0.05, 'min_child_weight': 3, 'n_estimators': 136, 'subsample': 0.88, 'max_depth': 9}
num_boost_round = 120
dtrain = xgb.DMatrix(X, Y)
#clf1 = xgb.cv(params=best_params, dtrain=dtrain, num_boost_round=num_boost_round, early_stopping_rounds=10, nfold=5)
clf2 = xgb.train(params=params, dtrain=dtrain, num_boost_round=num_boost_round)
#clf1.predict(xgb.DMatrix(X))
math.sqrt(mean_squared_error(Y[60000:], clf1.predict(xgb.DMatrix(X[60000:]))))

Ypred = [0.7 * max(1, min(3, y)) for y in clf1.predict(Xtest)]
Ypred = numpy.add(Ypred, [min(3, y[0]) * 0.3 for y in model.predict(Xtest)])
Ypred = [min(3, y) for y in Ypred]
id_test = raw_test['id']
pandas.DataFrame({"id": id_test, "relevance": Ypred}).to_csv('submission.csv',index=False)
