import csv
import gzip
###
import numpy as np
###
from keras.layers.advanced_activations import PReLU
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


stemmer = SnowballStemmer('english')
def words_in_common(words1, words2):
    sum = 0
    for word1 in words1:
        for word2 in words2:
            if word2.find(word1) >= 0:
                sum +=1
                break
    return sum

def edit_dist_less_two(words1, words2):
    sum = 0
    for word1 in words1:
        for word2 in words2:
            if len(word1) > 5 and len(word2) > 5 and editdistance.eval(word1, word2) <= 2:
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
        search_stemmed = [stemmer.stem(word) for word in row['search_term'].split()]
        title_stemmed = [stemmer.stem(word) for word in row['product_title'].split()]
        data_point.append(words_in_common(search_stemmed, title_stemmed))
        data_point.append(words_in_common(search_stemmed, descr_dict[pd_id]))
        data_point.append(words_in_common(title_stemmed, search_stemmed))
        data_point.append(words_in_common(descr_dict[pd_id], search_stemmed))
        data_point.append(edit_dist_less_two(search_stemmed, title_stemmed))
        X.append(data_point)
    X = numpy.array(X)
    return X

raw_train = pandas.read_csv("/Users/patrickhess/Documents/kaggle/home_depot/train.csv", encoding="ISO-8859-1")
descriptions_raw = pandas.read_csv("/Users/patrickhess/Documents/kaggle/home_depot/product_descriptions.csv", encoding="ISO-8859-1")

descr_dict = {}
descr_len = {}
# Stem descriptions
for it, row in descriptions_raw.iterrows():
    descr_dict[row['product_uid']] = [stemmer.stem(word) for word in row['product_description'].split()]

X = generate_features(raw_train)

Y = []
for it, row in raw_train.iterrows():
    Y.append(row['relevance'])

Y = numpy.array(Y)

def build_model(input_dim, hn=64, dp=0.5, layers=1):
    model = Sequential()
    model.add(Dense(input_dim=input_dim, output_dim=hn, init='glorot_uniform'))
    model.add(PReLU(input_shape=(hn,)))
    model.add(Dropout(dp))
    for i in range(layers):
        model.add(Dense(input_dim=hn, output_dim=hn, init='glorot_uniform'))
        model.add(PReLU(input_shape=(hn,)))
        model.add(BatchNormalization(input_shape=(hn,)))
        model.add(Dropout(dp))
    model.add(Dense(input_dim=hn, output_dim=1))
    model.compile(loss='mse', optimizer='sgd')
    return model

model = build_model(len(X[0]), 48, 0.25, 2)
model.fit(X, Y, nb_epoch=24, batch_size=128, verbose=2)

Ypred = model.predict(X)
math.sqrt(mean_squared_error(Y, Ypred))

raw_test = pandas.read_csv("/Users/patrickhess/Documents/kaggle/home_depot/test.csv", encoding="ISO-8859-1")
Xtest = generate_features(raw_test)
Ypred = [y[0] for y in model.predict(Xtest)]
id_test = raw_test['id']
pandas.DataFrame({"id": id_test, "relevance": Ypred}).to_csv('submission.csv',index=False)