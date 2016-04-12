import csv
import gzip
###
import numpy as np
###
from sklearn.linear_model import LinearRegression
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
import gensim
stopwords = set([word for word in stopwords.words('english')])
stemmer = SnowballStemmer('english')
wordToVec = gensim.models.Word2Vec.load_word2vec_format('/Users/patrickhess/Downloads/freebase-vectors-skipgram1000-en.bin', binary=True)

word_set = set([stemmer.stem(word) for word in words.words()])
documents_its_in = {}
num_documents = 0
descr_dict = {}

def words_in_common(words1, words2):
    sum = 0
    dct = {}
    for word1 in words1:
        for word2 in words2:
            if word2.find(word1) >= 0:
                if word1 in documents_its_in:
                    if word1 in dct:
                        dct[word1] += 1
                    else:
                        dct[word1] = 1
                else:
                    sum += 1
    for word1 in dct:
        sum += 1 * (math.log(num_documents / documents_its_in[word1]))
    return sum

def word_to_vec_score(words1, words2):
    sum = 0
    for word1 in words1:
        for word2 in words2:
            mod1 = '/en/' + word1
            mod2 = '/en/' + word2
            if word2.find(word1) == 0 and mod1 in wordToVec.vocab and mod2 in wordToVec.vocab and wordToVec.similarity(mod1, mod2) > 0:
                if word1 in documents_its_in:
                    sum += wordToVec.similarity(mod1, mod2)**2 * (math.log(num_documents / documents_its_in[word1]))
                else:
                    sum += wordToVec.similarity(mod1, mod2)**2
    return sum

def edit_dist_less_two(words1, words2):
    sum = 0
    for word1 in words1:
        for word2 in words2:
            if len(word1) > 5 and len(word2) > 5 and editdistance.eval(word1, word2) <= 2 and word1[0] == word2[0]:
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
            sm += math.log(1 + dct[word])
    return sm


def generate_features(raw_train):
    X = []
    # Create features
    nums_to_words = {}
    for it, row in raw_train.iterrows():
        data_point = []
        pd_id = row['product_uid']
        data_point.append(len(row['search_term']))
        data_point.append(len(row['product_title']))
        search_stemmed = [''.join(e for e in stemmer.stem(word) if e.isalnum()) for word in row['search_term'].split() if stemmer.stem(word)]
        search_stemmed = ['in.' if w == 'inch' else w for w in search_stemmed]
        search_stemmed = ['ft.' if w == 'feet' else w for w in search_stemmed]
        #search_stemmed = ['lb.' if w == 'pound' else w for w in search_stemmed]
        title_stemmed = [''.join(e for e in stemmer.stem(word) if e.isalnum()) for word in row['product_title'].split() if stemmer.stem(word)]
        title_stemmed = ['in.' if w == 'inch' else w for w in title_stemmed]
        title_stemmed = ['ft.' if w == 'feet' else w for w in title_stemmed]
        title_before_with = []
        for w in title_stemmed:
            title_before_with.append(w)
            if w == 'with':
                break
        #title_stemmed = ['lb.' if w == 'pound' else w for w in title_stemmed]
        data_point.append(word_to_vec_score(search_stemmed, title_stemmed) / len(search_stemmed))
        data_point.append(words_in_common(search_stemmed, title_before_with) - len(search_stemmed))
        data_point.append(words_in_common(search_stemmed, descr_dict[pd_id]) - len(search_stemmed))
        #data_point.append(words_in_common(search_stemmed, title_before_with) / len(search_stemmed))
        data_point.append(words_in_common(search_stemmed, title_stemmed) / len(search_stemmed))
        data_point.append(words_in_common(search_stemmed, descr_dict[pd_id]) / len(search_stemmed))
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



raw_train = pandas.read_csv("/Users/patrickhess/Documents/kaggle/home_depot/train.csv", encoding="ISO-8859-1")
descriptions_raw = pandas.read_csv("/Users/patrickhess/Documents/kaggle/home_depot/product_descriptions.csv", encoding="ISO-8859-1")
raw_test = pandas.read_csv("/Users/patrickhess/Documents/kaggle/home_depot/test.csv", encoding="ISO-8859-1")

# Stem descriptions
for it, row in descriptions_raw.iterrows():
    words = [''.join(e for e in stemmer.stem(word) if e.isalnum()) for word in row['product_description'].split()]
    words = ['in.' if w == 'inch' else w for w in words]
    words = ['ft.' if w == 'feet' else w for w in words]
    descr_dict[row['product_uid']] = words
    x = set()
    for w in words:
        if w not in x:
            if w in documents_its_in:
                documents_its_in[w] += 1
            else:
                documents_its_in[w] = 1
        x.add(w)
    num_documents += 1

# Stem descriptions
for it, row in descriptions_raw.iterrows():
    descr_dict[row['product_uid']] = [''.join(e for e in stemmer.stem(word) if e.isalnum()) for word in row['product_description'].split()]

X = generate_features(raw_train)
Xtest = generate_features(raw_test)

Y = []
for it, row in raw_train.iterrows():
    Y.append(row['relevance'])

Y = numpy.array(Y)

def build_model(input_dim, hn=64, dp=0.5, layers=1, init='he_uniform'):
    model = Sequential()
    model.add(Dense(input_dim=input_dim, output_dim=hn, init=init))
    model.add(PReLU(input_shape=(hn,)))
    model.add(Dropout(dp))
    for i in range(layers):
        model.add(Dense(input_dim=hn, output_dim=hn, init=init))
        model.add(PReLU(input_shape=(hn,)))
        model.add(BatchNormalization(input_shape=(hn,)))
        model.add(Dropout(dp))
    model.add(Dense(input_dim=hn, output_dim=1))
    model.compile(loss='mse', optimizer='adam')
    return model


model1 = build_model(len(X[0]), 64, 0.35, 2)
model1.fit(X, Y, nb_epoch=32, batch_size=84, verbose=2)

model2 = build_model(len(X[0]), 64, 0.35, 2, 'glorot_normal')
model2.fit(X, Y, nb_epoch=32, batch_size=84, verbose=2)
rfc1 = RandomForestRegressor(n_estimators=1200, min_samples_split=20, max_depth=50)
rfc1.fit(X, Y)
param_grid = {
    'n_estimators': [1200],
    "min_samples_split": [20],
    'max_depth': [50]
}

CV_rfc = GridSearchCV(estimator=rfc1, param_grid=param_grid, cv= 5, n_jobs=4)

CV_rfc.fit(X, Y)




# Classifier
model2 = xgb.XGBRegressor()
params = {
        'n_estimators': [250, 500, 750],
        'learning_rate': [0.03, 0.04, 0.05],
        'max_depth': [8, 10, 11, 12],
         'subsample': [0.8, 0.93],
        'colsample_bylevel': [0.65, 0.75, 0.85],
        'colsample_bytree': [0.6, 0.7, 0.8]
}

#best_params = {'colsample_bytree': 0.7, 'colsample_bylevel': 0.75, 'learning_rate': 0.04, 'n_estimators': 500, 'subsample': 0.93, 'max_depth': 10}
clf1 = GridSearchCV(model2, params, verbose=1, n_jobs = 4, cv=7)
clf1.fit(X, Y)
print(clf1.best_score_)
print(clf1.best_params_)


model2 = xgb.XGBRegressor()
params = {
        'n_estimators': [700, 800, 900],
        'learning_rate': [0.04],
        'max_depth': [9],
         'subsample': [0.9],
        'colsample_bylevel': [0.75],
        'colsample_bytree': [0.7]
}
clf2 = GridSearchCV(model2, params, verbose=1, n_jobs = 4, cv=7)
clf2.fit(X, Y)
print(clf2.best_score_)
print(clf2.best_params_)

"""
# Classifier
model2 = xgb.XGBRegressor()
params = {
        'n_estimators': [750, 800, 850],
        'learning_rate': [0.05],
        'max_depth': [8],
         'subsample': [0.75, 0.8],
        'colsample_bylevel': [0.75, 0.8],
        'colsample_bytree': [0.7]
}
clf2 = GridSearchCV(model2, params, verbose=1, n_jobs = 4, cv=7)
clf2.fit(X, Y)
print(clf2.best_score_)
print(clf2.best_params_)



# Classifier
model2 = xgb.XGBRegressor()
params = {
           'n_estimators': [350, 400, 450],
        'learning_rate': [0.04, 0.05],
        'max_depth': [10],
         'subsample': [0.85, 0.9],
        'colsample_bylevel': [0.8, 0.85],
        'colsample_bytree': [0.65, 0.7]
}
clf3 = GridSearchCV(model2, params, verbose=1, n_jobs=4, cv=7)
clf3.fit(X, Y)
print(clf3.best_score_)
print(clf3.best_params_)
"""

Xblended = np.column_stack((clf1.predict(X), clf2.predict(X), rfc1.predict(X)))

rfc2 = RandomForestRegressor(n_estimators=100)
rfc2.fit(Xblended, Y)
linModel = LinearRegression()
linModel.fit(Xblended, Y)
num_boost_round = 400
dtrain = xgb.DMatrix(X, Y)
best_params = {'colsample_bytree': 0.78, 'colsample_bylevel': 0.9, 'learning_rate': 0.05, 'min_child_weight': 3, 'n_estimators': 136, 'subsample': 0.88, 'max_depth': 9}
#clf1 = xgb.cv(params=params, dtrain=dtrain, num_boost_round=num_boost_round, early_stopping_rounds=10, nfold=5)
clf1 = xgb.train(params=best_params, dtrain=dtrain, num_boost_round=num_boost_round)
#clf1.predict(xgb.DMatrix(X))
math.sqrt(mean_squared_error(Y[60000:], rfc1.predict(X[60000:])))


Xblended = np.column_stack((clf1.predict(Xtest), clf2.predict(Xtest), rfc1.predict(Xtest)))
Ypred = linModel.predict(Xblended)
Ypred2 = rfc2.predict(Xblended)
#Xoutput2 = np.column_stack((model1.predict(Xtest), clf1.predict(Xtest), clf2.predict(Xtest)))
#Ypred = model2.predict(Xoutput2)
Ypred = [0.3 * max(1, min(3, y)) for y in clf1.predict(Xtest)]
Ypred = numpy.add(Ypred, [0.1 * max(1, min(3, y)) for y in clf2.predict(Xtest)])
#Ypred = numpy.add(Ypred, [0.2 * y for y in clf3.predict(Xtest)])
Ypred = numpy.add(Ypred, [0.4 * y for y in rfc1.predict(Xtest)])
Ypred = numpy.add(Ypred, [0.1 * y[0] for y in model1.predict(Xtest)])
Ypred = numpy.add(Ypred, [0.1 * y[0] for y in model2.predict(Xtest)])
#Ypred = numpy.add(Ypred, [0.1 * y[0] for y in model2.predict(Xtest)])
id_test = raw_test['id']
pandas.DataFrame({"id": id_test, "relevance": Ypred}).to_csv('submission.csv',index=False)


# Code for generating outliers
diff = []
Ypredtrain = clf1.predict(X)
for i in  range(len(Ypredtrain)):
    diff.append(abs(Ypredtrain[i] - Y[i]))
indexes = [i[0] for i in sorted(enumerate(diff), key=lambda x:x[1])]
worst = indexes[len(indexes) - 500:]
worst_high = Y[worst] > 2.5
worst_high_indexes = []
for i, x in enumerate(worst):
    if worst_high[i]:
        worst_high_indexes.append(x)


Xcleaned  = numpy.delete(X, worst_high_indexes, 0)
Ycleaned  = numpy.delete(Y, worst_high_indexes, 0)