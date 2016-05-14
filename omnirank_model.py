# -*- coding: utf-8 -*- 
import MySQLdb
import pandas as pd
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Embedding, LSTM, Dense, merge
from keras.layers import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
import numpy as np
from keras.utils.visualize_util import plot
from sklearn.metrics import roc_auc_score

#读取p2p平台各个特征的数据
conn = MySQLdb.connect(host="localhost",user="root",passwd="123",db="ppd",charset="utf8")
sql = "select * from platform_features"
platform_features = pd.read_sql(sql,conn,index_col="index")
names = platform_features['platName'].values
base = platform_features[['type', 'tzzj_cooperation', 'listed', 'vc', 'argue', 'third_party', 'join_society', 'score', 'averageProfit', 'registMoney', 'autobid', 'stockTransfer', 'fundsToken', 'ifGuarantee', 'ifGuaranteeOrg', 'lauchTime', 'category', 'lng', 'lat']].values
for i in xrange(0, len(base)):
	for j in xrange(0, len(base[i])):
		if isinstance(base[i][j], unicode):
			if base[i][j]=='':
				base[i][j] = 0
			else:
				base[i][j] = float(base[i][j])

series_old = [platform_features['comment_pos'].values, 
platform_features['comment_neu'].values, 
platform_features['comment_neg'].values]
series = []
for i in xrange(0, len(series_old)):
	#处理每个series
	series.append([])
	for j in xrange(0, len(series_old[i])):
		tmp = series_old[i][j].split(',')
		tmp = [int(tmp[k]) for k in xrange(0, len(tmp))]
		series[i].append(tmp)
	series[i] = np.array(series[i])
labels = platform_features['label'].values
labels = labels.reshape(len(labels),1)


#样本个数
N_train = 2000
N_test = 1000

#基本型变量
base_dim = 19
# x_train_base = np.random.random((N_train, base_dim))
# x_test_base = np.random.random((N_test, base_dim))
x_train_base = base[0:N_train]
x_test_base = base[N_train:(N_train+N_test)]

#序列型变量
series_dims = [27, 27, 27]
# x_train_series = [np.random.random((N_train, series_dims[i])) for i in xrange(0, len(series_dims))]
# x_test_series = [np.random.random((N_test, series_dims[i])) for i in xrange(0, len(series_dims))]
x_train_series = [series[i][0:N_train] for i in xrange(0, len(series_dims))]
x_test_series = [series[i][N_train:(N_train+N_test)] for i in xrange(0, len(series_dims))]
#标签
# y_train = np.random.randint(2, size=(N_train, 1))
# y_test = np.random.randint(2, size=(N_test, 1))
y_train = labels[0:N_train]
y_test = labels[N_train:(N_train+N_test)]

#名称
names_train = names[0:N_train]
name_test = names[N_train:(N_train+N_test)]

#建立深度学习模型
#基本特征
base_input = Input(shape=(base_dim,), name='base_input')
x = Dense(64, input_dim=base_dim, init='uniform', activation='relu')(base_input)
base_model = Dropout(0.5)(x)
#序列特征
series_inputs, series_models = [], []
for i in xrange(0, len(series_dims)):
	series_input = Input(shape=(series_dims[i],), dtype='int32', name='series_input' + str(i))
	series_inputs.append(series_input)
	x = Embedding(output_dim=128, input_dim=10000, input_length=series_dims[i])(series_input)
	lstm_out = LSTM(32)(x)
	series_models.append(lstm_out)
x = merge([base_model] + series_models, mode='concat')
x = Dense(64, activation='relu')(x)
main_loss = Dense(1, activation='sigmoid', name='main_output')(x)

model = Model(input=[base_input] + series_inputs, output=main_loss)
plot(model, to_file='model.png')

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=["accuracy"])

model.fit([x_train_base] + x_train_series, y_train,
           nb_epoch=16, batch_size=32)

a = model.predict([x_test_base] + x_test_series, batch_size=32)
b = model.evaluate([x_test_base] + x_test_series, y_test, batch_size=32)
print b
y_scores, y_true = [], []
for i in xrange(0, len(a)):
	y_scores.append(float(a[i][0]))
	y_true.append(int(y_test[i][0]))
y_scores = np.array(y_scores)
y_true = np.array(y_true)
print roc_auc_score(y_true, y_scores)