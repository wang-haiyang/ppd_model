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
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import KFold
import pickle

#确定当前时间，距离2016年3月份的时间
interval = 6
dynamic_labes = pd.read_csv('dynamic_labels.csv')
fw = open('scores.csv', 'a')
for rounds in xrange(0, interval):
	print 'rounds:	' + str(rounds)
	#读取p2p平台各个特征的数据
	conn = MySQLdb.connect(host="localhost",user="root",passwd="123",db="ppd",charset="utf8")
	sql = "select * from platform_features"
	platform_features = pd.read_sql(sql,conn,index_col="index")
	names = platform_features['platName'].values
	base = platform_features[['type', 'tzzj_cooperation', 'listed', 'vc', 'argue', 'third_party', 'join_society', 'score', 'averageProfit', 'registMoney', 'autobid', 'stockTransfer', 'fundsToken', 'ifGuarantee', 'ifGuaranteeOrg', 'lauchTime', 'category', 'lng', 'lat']].values

	#对特征进行预处理
	for i in xrange(0, len(base)):
		for j in xrange(0, len(base[i])):
			if isinstance(base[i][j], unicode):
				if base[i][j]=='':
					base[i][j] = 0
				else:
					base[i][j] = float(base[i][j])
	#更改距离公司成立的时间
	for i in xrange(0, len(base[15])):#base[15]表示lauchTime
		base[15][i]-=rounds

	#base = preprocessing.scale(base)
	min_max_scaler = preprocessing.MinMaxScaler()
	base = min_max_scaler.fit_transform(base)
	for i in xrange(0, len(base)):
		base[i,7]*=60

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
			tmp = tmp[0:len(tmp)-rounds]
			series[i].append(tmp)
		series[i] = np.array(series[i])
	#labels = platform_features['label'].values
	#换成动态label查看效果
	labels = dynamic_labes[dynamic_labes.columns[rounds]].values
	labels = labels.reshape(len(labels),1)

	#将中间变量存入pickle文件
	pkl_file = open('data.pkl', 'wb')
	pickle.dump([names, base, series, labels], pkl_file)
	pkl_file.close()
	#从pickle文件中读入中间变量
	# pkl_file = open('data.pkl', 'rb')
	# [names, base, series, labels] = pickle.load(pkl_file)
	# pkl_file.close()

	#样本个数
	N_Samples = 3050
	base_dim = 19
	series_dims = [27-rounds, 27-rounds, 27-rounds]

	accuracy_list, auc_list = [],[]

	name_list = []
	score_list = []
	label_list = []

	kf = KFold(N_Samples, n_folds=5)
	for train, test in kf:
		#基本型变量
		x_train_base = base[train]
		x_test_base = base[test]

		#序列型变量
		x_train_series = [series[i][train] for i in xrange(0, len(series_dims))]
		x_test_series = [series[i][test] for i in xrange(0, len(series_dims))]

		#标签
		y_train = labels[train]
		y_test = labels[test]

		#名称
		names_train = names[train]
		name_test = names[test]

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
		x = Dense(64, activation='relu')(x)
		x = Dense(64, activation='relu')(x)
		main_loss = Dense(1, activation='sigmoid', name='main_output')(x)

		model = Model(input=[base_input] + series_inputs, output=main_loss)
		plot(model, to_file='model.png')

		model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=["accuracy"])

		model.fit([x_train_base] + x_train_series, y_train,
		           nb_epoch=20, batch_size=32)

		score,accuracy = model.evaluate([x_test_base] + x_test_series, y_test, batch_size=32)
		a = model.predict([x_test_base] + x_test_series, batch_size=32)
		y_scores, y_true = [], []
		for i in xrange(0, len(a)):
			y_scores.append(float(a[i][0]))
			y_true.append(int(y_test[i][0]))
		y_scores = np.array(y_scores)
		y_true = np.array(y_true)
		auc = roc_auc_score(y_true, y_scores)
		accuracy_list.append(accuracy)
		auc_list.append(auc)
		print accuracy, auc

		name_list+=list(names[test])
		score_list+=list(y_scores)
		label_list+=list(y_true)
	print 'average results'
	print accuracy_list
	print auc_list
	print np.mean(accuracy_list), np.mean(auc_list)
	fw.write('month' + str(rounds) + '\n')
	for i in xrange(0, len(accuracy_list)):
		fw.write(str(accuracy_list[i]) + ',')
	fw.write('\n')
	for i in xrange(0, len(auc_list)):
		fw.write(str(auc_list[i]) + ',')
	fw.write('\n')
	fw.write(str(np.mean(accuracy_list)) + ',' + str(np.mean(auc_list)) + '\n')
	score_list = np.array([1-score_list[i] for i in xrange(0, len(score_list))])
	rankings = pd.DataFrame({
		'name':[name_list[i].encode('utf-8') for i in xrange(0, len(name_list))],
		'score':list(score_list),
		'label':label_list
		})
	rankings.sort(columns='score', ascending=False).to_csv('ppd_rankings_' + str(rounds) + '.csv',index=False)
fw.close()