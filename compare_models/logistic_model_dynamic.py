# -*- coding: utf-8 -*- 
import MySQLdb
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold

'''确定当前时间，距离2016年3月份的时间'''
interval = 6
dynamic_labes = pd.read_csv('dynamic_labels.csv')
fw = open('scores.csv', 'w')
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
	#获得序列的统计信息
	series_stat = []
	for i in xrange(0, len(series[0])):
		series_stat.append([])
	for i in xrange(0, len(series)):#feature数、sample数、序列点的个数
		for j in xrange(0, len(series[i])):
			series_stat[j].append(np.mean(series[i][j][0:(27-rounds)]))
			series_stat[j].append(np.var(series[i][j][0:(27-rounds)]))
			series_stat[j].append(np.percentile(series[i][j][0:(27-rounds)], 25))
			series_stat[j].append(np.percentile(series[i][j][0:(27-rounds)], 50))
			series_stat[j].append(np.percentile(series[i][j][0:(27-rounds)], 75))

	series_stat = min_max_scaler.fit_transform(series_stat)
	#换成动态label查看效果
	labels = dynamic_labes[dynamic_labes.columns[rounds]].values
	labels = labels.reshape(len(labels),1)

	#样本个数
	N_Samples = 3050

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
		x_train_series = series_stat[train]
		x_test_series = series_stat[test]

		x_train = np.hstack((x_train_base, x_train_series))
		x_test = np.hstack((x_test_base, x_test_series)) 

		#标签
		y_train = labels[train]
		y_test = labels[test]

		#名称
		names_train = names[train]
		name_test = names[test]

		#建立Logistic分类模型
		clf = LogisticRegression()
		clf.fit(x_train, y_train)
		#计算auc
		y_scores = clf.predict_proba(x_test)[:, 1]
		y_true = y_test
		auc = roc_auc_score(y_true, y_scores)
		auc_list.append(auc)
		#计算accuracy
		accuracy = clf.score(x_test, y_true)
		accuracy_list.append(accuracy)

		name_list+=list(names[test])
		score_list+=list(y_scores)
		label_list+=list(y_true[:,0])
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