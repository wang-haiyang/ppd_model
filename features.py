# -*- coding: utf-8 -*- 
import pandas as pd
import time, datetime
import MySQLdb
from keras.utils import np_utils
import numpy as np
from sklearn import preprocessing

def date_difference(d1, d2):#d1早于d2
	d1 = datetime.datetime.strptime(d1 + ' 00:00:00', '%Y-%m-%d %H:%M:%S')
	d2 = datetime.datetime.strptime(d2 + ' 00:00:00', '%Y-%m-%d %H:%M:%S')
	delta = d2 - d1
	return delta.days

def extr_tags(x):
#type：国资系/上市公司系/银行系/民营系（类别型标签）
#tzzj_cooperation     投之家合作平台
#listed     股权上市
#vc     接受过风投
#argue     争议
#third_party     加入第三方征信
#join_society     加入协会
#label：停业/提现困难/跑路/经侦介入（含有这些关键字就为1，否则就为0）
	types = ['国资系', '上市公司系', '银行系', '民营系']
	binas = ['投之家合作平台', '股权上市', '接受过风投', '争议', '加入第三方征信', '加入协会']
	labels = ['停业', '提现困难', '跑路','经侦介入']
	rslt = [ [0 for i in xrange(0, len(x))] for j in xrange(0, 8)]
	for i in xrange(0, len(x)):
		tag_list = x[i].split(',')
		for j in xrange(0, len(tag_list)):
			tag = tag_list[j].encode('utf-8')
			if tag in types:
				rslt[0][i]=types.index(tag)+1
				continue
			if tag in binas:
				rslt[binas.index(tag)+1][i] = 1
				continue
			if tag in labels:
				rslt[7][i] = 1
	return rslt
			
def one_feature(x, fill, cut):
	x_new = []
	for i in xrange(0, len(x)):
		if x[i]=='':
			x_new.append(0)
		else:
			x_new.append(x[i].strip(cut))
	return x_new

def autobid_feature(x):
	rslt = []
	for i in xrange(0, len(x)):
		if x[i]=='':
			rslt.append(0)
		elif x[i]==u'支持':
			rslt.append(1)
		else:
			rslt.append(-1)
	return rslt

def stockTransfer_feature(x):
	rslt = []
	for i in xrange(0, len(x)):
		if x[i]=='':
			rslt.append(-1)
		elif x[i]==u'随时':
			rslt.append(0)
		elif x[i]==u'1年':
			rslt.append(12)
		elif x[i]==u'不可转让':
			rslt.append(300)
		else:
			rslt.append(x[i].strip(u'个月'))
		print x, 
	return rslt

def fundsToken_feature(x):
	rslt = []
	for i in xrange(0, len(x)):
		if x[i]=='' or x[i]==u'无托管':
			rslt.append(0)
		else:
			rslt.append(1)
	return rslt

def ifGuarantee_feature(x):
	rslt = []
	for i in xrange(0, len(x)):
		if x[i]=='':
			rslt.append(0)
		else:
			rslt.append(1)
	return rslt

def ifGuaranteeOrg_feature(x):
	rslt = []
	for i in xrange(0, len(x)):
		if x[i]=='':
			rslt.append(0)
		else:
			rslt.append(1)
	return rslt

def lauchTime_feature(x):
	rslt = []
	for i in xrange(0, len(x)):
		if x[i]!='':
			rslt.append(date_difference(x[i], '2016-04-17'))
		else:
			rslt.append(0)
	avg = np.mean(rslt)
	for i in xrange(0, len(rslt)):
		if rslt[i]==0:
			rslt[i] = avg
		rslt[i] = rslt[i]/30
	return rslt

def category_feature(x):
	dic = {'':-1, u'股份合作企业':0, u'私营企业':1, u'港、澳、台投资企业':2,
	u'股份制企业':3, u'集体所有制企业':4, u'外商投资企业':5, u'国有企业':6, u'联营企业':7}
	rslt = []
	for i in xrange(0, len(x)):
		rslt.append(dic[x[i]])
	return rslt

conn = MySQLdb.connect(host="localhost",user="root",passwd="123",db="ppd",charset="utf8")


#平台基本信息
sql = "select id,platId,platName,platPin,tags,score,averageProfit,registMoney,\
autobid,stockTransfer,fundsToken,bidGuarantee,guaranteeMode,guaranteeOrg,\
lauchTime,category,lng,lat from platform"
platform = pd.read_sql(sql,conn,index_col="id")
#定义提取特征的DataFrame
platform_features = platform[['platId', 'platName', 'platPin']]

#开始提取特征
#tags特征
tags = list(platform['tags'])
tag_features = ['type', 'tzzj_cooperation', 'listed', 'vc', 'argue' ,'third_party', 'join_society', 'label']
rslt = extr_tags(tags)
for i in xrange(0, len(tag_features)):
	platform_features[tag_features[i]] = pd.Series(rslt[i], index = platform_features.index)
#score
score = list(platform['score'])
platform_features['score'] = pd.Series(one_feature(score,0,''), index = platform_features.index)
#averageProfit
averageProfit = list(platform['averageProfit'])
platform_features['averageProfit'] = pd.Series(one_feature(averageProfit,0,'%'), index = platform_features.index)
#registMoney
registMoney = list(platform['registMoney'])
platform_features['registMoney'] = pd.Series(one_feature(registMoney,0,u' 万元'), index = platform_features.index)
#autobid
autobid = list(platform['autobid'])
platform_features['autobid'] = pd.Series(autobid_feature(autobid), index = platform_features.index)
#stockTransfer
stockTransfer = list(platform['stockTransfer'])
platform_features['stockTransfer'] = pd.Series(stockTransfer_feature(stockTransfer), index = platform_features.index)
#fundsToken
fundsToken = list(platform['fundsToken'])
platform_features['fundsToken'] = pd.Series(fundsToken_feature(fundsToken), index = platform_features.index)
#ifGuarantee
guaranteeMode = list(platform['guaranteeMode'])
platform_features['ifGuarantee'] = pd.Series(ifGuarantee_feature(guaranteeMode), index = platform_features.index)
#ifGuaranteeOrg
guaranteeOrg = list(platform['guaranteeOrg'])
platform_features['ifGuaranteeOrg'] = pd.Series(ifGuaranteeOrg_feature(guaranteeOrg), index = platform_features.index)
#lauchTime
lauchTime = list(platform['lauchTime'])
platform_features['lauchTime'] = pd.Series(lauchTime_feature(lauchTime), index = platform_features.index)
#category
category = list(platform['category'])
platform_features['category'] = pd.Series(category_feature(category), index = platform_features.index)
#lng
lng = list(platform['lng'])
platform_features['lng'] = pd.Series(one_feature(lng,0,''), index = platform_features.index)
#lat
lat = list(platform['lat'])
platform_features['lat'] = pd.Series(one_feature(lat,0,''), index = platform_features.index)

#用户评论信息comment
sql = "select timestamp,platName,attitude from comment"
comment = pd.read_sql(sql,conn)

platName = list(set(list(comment['platName'])))
comment_features = pd.DataFrame({'platName': pd.Series(platName)})
pos,neu,neg = [],[],[]
for i in xrange(0, len(platName)):
	pos.append([0 for k in xrange(0, 12*2+3)])
	neu.append([0 for k in xrange(0, 12*2+3)])
	neg.append([0 for k in xrange(0, 12*2+3)])

	tmp = comment[comment.platName==platName[i]]
	tsp = list(tmp['timestamp'])
	atti = list(tmp['attitude'])
	for j in xrange(0, len(tsp)):
		year = int(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(tsp[j]))).split(' ')[0].split('-')[0])
		month = int(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(tsp[j]))).split(' ')[0].split('-')[1])
		if year in [2014, 2015] or (year==2016 and month in [1, 2, 3]):
			idx = (year-2014)*12+month-1
			if atti[j]==u'推荐':
				pos[i][idx]+=1
			elif atti[j]==u'一般':
				neu[i][idx]+=1
			else:
				neg[i][idx]+=1
	pos[i] = ','.join([str(pos[i][j]) for j in xrange(0, len(pos[i]))])
	neu[i] = ','.join([str(neu[i][j]) for j in xrange(0, len(neu[i]))])
	neg[i] = ','.join([str(neg[i][j]) for j in xrange(0, len(neg[i]))])
comment_features['comment_pos'] = pd.Series(pos, index = comment_features.index)
comment_features['comment_neu'] = pd.Series(neu, index = comment_features.index)
comment_features['comment_neg'] = pd.Series(neg, index = comment_features.index)
platform_features = platform_features.merge(comment_features, how='left', on='platName')

platform_features.fillna(','.join(['0' for i in xrange(0, 12*2+3)]), inplace=True)

#把特征写入数据库
platform_features.to_sql('platform_features1', conn, flavor='mysql', if_exists='replace', index=True)