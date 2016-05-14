# -*- coding: utf-8 -*- 
import pandas as pd
import MySQLdb

conn = MySQLdb.connect(host="localhost",user="root",passwd="123",db="ppd",charset="utf8")
sql = "select * from platform_features"
platform_features = pd.read_sql(sql,conn,index_col="index")

#得到平台的名称与标记
platform = list(platform_features['platName'].values)
labels = list(platform_features['label'].values)

#跑路平台数据
bad_platforms = pd.read_csv('bad_platforms.txt', sep='\t')
bad_platforms_time = bad_platforms[['name', 'run_time']]

#回到三月底，首先都更新一遍
all_bad =  list(bad_platforms_time[bad_platforms_time.run_time!=2016.04]['name'].values)
all_bad = [all_bad[i].decode('utf-8') for i in xrange(0, len(all_bad))]
for i in xrange(0, len(platform)):
	if platform[i] in all_bad and labels[i]==0:
		labels[i] = 1

#我们现在在3月底，2月底，1月底，……
rslt_labels = []
months = [2016.04, 2016.03, 2016.02, 2016.01, 2015.12, 2015.11]
for i in xrange(0, len(months)):
	print months[i]
	up_list = list(bad_platforms_time[bad_platforms_time.run_time==months[i]]['name'].values)
	up_list = [up_list[j].decode('utf-8') for j in xrange(0, len(up_list))]
	for j in xrange(0, len(platform)):
		if platform[j] in up_list and labels[j]==1:
			labels[j] = 0
	tmp = labels[:]
	rslt_labels.append(tmp)

df_rslt = pd.DataFrame({'01':rslt_labels[0],
	'02':rslt_labels[1],'03':rslt_labels[2],
	'04':rslt_labels[3],'05':rslt_labels[4],
	'06':rslt_labels[5],
	})
df_rslt.to_csv('dynamic_labels.csv', index=False)


