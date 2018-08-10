# Created by Yuexiong Ding
# Date: 2018/8/10
# Description: clustering the economic data

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

eco_data = pd.read_csv('../DataSet/ProcessedData/FinalData/eco_2016.csv')
eco_data.pop('State County Code')
county_name = eco_data.pop('County Name')
eco_data.pop('Region')
eco_data.pop('Year')
eco_data.pop('State and local')
eco_data.pop('State government')
eco_data.pop('Local government')
# eco_data = eco_data.iloc[:, :20]

# 均值填充，标准化
# X = eco_data.fillna(eco_data.mean())
X = eco_data.fillna(0)
X = StandardScaler().fit_transform(X)

# 密度聚类
db = DBSCAN(eps=0.04, min_samples=5)
db.fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
n_clusters_ = len(set(labels))-(1 if -1 in labels else 0)

print(db.core_sample_indices_)
print(core_samples_mask)
print(n_clusters_)
print(set(labels))
print(len(county_name))
j = 0
for i in range(len(labels)):
    if labels[i] >= 0:
        j += 1
        print(i, county_name[i], labels[i])

print(j)
# plt.scatter(X[:, 0], X[:, 1])
# plt.show()
