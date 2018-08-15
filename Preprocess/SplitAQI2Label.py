# Created by Yuexiong Ding
# Date: 2018/8/10
# Description: split AQI to 2 label

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 将AQI分为两类,分割点为中值
# raw_data = pd.read_csv('../DataSet/ProcessedData/TrainData/new_train_2016.csv')
# raw_data['AQI Label'] = 0
# raw_data['AQI Label'][raw_data['AQI'] > raw_data['AQI'].median()] = 1
# raw_data.to_csv('../DataSet/ProcessedData/TrainData/new_train_2016.csv', index=False)
# print(raw_data['AQI Label'])

# 将AQI分为两类,分割点为2/3位点
# raw_data = pd.read_csv('../DataSet/ProcessedData/TrainData/new_train_fill_with_state_mean_2016.csv')
# # aqi = list(raw_data['AQI'])
# # aqi_sorted = sorted(aqi)
# # print(aqi_sorted[int((2 * len(aqi_sorted)) / 3)])
# print(len(raw_data[raw_data['AQI'] <= 40]))
# print(len(raw_data[raw_data['AQI'] > 40]))
#
# raw_data['AQI Label'] = 0
# raw_data['AQI Label'][raw_data['AQI'] > 40] = 1
# print(len(raw_data[raw_data['AQI Label'] == 0]))
# print(len(raw_data[raw_data['AQI Label'] == 1]))
# raw_data.to_csv('../DataSet/ProcessedData/TrainData/new_train_fill_with_state_mean_2016.csv', index=False)

# 将AQI分为两类,分割点为2倍标准差位点
raw_data = pd.read_csv('../DataSet/ProcessedData/TrainData/new_train_fill_with_state_mean_2016.csv')
# plt.scatter(range(len(raw_data['AQI'])), raw_data['AQI'])
# raw_data['AQI'][raw_data['AQI'] > 70] = 70
plt.hist(raw_data['AQI'])
plt.show()
std = raw_data['AQI'].std()
print(std)
# # aqi = list(raw_data['AQI'])
# # aqi_sorted = sorted(aqi)
# # print(aqi_sorted[int((2 * len(aqi_sorted)) / 3)])
# print(len(raw_data[raw_data['AQI'] <= 42]))
# print(len(raw_data[raw_data['AQI'] > 42]))
#
raw_data['AQI Label'] = 0
raw_data['AQI Label'][raw_data['AQI'] > 3 * std] = 1
print(len(raw_data[raw_data['AQI Label'] == 0]))
print(len(raw_data[raw_data['AQI Label'] == 1]))
# raw_data.to_csv('../DataSet/ProcessedData/TrainData/new_train_fill_with_state_mean_2016.csv', index=False)