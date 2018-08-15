# Created by Yuexiong Ding
# Date: 2018/8/14
# Description: fill the NaN with state's mean
import pandas as pd
import numpy as np

# 用州均值填充空值
raw_data = pd.read_csv('../DataSet/ProcessedData/TrainData/new_train_2016.csv')
columns = ['PCP', 'TEMP', 'WIND', 'Personal income (thousands of dollars)', 'Population (persons)',
           'Per capita personal income (dollars)','Construction', 'Manufacturing', 'Wholesale trade', 'Retail trade',
           'Information','Professional, scientific, and technical services',
           'Administrative and support and waste management and remediation services',
           'Other services (except government and government enterprises)',
           'Farm earnings Forestry Fishing Mining Quarrying Oil Gas extraction',
           'Finance Insurance Real estate Rental Leasing', 'Arts Entertainment Recreation Accommodation Food services',
           'Workers 16 years and over', 'Car truck or van', 'Car truck or van-Drove alone',
           'Car truck or van-Carpooled', 'Public transportation (excluding taxicab)', 'Walked', 'Bicycle',
           'Taxicab motorcycle or other means', 'Worked at home', 'No vehicle available', '1 vehicle available',
           '2 vehicles available', '3 or more vehicles available']
for c in raw_data.columns:
    j = 0
    for i in raw_data[c].isnull():
        if i:
            j += 1
    percent = (j / len(raw_data[c]))
    print(c, j, percent * 100)
df_group_by = raw_data.groupby('State Code')
for c in columns:
    df_group_by_mean = df_group_by[c].mean()
    temp_array = raw_data[c].isnull()
    for i in range(len(temp_array)):
        if temp_array[i]:
            raw_data[c][i] = df_group_by_mean[raw_data['State Code'][i]]
    print(c)

for c in raw_data.columns:
    j = 0
    for i in raw_data[c].isnull():
        if i:
            j += 1
    percent = (j / len(raw_data[c]))
    print(c, j, percent * 100)

raw_data.to_csv('../DataSet/ProcessedData/TrainData/new_train_fill_with_state_mean_2016.csv', index=False)

raw_data = pd.read_csv('../DataSet/ProcessedData/TrainData/new_train_fill_with_state_mean_2016.csv', dtype=str)
raw_data['State County Code'] = ['0' + x if len(x) < 5 else x for x in raw_data['State County Code']]
raw_data['State Code'] = [x[0: 2] for x in raw_data['State County Code']]
raw_data['County Code'] = [x[2:] for x in raw_data['State County Code']]
raw_data.to_csv('../DataSet/ProcessedData/TrainData/new_train_fill_with_state_mean_2016.csv', index=False)
