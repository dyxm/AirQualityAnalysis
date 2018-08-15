# Created by Yuexiong Ding
# Date: 2018/8/7
# Description: preprocessing geography data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

# 处理占地面积、水域面积
# raw_data = pd.read_csv('../DataSet/RawData/Geography/county_geography_2016.txt', low_memory=False, dtype=str,
#                        delimiter="\t")
# new_data_df = pd.DataFrame({'State County Code': raw_data['GEOID'], 'Year': ['2016'] * len(raw_data),
#                             'ALAND': raw_data['ALAND'], 'AWATER': raw_data['AWATER'], 'INTPTLAT': raw_data['INTPTLAT'],
#                             'INTPTLONG': raw_data['INTPTLONG']})
# new_data_df.to_csv('../DataSet/ProcessedData/Geography/county_geography_2016.csv', index=False)
# print(new_data_df)

# 处理道路长度,合并到训练集中
# road_data = pd.read_csv('../DataSet/RawData/Geography/road_length_2016.csv', dtype=str)
# road_data.pop('statefp')
# road_data.pop('countyfp')
# road_data.pop('name')
# train_data = pd.read_csv('../DataSet/ProcessedData/TrainData/train_fill_with_state_mean_2_label_2016.csv', dtype=str)
# # train_data['State County Code'] = ['0' + x if len(x) < 5 else x for x in train_data['State County Code']]
# # train_data.to_csv('../DataSet/ProcessedData/TrainData/train_fill_with_state_mean_2_label_2016.csv', index=False)
# train_add_road_length_data = pd.merge(train_data, road_data, on='State County Code', how='left')
# train_add_road_length_data.to_csv(
#     '../DataSet/ProcessedData/TrainData/train_add_road_data_fill_with_state_mean_2_label_2016.csv', index=False)
# print(train_add_road_length_data)


# 查看道路长度跟AQI的关系
train_data = pd.read_csv('../DataSet/ProcessedData/TrainData/train_add_road_data_fill_with_state_mean_2_label_2016.csv')
columns = ["unclassified", "footway", "motorway", "proposed", "abandoned", "tertiary", "trunk", "residential_link",
           "raceway", "tertiary_link", "motorway_link", "steps", "bridleway", "pedestrian", "conveyor", "turning_loop",
           "secondary_link", "primary_link", "escape", "service", "cycleway", "trunk_link", "bus_guideway",
           "living_street", "path", "residential", "road", "corridor", "rest_area", "disused", "construction",
           "primary", "secondary", "track"]

# final_col = []
# for c in train_data.columns:
#     j = 0
#     for i in train_data[c] == 0:
#         if i:
#             j += 1
#     percent = (j / len(train_data[c]))
#     if percent < 0.1:
#         final_col.append(c)
#     print(c, j, percent * 100)
# print(final_col)
# df_label_0 = train_data[train_data['AQI Label'] == 0]
# df_label_1 = train_data[train_data['AQI Label'] == 1]
# for c in columns:
#     print(c, len(df_label_0[c]), len(df_label_1[c]))
# plt.scatter(train_data['unclassified'], train_data['AQI'])
# 'unclassified', 'tertiary', 'service', 'residential', 'primary', 'secondary', 'track'
# plt.scatter(train_data['tertiary'], train_data['AQI'])
# plt.scatter(train_data['service'], train_data['AQI'])
# plt.scatter(train_data['residential'], train_data['AQI'])
plt.scatter(train_data['living_street'], train_data['AQI'])
# plt.scatter(train_data['secondary'], train_data['AQI'])
# plt.scatter(train_data['track'], train_data['AQI'])
plt.show()
