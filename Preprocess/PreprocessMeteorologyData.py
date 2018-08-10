# Created by Yuexiong Ding
# Date: 2018/8/7
# Description: preprocessing meteorology data

import pandas as pd
import numpy as np
classes = ['TEMP', 'WIND', 'PRESS', 'RH_DP']
year = ['2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014',
        '2015', '2016']

# 用每日气象数据生成年均气象数据
# for c in classes:
#     for y in year:
#         raw_data = pd.read_csv('../DataSet/RawData/Meteorology/' + c + '/daily_' + c + '_' + y + '.csv',
#                                low_memory=False, dtype=str)
#         if c == 'WIND':
#             raw_data = raw_data[raw_data['Parameter Name'] == 'Wind Speed - Resultant']
#
#         raw_data['State County Code'] = raw_data['State Code'] + raw_data['County Code']
#         raw_data['Arithmetic Mean'] = np.array(raw_data['Arithmetic Mean']).astype(float)
#
#         df_groupby = raw_data.groupby('State County Code')
#         df_groupby_index = df_groupby.count().index
#         df_groupby_mean = df_groupby['Arithmetic Mean'].mean()
#         new_data_df = pd.DataFrame(
#             {'State County Code': np.array(df_groupby_index), 'Year': [y] * len(df_groupby_index),
#              c: np.array(df_groupby_mean), })
#         new_data_df.to_csv('../DataSet/ProcessedData/Meteorology/' + c + '/' + y + '.csv', index=False)
#         print(c, y)


# 将四种年均气象数据融合成一份
temp_data = pd.read_csv('../DataSet/ProcessedData/Meteorology/TEMP/2016.csv', low_memory=False, dtype=str)
press_data = pd.read_csv('../DataSet/ProcessedData/Meteorology/PRESS/2016.csv', low_memory=False, dtype=str)
wind_data = pd.read_csv('../DataSet/ProcessedData/Meteorology/WIND/2016.csv', low_memory=False, dtype=str)
ph_dp_data = pd.read_csv('../DataSet/ProcessedData/Meteorology/RH_DP/2016.csv', low_memory=False, dtype=str)
# print(ph_dp_data)

temp_press_inner_data = pd.merge(temp_data, press_data, on='State County Code', how='inner')
temp_press_outer_data = pd.merge(temp_data, press_data, on='State County Code', how='outer')
wind_ph_dp_inner_data = pd.merge(wind_data, ph_dp_data, on='State County Code', how='inner')
wind_ph_dp_outer_data = pd.merge(wind_data, ph_dp_data, on='State County Code', how='outer')
temp_wind_inner_data = pd.merge(temp_data, wind_data, on='State County Code', how='inner')
temp_wind_outer_data = pd.merge(temp_data, wind_data, on='State County Code', how='outer')
total_inner_data = pd.merge(temp_press_inner_data, wind_ph_dp_inner_data, on='State County Code', how='inner')
total_outer_data = pd.merge(temp_press_outer_data, wind_ph_dp_outer_data, on='State County Code', how='outer')
# print(total_outer_data)

temp_wind_inner_data.pop('Year_x')
temp_wind_inner_data.pop('Year_y')
temp_wind_outer_data.pop('Year_x')
temp_wind_outer_data.pop('Year_y')
temp_wind_inner_data['Year'] = '2016'
temp_wind_outer_data['Year'] = '2016'

total_inner_data.pop('Year_x_x')
total_inner_data.pop('Year_y_x')
total_inner_data.pop('Year_x_y')
total_inner_data.pop('Year_y_y')
total_inner_data['Year'] = '2016'

total_outer_data.pop('Year_x_x')
total_outer_data.pop('Year_y_x')
total_outer_data.pop('Year_x_y')
total_outer_data.pop('Year_y_y')
total_outer_data['Year'] = '2016'
#
# total_inner_data.to_csv('../DataSet/ProcessedData/Meteorology/total_inner_data_2016.csv', index=False)
# total_outer_data.to_csv('../DataSet/ProcessedData/Meteorology/total_outer_data_2016.csv', index=False)
temp_wind_inner_data.to_csv('../DataSet/ProcessedData/Meteorology/temp_wind_inner_data_2016.csv', index=False)
temp_wind_outer_data.to_csv('../DataSet/ProcessedData/Meteorology/temp_wind_outer_data_2016.csv', index=False)
print(temp_wind_inner_data)
print(temp_wind_outer_data)
