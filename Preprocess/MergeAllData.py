# Created by Yuexiong Ding
# Date: 2018/8/7
# Description: merge all data

import pandas as pd
import numpy as np

# ########################## 逐个合并数据 ###################################
# aqi_data = pd.read_csv('../DataSet/ProcessedData/AirQuality/annual_aqi_by_county_2016.csv', low_memory=False, dtype=str)
# geo_data = pd.read_csv('../DataSet/ProcessedData/Geography/county_geography_2016.csv', low_memory=False, dtype=str)
# temp_wind_inner_data = pd.read_csv('../DataSet/ProcessedData/Meteorology/temp_wind_inner_data_2016.csv',
#                                    low_memory=False, dtype=str)
# temp_wind_outer_data = pd.read_csv('../DataSet/ProcessedData/Meteorology/temp_wind_outer_data_2016.csv',
#                                    low_memory=False, dtype=str)
# met_inner_data = pd.read_csv('../DataSet/ProcessedData/Meteorology/total_inner_data_2016.csv', low_memory=False,
#                              dtype=str)
# met_outer_data = pd.read_csv('../DataSet/ProcessedData/Meteorology/total_inner_data_2016.csv', low_memory=False,
#                              dtype=str)
#
# aqi_data.pop('Year')
# geo_data.pop('Year')
# temp_wind_inner_data.pop('Year')
# temp_wind_outer_data.pop('Year')
# met_inner_data.pop('Year')
# met_outer_data.pop('Year')
#
# # 合并空气质量和地理数据
# aqi_geo_inner_data = pd.merge(aqi_data, geo_data, on='State County Code', how='inner')
# aqi_geo_outer_data = pd.merge(aqi_data, geo_data, on='State County Code', how='outer')
# aqi_geo_inner_data.to_csv('../DataSet/ProcessedData/aqi_geo_inner_data_2016.csv', index=False)
# aqi_geo_outer_data.to_csv('../DataSet/ProcessedData/aqi_geo_outer_data_2016.csv', index=False)
# # print(aqi_geo_outer_data)
#
# # 合并空气质量、地理和气象数据
# aqi_geo_met_inner_data = pd.merge(aqi_geo_inner_data, met_inner_data, on='State County Code', how='inner')
# aqi_geo_met_outer_data = pd.merge(aqi_geo_outer_data, met_outer_data, on='State County Code', how='outer')
# aqi_geo_temp_wind_inner_data = pd.merge(aqi_geo_inner_data, temp_wind_inner_data, on='State County Code', how='inner')
# aqi_geo_temp_wind_outer_data = pd.merge(aqi_geo_outer_data, temp_wind_outer_data, on='State County Code', how='outer')
# # aqi_geo_met_inner_data.to_csv('../DataSet/ProcessedData/aqi_geo_met_inner_data_2016.csv', index=False)
# # aqi_geo_met_outer_data.to_csv('../DataSet/ProcessedData/aqi_geo_met_outer_data_2016.csv', index=False)
# # aqi_geo_temp_wind_inner_data.to_csv('../DataSet/ProcessedData/aqi_geo_temp_wind_inner_data_2016.csv', index=False)
# # aqi_geo_temp_wind_outer_data.to_csv('../DataSet/ProcessedData/aqi_geo_temp_wind_outer_data_2016.csv', index=False)
# # print(aqi_geo_temp_wind_inner_data['State County Code'].values)
# # print(aqi_geo_temp_wind_outer_data)
#
# # 合并空气质量、地理、气象、经济数据
# eco_data = pd.read_csv('../DataSet/ProcessedData/Economy/eco_total_data_2016.csv', low_memory=False, dtype=str)
# aqi_geo_met_eco_inner_data = pd.merge(aqi_geo_met_inner_data, eco_data, left_on='State County Code', right_on='GeoFIPS', how='inner')
# aqi_geo_met_eco_outer_data = pd.merge(aqi_geo_met_outer_data, eco_data, left_on='State County Code', right_on='GeoFIPS', how='outer')
# aqi_geo_temp_wind_eco_inner_data = pd.merge(aqi_geo_temp_wind_inner_data, eco_data, left_on='State County Code', right_on='GeoFIPS', how='inner')
# aqi_geo_temp_wind_eco_outer_data = pd.merge(aqi_geo_temp_wind_outer_data, eco_data, left_on='State County Code', right_on='GeoFIPS', how='outer')
# aqi_geo_met_eco_inner_data.to_csv('../DataSet/ProcessedData/aqi_geo_met_eco_inner_data_2016.csv', index=False)
# aqi_geo_met_eco_outer_data.to_csv('../DataSet/ProcessedData/aqi_geo_met_eco_outer_data_2016.csv', index=False)
# aqi_geo_temp_wind_eco_inner_data.to_csv('../DataSet/ProcessedData/aqi_geo_temp_wind_eco_inner_data_2016.csv', index=False)
# aqi_geo_temp_wind_eco_outer_data.to_csv('../DataSet/ProcessedData/aqi_geo_temp_wind_eco_outer_data_2016.csv', index=False)
# print(aqi_geo_temp_wind_eco_inner_data)
# print(aqi_geo_temp_wind_eco_outer_data)

# ###############################################################################


# ########################## 一次完成数据合并 ###################################
aqi_data = pd.read_csv('../DataSet/ProcessedData/FinalData/aqi_2016.csv', low_memory=False, dtype=str)
geo_data = pd.read_csv('../DataSet/ProcessedData/FinalData/geo_2016.csv', low_memory=False, dtype=str)
pcp_data = pd.read_csv('../DataSet/ProcessedData/FinalData/pcp_2016.csv', low_memory=False, dtype=str)
temp_data = pd.read_csv('../DataSet/ProcessedData/FinalData/temp_2016.csv', low_memory=False, dtype=str)
wind_data = pd.read_csv('../DataSet/ProcessedData/FinalData/wind_2016.csv', low_memory=False, dtype=str)
eco_data = pd.read_csv('../DataSet/ProcessedData/FinalData/eco_2016.csv', low_memory=False, dtype=str)
pop_transp_data = pd.read_csv('../DataSet/ProcessedData/FinalData/pop_transp_2016.csv', low_memory=False, dtype=str)

aqi_data.pop('Year')
geo_data.pop('Year')
eco_data.pop('Year')
wind_data.pop('Year')

aqi_geo_data = pd.merge(aqi_data, geo_data, on='State County Code', how='left')
aqi_geo_pcp_data = pd.merge(aqi_geo_data, pcp_data, on='State County Code', how='left')
aqi_geo_pcp_temp_data = pd.merge(aqi_geo_pcp_data, temp_data, on='State County Code', how='left')
aqi_geo_pcp_temp_wind_data = pd.merge(aqi_geo_pcp_temp_data, wind_data, on='State County Code', how='left')
aqi_geo_pcp_temp_wind_eco_data = pd.merge(aqi_geo_pcp_temp_wind_data, eco_data, on='State County Code', how='left')
aqi_geo_pcp_temp_wind_eco_transp_data = pd.merge(aqi_geo_pcp_temp_wind_eco_data, pop_transp_data,
                                                 on='State County Code', how='left')
aqi_geo_pcp_temp_wind_eco_transp_data['State Code'] = [x[0: 2] for x in aqi_geo_pcp_temp_wind_eco_transp_data['State County Code']]
aqi_geo_pcp_temp_wind_eco_transp_data['County Code'] = [x[2:] for x in aqi_geo_pcp_temp_wind_eco_transp_data['State County Code']]
aqi_geo_pcp_temp_wind_eco_transp_data.to_csv('../DataSet/ProcessedData/TrainData/new_train_2016.csv', index=False)
print(aqi_geo_pcp_temp_wind_eco_transp_data)
print(len(aqi_data))
