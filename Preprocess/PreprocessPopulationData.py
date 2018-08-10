# Created by Yuexiong Ding
# Date: 2018/8/7
# Description: preprocessing population data

import pandas as pd
import numpy as np

# 人口出行方式特征
# 选取一下字段
# HC01_EST_VC01, Workers 16 years and over
# HC01_EST_VC03, Car truck or van
# HC01_EST_VC04, Car truck or van-Drove alone
# HC01_EST_VC05, Car truck or van-Carpooled
# HC01_EST_VC10, Public transportation (excluding taxicab)
# HC01_EST_VC11, Walked
# HC01_EST_VC12, Bicycle
# HC01_EST_VC13, Taxicab motorcycle or other means
# HC01_EST_VC14, Worked at home
# HC01_EST_VC59, No vehicle available
# HC01_EST_VC60, 1 vehicle available
# HC01_EST_VC61, 2 vehicles available
# HC01_EST_VC62, 3 or more vehicles available
raw_pop_transp_data = pd.read_csv('../DataSet/RawData/Population/2016/ACS_16_1YR_S0801_with_ann.csv', low_memory=False,
                                 dtype=str)
raw_pop_transp_data = raw_pop_transp_data.drop([0])
# print(raw_pop_transp_data)
new_pop_transp_data = pd.DataFrame({
    'State County Code': raw_pop_transp_data['GEO.id2'],
    'Workers 16 years and over': raw_pop_transp_data['HC01_EST_VC01'],
    'Car truck or van': raw_pop_transp_data['HC01_EST_VC03'],
    'Car truck or van-Drove alone': raw_pop_transp_data['HC01_EST_VC04'],
    'Car truck or van-Carpooled': raw_pop_transp_data['HC01_EST_VC05'],
    'Public transportation (excluding taxicab)': raw_pop_transp_data['HC01_EST_VC10'],
    'Walked': raw_pop_transp_data['HC01_EST_VC11'],
    'Bicycle': raw_pop_transp_data['HC01_EST_VC12'],
    'Taxicab motorcycle or other means': raw_pop_transp_data['HC01_EST_VC13'],
    'Worked at home': raw_pop_transp_data['HC01_EST_VC14'],
    'No vehicle available': raw_pop_transp_data['HC01_EST_VC59'],
    '1 vehicle available': raw_pop_transp_data['HC01_EST_VC60'],
    '2 vehicles available': raw_pop_transp_data['HC01_EST_VC61'],
    '3 or more vehicles available': raw_pop_transp_data['HC01_EST_VC62'],
})
new_pop_transp_data.to_csv('../DataSet/ProcessedData/Population/pop_transp_data_2016.csv', index=False)
print(new_pop_transp_data)
