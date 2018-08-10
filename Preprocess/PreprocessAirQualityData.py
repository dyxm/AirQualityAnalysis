# Created by Yuexiong Ding
# Date: 2018/8/7
# Description: preprocessing air quality data

import pandas as pd
import numpy as np

raw_data = pd.read_csv('../DataSet/RawData/AirQuality/2016/daily_aqi_by_county_2016.csv', low_memory=False, dtype=str)
raw_data['State County Code'] = raw_data['State Code'] + raw_data['County Code']
raw_data['AQI'] = np.array(raw_data['AQI']).astype(float)
# print(raw_data['AQI'])

df_groupby = raw_data.groupby('State County Code')
df_groupby_index = df_groupby.count().index
df_groupby_mean = df_groupby['AQI'].mean()
# print(df_groupby_mean)
new_data_df = pd.DataFrame({'State County Code': np.array(df_groupby_index), 'Year': ['2016'] * len(df_groupby_index),
                            'AQI': np.array(df_groupby_mean)})
print(new_data_df)
new_data_df.to_csv('../DataSet/ProcessedData/AirQuality/annual_aqi_by_county_2016.csv', index=False)
