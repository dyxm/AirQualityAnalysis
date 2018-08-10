# Created by Yuexiong Ding
# Date: 2018/8/10
# Description: 
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

raw_data = pd.read_csv('../DataSet/ProcessedData/TrainData/train_2016.csv', dtype=str)
raw_data.pop('Internet publishing and broadcasting 8/')
raw_data['AQI'] = np.array(raw_data['AQI']).astype(float)
raw_data['State County Code'] = [('0' + x if len(x) == 4 else x) for x in raw_data['State County Code']]
raw_data['AQI Label'] = 0
raw_data['AQI Label'][raw_data['AQI'] > 50] = 1
print(len(raw_data[raw_data['AQI'] <= 50]))
print(len(raw_data[raw_data['AQI'] > 50]))
high_aqi_data = raw_data[raw_data['AQI'] > 50]
meddle_aqi_data = raw_data[(raw_data['AQI'] > 20) & (raw_data['AQI'] <= 50)]
low_aqi_data = raw_data[raw_data['AQI'] <= 20]
print(len(low_aqi_data))
for i in range(0):
    raw_data = raw_data.append(high_aqi_data)
    if i % 2 == 0:
        raw_data = raw_data.append(low_aqi_data)
    # if i % 5 == 0:
        # raw_data = raw_data.append(meddle_aqi_data)

print(len(raw_data[raw_data['AQI'] <= 50]))
print(len(raw_data[raw_data['AQI'] > 50]))
print(len(raw_data[raw_data['AQI'] <= 20]))

print(len(raw_data[raw_data['AQI Label'] == 0]))
print(len(raw_data[raw_data['AQI Label'] == 1]))

raw_data = shuffle(raw_data)
raw_data.to_csv('../DataSet/ProcessedData/TrainData/train_oversampling_2_label_of_aqi_2016.csv', index=False)
# print(raw_data['AQI Label'])
