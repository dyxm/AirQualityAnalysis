# Created by Yuexiong Ding
# Date: 2018/8/10
# Description: 
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

# raw_data = pd.read_csv('../DataSet/ProcessedData/TrainData/train_2016.csv', dtype=str)
# raw_data = pd.read_csv('../DataSet/ProcessedData/TrainData/train_2016.csv')

# raw_data['AQI'] = np.array(raw_data['AQI']).astype(float)
# raw_data['State County Code'] = [('0' + x if len(x) == 4 else x) for x in raw_data['State County Code']]
# raw_data['State Code'] = [x[0: 2] for x in raw_data['State County Code']]
# raw_data['County Code'] = [x[2:] for x in raw_data['State County Code']]
# raw_data['AQI Label'] = 0
# raw_data['AQI Label'][raw_data['AQI'] > 50] = 1
# print(len(raw_data[raw_data['AQI'] <= 50]))
# print(len(raw_data[raw_data['AQI'] > 50]))
# high_aqi_data = raw_data[raw_data['AQI'] > 50]
# meddle_aqi_data = raw_data[(raw_data['AQI'] > 20) & (raw_data['AQI'] <= 50)]
# low_aqi_data = raw_data[raw_data['AQI'] <= 20]
# print(len(low_aqi_data))
# for i in range(0):
#     raw_data = raw_data.append(high_aqi_data)
#     if i % 2 == 0:
#         raw_data = raw_data.append(low_aqi_data)
#     # if i % 5 == 0:
#         # raw_data = raw_data.append(meddle_aqi_data)
#
# print(len(raw_data[raw_data['AQI'] <= 50]))
# print(len(raw_data[raw_data['AQI'] > 50]))
# print(len(raw_data[raw_data['AQI'] <= 20]))
#
# print(len(raw_data[raw_data['AQI Label'] == 0]))
# print(len(raw_data[raw_data['AQI Label'] == 1]))

# raw_data = shuffle(raw_data)
# raw_data.to_csv('../DataSet/ProcessedData/TrainData/train_oversampling_2_label_of_aqi_2016.csv', index=False)
# raw_data.to_csv('../DataSet/ProcessedData/TrainData/train_2016.csv', index=False)
# print(raw_data['AQI Label'])



# raw_data = pd.read_csv('../DataSet/ProcessedData/TrainData/train_2016.csv')
# columns = ['ALAND', 'AWATER', 'INTPTLAT', 'INTPTLONG', 'PCP', 'TEMP', 'Personal income (thousands of dollars)',
#            'Population (persons) 2/', 'Per capita personal income (dollars)', 'Earnings by place of work',
#            'Less: Contributions for government social insurance 3/',
#            'Employee and self-employed contributions for government social insurance',
#            'Employer contributions for government social insurance', 'Plus: Adjustment for residence 4/',
#            'Equals: Net earnings by place of residence', 'Plus: Dividends, interest, and rent 5/',
#            'Plus: Personal current transfer receipts', 'Wages and salaries', 'Supplements to wages and salaries',
#            'Employer contributions for employee pension and insurance funds 6/', "Proprietors' income 7/",
#            "Farm proprietors' income", "Nonfarm proprietors' income", 'Farm earnings', 'Nonfarm earnings',
#            'Private nonfarm earnings', 'Construction', 'Construction of buildings',
#            'Heavy and civil engineering construction', 'Specialty trade contractors', 'Manufacturing',
#            'Durable goods manufacturing', 'Fabricated metal product manufacturing', 'Nondurable goods manufacturing',
#            'Food manufacturing', 'Wholesale trade', 'Retail trade', 'Motor vehicle and parts dealers',
#            'Furniture and home furnishings stores', 'Electronics and appliance stores',
#            'Building material and garden equipment and supplies dealers', 'Food and beverage stores',
#            'Health and personal care stores', 'Gasoline stations', 'Clothing and clothing accessories stores',
#            'Sporting goods, hobby, musical instrument, and book stores', 'General merchandise stores',
#            'Miscellaneous store retailers', 'Nonstore retailers', 'Transportation and warehousing',
#            'Rail transportation', 'Truck transportation', 'Support activities for transportation', 'Information',
#            'Telecommunications', 'Finance and insurance', 'Monetary Authorities-central bank',
#            'Credit intermediation and related activities', 'Insurance carriers and related activities',
#            'Real estate and rental and leasing', 'Real estate', 'Professional, scientific, and technical services',
#            'Management of companies and enterprises',
#            'Administrative and support and waste management and remediation services',
#            'Administrative and support services', 'Waste management and remediation services', 'Educational services',
#            'Health care and social assistance', 'Ambulatory health care services',
#            'Nursing and residential care facilities', 'Arts, entertainment, and recreation',
#            'Amusement, gambling, and recreation industries', 'Accommodation and food services', 'Accommodation',
#            'Food services and drinking places', 'Other services (except government and government enterprises)',
#            'Repair and maintenance', 'Personal and laundry services',
#            'Religious, grantmaking, civic, professional, and similar organizations', 'Private households',
#            'Government and government enterprises', 'Federal civilian', 'Military', 'State and local',
#            'State government', 'Local government']
# for c in raw_data.columns:
#     j = 0
#     for i in raw_data[c].isnull():
#         if i:
#             j += 1
#     percent = (j / len(raw_data[c]))
#     if percent > 0.3:
#         raw_data.pop(c)
#






# 将AQI分为两类
# raw_data = pd.read_csv('../DataSet/ProcessedData/TrainData/train_fill_with_state_mean_2016.csv')
# raw_data['AQI Label'] = 0
# raw_data['AQI Label'][raw_data['AQI'] > raw_data['AQI'].median()] = 1
# raw_data.to_csv('../DataSet/ProcessedData/TrainData/train_fill_with_state_mean_2_label_2016.csv', index=False)
# print(raw_data['AQI Label'])
raw_data = pd.read_csv('../DataSet/ProcessedData/TrainData/train_2016.csv')
raw_data['AQI Label'] = 0
raw_data['AQI Label'][raw_data['AQI'] > raw_data['AQI'].median()] = 1
raw_data.to_csv('../DataSet/ProcessedData/TrainData/train_2016.csv', index=False)
# print(raw_data['AQI Label'])


