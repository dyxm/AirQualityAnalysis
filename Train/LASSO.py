# Created by Yuexiong Ding
# Date: 2018/8/9
# Description: use lasso model to analyze the main factors

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV



col = ['ALAND', 'AWATER', 'PCP', 'TEMP', 'WIND',
       'Region', 'Personal income (thousands of dollars)', 'Population (persons) 2/',
       'Per capita personal income (dollars)', 'Earnings by place of work',
       'Less: Contributions for government social insurance 3/',
       'Employee and self-employed contributions for government social insurance',
       'Employer contributions for government social insurance', 'Plus: Adjustment for residence 4/',
       'Equals: Net earnings by place of residence', 'Plus: Dividends, interest, and rent 5/',
       'Plus: Personal current transfer receipts', 'Wages and salaries', 'Supplements to wages and salaries',
       'Employer contributions for employee pension and insurance funds 6/', "Proprietors' income 7/",
       "Farm proprietors' income", "Nonfarm proprietors' income", 'Farm earnings', 'Nonfarm earnings',
       'Private nonfarm earnings', 'Forestry, fishing, and related activities', 'Forestry and logging',
       'Fishing, hunting and trapping', 'Support activities for agriculture and forestry',
       'Mining, quarrying, and oil and gas extraction', 'Oil and gas extraction', 'Mining (except oil and gas)',
       'Support activities for mining', 'Utilities', 'Construction', 'Construction of buildings',
       'Heavy and civil engineering construction', 'Specialty trade contractors', 'Manufacturing',
       'Durable goods manufacturing', 'Wood product manufacturing', 'Nonmetallic mineral product manufacturing',
       'Primary metal manufacturing', 'Fabricated metal product manufacturing', 'Machinery manufacturing',
       'Computer and electronic product manufacturing', 'Electrical equipment, appliance, and component manufacturing',
       'Motor vehicles, bodies and trailers, and parts manufacturing', 'Other transportation equipment manufacturing',
       'Furniture and related product manufacturing', 'Miscellaneous manufacturing', 'Nondurable goods manufacturing',
       'Food manufacturing', 'Beverage and tobacco product manufacturing', 'Textile mills', 'Textile product mills',
       'Apparel manufacturing', 'Leather and allied product manufacturing', 'Paper manufacturing',
       'Printing and related support activities', 'Petroleum and coal products manufacturing', 'Chemical manufacturing',
       'Plastics and rubber products manufacturing', 'Wholesale trade', 'Retail trade',
       'Motor vehicle and parts dealers', 'Furniture and home furnishings stores', 'Electronics and appliance stores',
       'Building material and garden equipment and supplies dealers', 'Food and beverage stores',
       'Health and personal care stores', 'Gasoline stations', 'Clothing and clothing accessories stores',
       'Sporting goods, hobby, musical instrument, and book stores', 'General merchandise stores',
       'Miscellaneous store retailers', 'Nonstore retailers', 'Transportation and warehousing', 'Air transportation',
       'Rail transportation', 'Water transportation', 'Truck transportation',
       'Transit and ground passenger transportation', 'Pipeline transportation',
       'Scenic and sightseeing transportation', 'Support activities for transportation', 'Couriers and messengers',
       'Warehousing and storage', 'Information', 'Publishing industries (except Internet)',
       'Motion picture and sound recording industries', 'Broadcasting (except Internet)', 'Telecommunications',
       'Data processing, hosting, and related services', 'Other information services 8/', 'Finance and insurance',
       'Monetary Authorities-central bank', 'Credit intermediation and related activities',
       'Securities, commodity contracts, and other financial investments and related activities',
       'Insurance carriers and related activities', 'Funds, trusts, and other financial vehicles',
       'Real estate and rental and leasing', 'Real estate', 'Rental and leasing services',
       'Lessors of nonfinancial intangible assets (except copyrighted works)',
       'Professional, scientific, and technical services', 'Management of companies and enterprises',
       'Administrative and support and waste management and remediation services',
       'Administrative and support services', 'Waste management and remediation services', 'Educational services',
       'Health care and social assistance', 'Ambulatory health care services', 'Hospitals',
       'Nursing and residential care facilities', 'Social assistance', 'Arts, entertainment, and recreation',
       'Performing arts, spectator sports, and related industries',
       'Museums, historical sites, and similar institutions', 'Amusement, gambling, and recreation industries',
       'Accommodation and food services', 'Accommodation', 'Food services and drinking places',
       'Other services (except government and government enterprises)', 'Repair and maintenance',
       'Personal and laundry services', 'Religious, grantmaking, civic, professional, and similar organizations',
       'Private households', 'Government and government enterprises', 'Federal civilian', 'Military', 'State and local',
       'State government', 'Local government', 'Workers 16 years and over', 'Car truck or van',
       'Car truck or van-Drove alone', 'Car truck or van-Carpooled', 'Public transportation (excluding taxicab)',
       'Walked', 'Bicycle', 'Taxicab motorcycle or other means', 'Worked at home', 'No vehicle available',
       '1 vehicle available', '2 vehicles available', '3 or more vehicles available']
df_data = pd.read_csv('../DataSet/ProcessedData/TrainData/train_fill_with_state_mean_2016.csv')
df_data.pop('State County Code')
df_data.pop('Region')
df_data.pop('INTPTLAT')
df_data.pop('INTPTLONG')
df_data.pop('County Name')
y = preprocessing.scale(df_data.pop('AQI'))
X = df_data.fillna(0)
X = preprocessing.scale(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# X_train, X_test, y_train, y_test = over_sampling(df_data, split=0.7, over_num=10)
# columns = X_train.columns
columns = df_data.columns
lassocv = LassoCV()
lassocv.fit(preprocessing.scale(X_train), y_train)
print('alpha: ', lassocv.alpha_)
print(sorted(zip(np.fabs(lassocv.coef_), columns), reverse=True))
y_pred = lassocv.predict(preprocessing.scale(X_test))
print(metrics.mean_squared_error(y_test, y_pred))
print(metrics.r2_score(y_test, y_pred))
plt.scatter(range(len(y_test)), y_test, c='b')
plt.scatter(range(len(y_pred)), y_pred, c='r')
plt.show()
