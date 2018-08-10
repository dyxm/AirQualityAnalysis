# Created by Yuexiong Ding
# Date: 2018/8/10
# Description: use RF model to analyze the main factors
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle


def over_sampling(df_data, split=0.7, over_num=5):
    df_data = df_data.fillna(0)
    X_train = df_data.sample(frac=split)
    X_test = df_data[~df_data.index.isin(X_train.index)]
    y_test = preprocessing.scale(X_test.pop('AQI'))
    high_aqi_data = X_train[X_train['AQI'] > 50]
    # meddle_aqi_data = raw_data[(raw_data['AQI'] > 20) & (raw_data['AQI'] <= 50)]
    low_aqi_data = X_train[X_train['AQI'] <= 20]
    for i in range(over_num):
        X_train = X_train.append(high_aqi_data)
        if i % 2 == 0:
            X_train = X_train.append(low_aqi_data)
    X_train = shuffle(X_train)
    y_train = preprocessing.scale(X_train.pop('AQI'))
    print(len(y_train))

    return X_train, X_test, y_train, y_test

def lasso(df_data):
    # df_data.pop('AQI Label')
    # y = preprocessing.scale(df_data.pop('AQI'))
    # y = df_data.pop('AQI')
    # X = df_data.fillna(0)
    # X = preprocessing.scale(X)

    X_train, X_test, y_train, y_test = over_sampling(df_data, split=0.7, over_num=10)
    columns = X_train.columns
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


def rf_regressor(df_data):
    # df_data.pop('AQI Label')
    df_data = df_data.fillna(0)
    X_train = df_data.sample(frac=0.7)
    X_test = df_data[~df_data.index.isin(X_train.index)]
    y_test = preprocessing.scale(X_test.pop('AQI'))
    high_aqi_data = X_train[X_train['AQI'] > 50]
    # meddle_aqi_data = raw_data[(raw_data['AQI'] > 20) & (raw_data['AQI'] <= 50)]
    low_aqi_data = X_train[X_train['AQI'] <= 20]
    for i in range(11):
        X_train = X_train.append(high_aqi_data)
        if i % 2 == 0:
            X_train = X_train.append(low_aqi_data)
    X_train = shuffle(X_train)
    # y = preprocessing.scale(df_data.pop('AQI'))
    y_train = preprocessing.scale(X_train.pop('AQI'))
    # X = df_data.fillna(0)
    # X_train= X_train.fillna(0)
    # X = preprocessing.scale(X)

    rf = RandomForestRegressor(n_estimators=1000, n_jobs=-1, oob_score=True, max_features=100, max_depth=10)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # scores = cross_val_score(clf, iris.data, iris.target, cv=5)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print(sorted(zip(rf.feature_importances_, df_data.columns), reverse=True))
    print(rf.oob_score_)
    print(metrics.mean_squared_error(y_test, y_pred))
    print(metrics.r2_score(y_test, y_pred))
    plt.scatter(range(len(y_test)), y_test, c='b')
    plt.scatter(range(len(y_pred)), y_pred, c='r')
    plt.show()


def rf_clsaaifier(df_data):
    df_data = df_data.fillna(0)
    X_train = df_data.sample(frac=0.7)
    X_test = df_data[~df_data.index.isin(X_train.index)]
    y_test = X_test.pop('AQI Label')
    X_test.pop('AQI')

    high_aqi_data = X_train[X_train['AQI'] > 50]
    low_aqi_data = X_train[X_train['AQI'] <= 20]
    for i in range(200):
        X_train = X_train.append(high_aqi_data)
        # if i % 10 == 0:
        #     X_train = X_train.append(low_aqi_data)
    X_train = shuffle(X_train)

    X_train.pop('AQI')
    y_train = X_train.pop('AQI Label')
    print(len(y_train[y_train == 1]))
    print(len(y_train[y_train == 0]))

    y_label_names = ['class0', 'class1']
    rf = RandomForestClassifier(n_estimators=1000, n_jobs=-1, oob_score=True)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print(sorted(zip(rf.feature_importances_, df_data.columns), reverse=True))
    print('oob_score: ', rf.oob_score_)
    print('auc: ', metrics.auc(y_test, y_pred, reorder=True))
    print('roc_auc: ', metrics.roc_auc_score(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred, target_names=y_label_names))
    plt.scatter(range(len(y_test)), y_test, c='b')
    plt.scatter(range(len(y_pred)), y_pred, c='r')
    plt.show()


def clustering_aqi(x, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(x)
    return kmeans.predict(x)


def main():
    raw_data = pd.read_csv('../DataSet/ProcessedData/TrainData/train_2016.csv')
    # raw_data = pd.read_csv('../DataSet/ProcessedData/TrainData/train_oversampling_2_label_of_aqi_2016.csv')
    raw_data.pop('State County Code')
    raw_data.pop('Region')
    raw_data.pop('INTPTLAT')
    raw_data.pop('INTPTLONG')
    raw_data.pop('County Name')
    # raw_data.pop('Internet publishing and broadcasting 8/')



    # 分类
    # rf_clsaaifier(raw_data)

    # 回归
    # RF
    # rf_regressor(raw_data)

    # LASSO
    lasso(raw_data)


if __name__ == '__main__':
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
           'Computer and electronic product manufacturing',
           'Electrical equipment, appliance, and component manufacturing',
           'Motor vehicles, bodies and trailers, and parts manufacturing',
           'Other transportation equipment manufacturing',
           'Furniture and related product manufacturing', 'Miscellaneous manufacturing',
           'Nondurable goods manufacturing',
           'Food manufacturing', 'Beverage and tobacco product manufacturing', 'Textile mills', 'Textile product mills',
           'Apparel manufacturing', 'Leather and allied product manufacturing', 'Paper manufacturing',
           'Printing and related support activities', 'Petroleum and coal products manufacturing',
           'Chemical manufacturing',
           'Plastics and rubber products manufacturing', 'Wholesale trade', 'Retail trade',
           'Motor vehicle and parts dealers', 'Furniture and home furnishings stores',
           'Electronics and appliance stores',
           'Building material and garden equipment and supplies dealers', 'Food and beverage stores',
           'Health and personal care stores', 'Gasoline stations', 'Clothing and clothing accessories stores',
           'Sporting goods, hobby, musical instrument, and book stores', 'General merchandise stores',
           'Miscellaneous store retailers', 'Nonstore retailers', 'Transportation and warehousing',
           'Air transportation',
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
           'Private households', 'Government and government enterprises', 'Federal civilian', 'Military',
           'State and local',
           'State government', 'Local government', 'Workers 16 years and over', 'Car truck or van',
           'Car truck or van-Drove alone', 'Car truck or van-Carpooled', 'Public transportation (excluding taxicab)',
           'Walked', 'Bicycle', 'Taxicab motorcycle or other means', 'Worked at home', 'No vehicle available',
           '1 vehicle available', '2 vehicles available', '3 or more vehicles available']

    main()




# y[y <= 50] = 0
# y[(y > 50) & (y <= 100)] = 1
# y[(y > 100)] = 2
# print(len(y[y == 0]))
# print(len(y[y == 1]))
# print(len(y[y == 2]))


# X = raw_data.fillna(raw_data.mean())
# X = raw_data.fillna(method='pad')

# print(X)



# roc_auc = metrics.roc_auc_score(y_test, y_pred)
# print(roc_auc)
# mean_squared_error = metrics.mean_squared_error(y_test, y_pred)
# r2_score = metrics.r2_score(y_test, y_pred)
# print(mean_squared_error)
# print(r2_score)
# plt.scatter(range(len(y_test)), y_test, c='b')
# plt.scatter(range(len(y_pred)), y_pred, c='r')
# plt.show()
# print(raw_data)
# raw_data.to_csv('../DataSet/ProcessedData/TrainData/train_fill_by_pad_2016.csv')
# plt.scatter(raw_data['WIND'], raw_data['AQI'], c='r')
# plt.scatter(raw_data['TEMP'], raw_data['AQI'], c='b')
# plt.scatter(raw_data['AWATER'] / raw_data['Population (persons) 2/'], raw_data['AQI'], c='y')
# plt.show()
