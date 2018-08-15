# Created by Yuexiong Ding
# Date: 2018/8/10
# Description: use RF classification model to analyze the main factors
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
from sklearn.model_selection import GridSearchCV
from mpl_toolkits.mplot3d import Axes3D


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


def rf_clsaaifier():
    colnums = ['State County Code', 'AQI', 'ALAND', 'AWATER', 'INTPTLAT', 'INTPTLONG', 'PCP', 'TEMP', 'County Name',
               'Region', 'Personal income (thousands of dollars)', 'Population (persons) 2/',
               'Per capita personal income (dollars)', 'Earnings by place of work',
               'Less: Contributions for government social insurance 3/',
               'Employee and self-employed contributions for government social insurance',
               'Employer contributions for government social insurance', 'Plus: Adjustment for residence 4/',
               'Equals: Net earnings by place of residence', 'Plus: Dividends, interest, and rent 5/',
               'Plus: Personal current transfer receipts', 'Wages and salaries', 'Supplements to wages and salaries',
               'Employer contributions for employee pension and insurance funds 6/', "Proprietors' income 7/",
               "Farm proprietors' income", "Nonfarm proprietors' income", 'Farm earnings', 'Nonfarm earnings',
               'Private nonfarm earnings', 'Construction', 'Specialty trade contractors', 'Manufacturing',
               'Retail trade', 'Motor vehicle and parts dealers', 'Gasoline stations', 'Finance and insurance',
               'Real estate and rental and leasing', 'Other services (except government and government enterprises)',
               'Repair and maintenance', 'Government and government enterprises', 'Federal civilian', 'Military',
               'State and local', 'State Code', 'County Code', 'AQI Label']
    # df_data = pd.read_csv('../DataSet/ProcessedData/TrainData/train_fill_with_state_mean_2_label_2016.csv',
    #                       usecols=colnums)
    # df_data = pd.read_csv('../DataSet/ProcessedData/TrainData/train_fill_with_state_mean_2_label_2016.csv')
    # df_data = pd.read_csv('../DataSet/ProcessedData/TrainData/train_add_road_data_fill_with_state_mean_2_label_2016.csv')
    df_data = pd.read_csv('../DataSet/ProcessedData/TrainData/new_train_fill_with_state_mean_2016.csv')
    # df_data = df_data.fillna(0)
    # df_data[df_data['AQI'] < 80]
    df_data = df_data.fillna(df_data.mean())
    df_data['Density'] = df_data['Population (persons)'] / df_data['ALAND']
    df_data['Construction'] = df_data['Construction'] / df_data['ALAND']
    df_data['Manufacturing'] = df_data['Manufacturing'] / df_data['ALAND']
    df_data['Wholesale trade'] = df_data['Wholesale trade'] / df_data['ALAND']
    df_data['Retail trade'] = df_data['Retail trade'] / df_data['ALAND']
    df_data['Information'] = df_data['Information'] / df_data['ALAND']
    df_data['Professional scientific and technical services'] = \
        df_data['Professional scientific and technical services'] / df_data['ALAND']
    df_data['Administrative and support and waste management and remediation services'] = \
        df_data['Administrative and support and waste management and remediation services'] / df_data['ALAND']
    df_data['Other services (except government and government enterprises)'] = \
        df_data['Other services (except government and government enterprises)'] / df_data['ALAND']
    df_data['Farm earnings Forestry Fishing Mining Quarrying Oil Gas extraction'] = \
        df_data['Farm earnings Forestry Fishing Mining Quarrying Oil Gas extraction'] / df_data['ALAND']
    df_data['Finance Insurance Real estate Rental Leasing'] = \
        df_data['Finance Insurance Real estate Rental Leasing'] / df_data['ALAND']
    df_data['Arts Entertainment Recreation Accommodation Food services'] = \
        df_data['Arts Entertainment Recreation Accommodation Food services'] / df_data['ALAND']
    df_data.pop('State County Code')
    df_data.pop('Region')
    df_data.pop('INTPTLAT')
    df_data.pop('INTPTLONG')
    df_data.pop('Workers 16 years and over')
    df_data.pop('Car truck or van')
    df_data.pop('Car truck or van-Drove alone')
    df_data.pop('Car truck or van-Carpooled')
    df_data.pop('Public transportation (excluding taxicab)')
    df_data.pop('Walked')
    df_data.pop('Bicycle')
    df_data.pop('Taxicab motorcycle or other means')
    df_data.pop('Worked at home')
    df_data.pop('No vehicle available')
    df_data.pop('1 vehicle available')
    df_data.pop('2 vehicles available')
    df_data.pop('3 or more vehicles available')
    df_data.pop('State Code')
    df_data.pop('County Code')
    df_data.pop('AQI')
    # df_data = shuffle(df_data)
    y = df_data.pop('AQI Label')
    X = preprocessing.scale(df_data, with_mean=True, with_std=True, axis=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    y_label_names = ['class0', 'class1']

    # max_depth = list(range(1, 50, 2))
    # max_features = list(range(1, 32, 2))
    # tuned_parameters = [{'max_depth': max_depth, 'max_features': max_features}]
    # rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, oob_score=True)
    # clf = GridSearchCV(rf, tuned_parameters, cv=5)
    # clf.fit(X_train, y_train)
    # mean = clf.cv_results_['mean_test_score']
    # mean = np.array(mean).reshape(len(max_depth), -1)
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # X = max_depth
    # Y = max_features
    # X, Y = np.meshgrid(X, Y)
    # ax.plot_surface(np.array(X).T, np.array(Y).T, mean, rstride=1, cstride=1, cmap='rainbow')
    # plt.show()

    # rf = RandomForestClassifier(n_estimators=100, oob_score=True, class_weight='balanced')
    # scores = cross_val_score(rf, X, y, cv=5, scoring='roc_auc')
    # print(scores.mean())

    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, oob_score=True, max_depth=8, max_features=6, class_weight='balanced')
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


if __name__ == '__main__':
    rf_clsaaifier()
