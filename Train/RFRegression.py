# Created by Yuexiong Ding
# Date: 2018/8/15
# Description: 
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from mpl_toolkits.mplot3d import Axes3D

# df_data = pd.read_csv('../DataSet/ProcessedData/TrainData/train_fill_with_state_mean_2016.csv')
# df_data = pd.read_csv(
#     '../DataSet/ProcessedData/TrainData/train_add_road_data_fill_with_state_mean_2_label_2016.csv')
df_data = pd.read_csv('../DataSet/ProcessedData/TrainData/new_train_fill_with_state_mean_2016.csv')
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
# df_data.pop('County Name')
df_data.pop('State Code')
df_data.pop('County Code')
df_data.pop('AQI Label')
y = preprocessing.scale(df_data.pop('AQI'))
# X = df_data.fillna(0)
X = df_data.fillna(df_data.mean())
X = preprocessing.scale(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

max_depth = list(range(1, 100, 2))
max_features = list(range(1, 32, 2))
tuned_parameters = [{'max_depth': max_depth, 'max_features': max_features}]
rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, oob_score=True)
clf = GridSearchCV(rf, tuned_parameters, cv=5, scoring='neg_mean_squared_error')
# clf = GridSearchCV(rf, tuned_parameters, cv=5, scoring='r2')
clf.fit(X_train, y_train)
mean = clf.cv_results_['mean_test_score']
mean = np.array(mean).reshape(len(max_depth), -1)
print(mean)
fig = plt.figure()
ax = Axes3D(fig)
X = max_depth
Y = max_features
X, Y = np.meshgrid(X, Y)
ax.plot_surface(np.array(X).T, np.array(Y).T, mean, rstride=1, cstride=1, cmap='rainbow')
plt.show()

# rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
# # scores = cross_val_score(clf, iris.data, iris.target, cv=5)
# rf.fit(X_train, y_train)
# y_pred = rf.predict(X_test)
# print(sorted(zip(rf.feature_importances_, df_data.columns), reverse=True))
# print(metrics.mean_squared_error(y_test, y_pred))
# print(metrics.r2_score(y_test, y_pred))
# plt.scatter(range(len(y_test)), y_test, c='b')
# plt.scatter(range(len(y_pred)), y_pred, c='r')
# plt.show()


