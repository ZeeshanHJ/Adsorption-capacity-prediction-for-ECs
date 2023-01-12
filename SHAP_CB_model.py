"""
This file shows application of SHAP to trained CatBoost model'
"""

## 00. Clear
all = [var for var in globals() if var[0] != "_"]
for var in all:
    del globals()[var]
import warnings
warnings.filterwarnings(action='ignore')

## 01. Import library
import shap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

## 02. Load the trained CatBoost model and Data
cb_run = pickle.load(open("CatBoost.pkl", 'rb'))
df = pd.read_excel("Raw_data.xlsx")
# X Data for input features
X_ad = df.drop(['Capacity','Final concentration'],axis=1)
# Non categorical data
X_non_cat = X_ad.drop(['Adsorbent', 'Pollutant', 'Wastewater type', 'Adsorption type','H', 'O', 'N'],axis=1)
# Categorical data
Adsorbent = X_ad.loc[:,'Adsorbent']
Pollutant = X_ad.loc[:,'Pollutant']
Wastewater = X_ad.loc[:,'Wastewater type']
Adsorption = X_ad.loc[:,'Adsorption type']
X_cat =  pd.concat([Adsorbent, Pollutant, Wastewater, Adsorption],axis=1)

## 03. Data preprocessing
#fit encoder
enc = OneHotEncoder()
enc.fit(X_cat)
#transform categorical features
X_encoded = enc.transform(X_cat).toarray()
#create feature matrix
feature_names = X_cat.columns
new_feature_names = enc.get_feature_names(feature_names)
X_encoded = pd.DataFrame(X_encoded, columns= new_feature_names)
X = pd.concat([X_non_cat, X_encoded],axis=1)
Y = df.loc[:,'Capacity']
# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)
# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Data for De-one hot encoding
X_train_rev, X_test_rev, y_train_rev, y_test_rev = train_test_split(X_non_cat, Y, test_size = 0.3, random_state = 0)
np.shape(X_train_rev)
X_non_cat = pd.DataFrame(X_train_rev, columns=X_non_cat.columns)
X_new = pd.concat([X_non_cat,X_cat], axis=1)

## 04. Calculating Shap values for one-hot encoded data
explainer = shap.TreeExplainer(cb_run)
X_sc = sc.transform(X)
X_sc = pd.DataFrame(X_sc, columns=X.columns)
shap_values = explainer(X_sc)

## 05. Shap summary plot for one-hot encoded data
plt.figure(figsize=(8,8))
shap.summary_plot(shap_values, X, feature_names = X.columns, class_inds='original')
plt.show()

## 06. Shap values for de-encoded data
# 06-01. Get number of unique categories for each feature
n_categories = []
for variables in X_non_cat.columns:
    n = 1
    n_categories.append(n)
for feat in feature_names[:-1]:
    n = X_cat[feat].nunique()
    print(feat, n)
    n_categories.append(n)
# 06-02. Calculate the Shap values for de-encoded data
new_shap_values = []
for values in shap_values.values:
    # split shap values into a list for each feature
    values_split = np.split(values, np.cumsum(n_categories))
    # sum values within each list
    values_sum = [sum(l) for l in values_split]
    new_shap_values.append(values_sum)
# Replace shap values
shap_values.values = np.array(new_shap_values)
# Replace data with categorical feature values
new_data = np.array(X_new)
shap_values.data = np.array(new_data)
# update feature names
shap_values.feature_names = list(X_new.columns)

## 07. Shap summary plot for de-encoded data
plt.figure(figsize=(15,15))
shap.summary_plot(shap_values, X_new, feature_names = X_new.columns, class_inds='original')
plt.show()

## 08. Shap bar plot for de-encoded data
plt.figure(figsize=(20,20))
shap.plots.bar(shap_values)
plt.show()

## 09. Shap waterfall plot for highest adsorption capacity data point
max_value = max(Y)
max_index = Y.values.tolist().index(max_value)
plt.figure(figsize=(18,18))
shap.plots.waterfall(shap_values[max_index])
plt.tight_layout()
plt.rc('font',family='Times New Roman', size=14)
plt.rcParams["font.weight"] = "bold"
plt.show()






