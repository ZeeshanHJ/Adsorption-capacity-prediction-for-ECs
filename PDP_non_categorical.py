"""
This file shows PDP for trained CatBoost model (Non-categorical data)'
"""

## 00. Clear
all = [var for var in globals() if var[0] != "_"]
for var in all:
    del globals()[var]
import warnings
warnings.filterwarnings(action='ignore')

## 01. Import library
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

## 02. Load the trained CatBoost model and Data
cb_run = pickle.load(open("CatBoost.pkl", 'rb'))
df = pd.read_excel("Raw_data.xlsx")

## 03. Data preprocess
Material_characteristics = df.loc[:,'Adsorbent':'Average pore size']
Adsorption_experiment = df.loc[:,'Pollutant':'Final concentration']
Capacity = df.loc[:,'Capacity']
## 03-01. One hot encoder - Adsorbent, Pollutant, Wastewater type, Adsorption type
Adsorbent = pd.get_dummies(Material_characteristics.loc[:,'Adsorbent'])
Pollutant = pd.get_dummies(Adsorption_experiment.loc[:,'Pollutant'])
Wastewater_type = pd.get_dummies(Adsorption_experiment.loc[:,'Wastewater type'])
Adsorption_type = pd.get_dummies(Adsorption_experiment.loc[:,'Adsorption type'])
## 03-02. Preparing X and Y
Data = pd.concat([Material_characteristics, Adsorption_experiment, Adsorbent, Pollutant, Wastewater_type, Adsorption_type],axis=1)
X = Data.drop(['Adsorbent','H', 'O', 'N', 'Pollutant', 'Wastewater type', 'Adsorption type', 'Final concentration'],axis=1)
Y = Capacity
# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)
X_train_rev, X_test_rev, y_train_rev, y_test_rev = train_test_split(X, Y, test_size = 0.3, random_state = 0)
# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

## 04. PDP plot for non-categorical data
from sklearn.inspection import partial_dependence

Top_rank = ['N/C', 'Initial concentration', 'Surface area', 'C', 'Adsorption time', 'Pore volume']
Top_rank_ind = [7, 12, 8, 2, 11, 9]

print(X.columns[Top_rank_ind[4]])

font = {'family' : 'Times New Roman',
        'weight' : 'bold',
        'size'   : 14}
j = 0
X_new = np.concatenate((X_train_rev,X_test_rev), axis=0)
for i in Top_rank:
    print(i)
    print(X.columns[Top_rank_ind[j]])
    plt.rc('font', **font)
    results = partial_dependence(cb_run, X_train, [Top_rank_ind[j]])

    x_data = results[1][0] * np.std(X_new[:,Top_rank_ind[j]]) + np.mean(X_new[:,Top_rank_ind[j]])
    fig, ax1 = plt.subplots(figsize=(5,4))
    plt.plot(x_data, results[0][0], color='red')
    plt.ylabel('Partial dependence', fontweight='bold')
    plt.xlabel(i, fontweight='bold')

    ax2 = ax1.twinx()
    sns.rugplot(X_new[:,Top_rank_ind[j]], color='0')
    sns.distplot(x=X_new[:,Top_rank_ind[j]], kde=False, color='lightgray')
    plt.margins(0)
    plt.ylabel('Number of data', fontweight='bold')
    plt.show()
    j = j+1
