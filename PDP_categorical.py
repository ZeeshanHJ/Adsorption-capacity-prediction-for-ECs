"""
This file shows PDP for trained CatBoost model (Categorical data)'
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
X_train1 = X_train.loc[:,'Pyrolysis temperature ':'Humic acid']
X_train2 = X_train.loc[:,'AMCB':'Single']
X_train2.to_numpy()
X_test1 = X_test.loc[:,'Pyrolysis temperature ':'Humic acid']
X_test2 = X_test.loc[:,'AMCB':'Single']
X_test2.to_numpy()
# Feature Scaling
sc = StandardScaler()
X_train1 = sc.fit_transform(X_train1)
X_test1 = sc.transform(X_test1)
np.shape(X_train1)
np.shape(X_train2)
X_train = np.append(X_train1, X_train2, axis=1)
X_test = np.append(X_test1, X_test2, axis=1)

## 04. PDP plot
from pdpbox import pdp

X_new = np.concatenate((X_train,X_test), axis=0)
X_new1 = pd.DataFrame(X_new , columns = X.columns)
X_new2 = np.concatenate((X_train_rev,X_test_rev), axis=0)

## 04-01. PDP plot for Pollutant
pdp_Pollutant = pdp.pdp_isolate(
    model=cb_run, dataset=X_new1, model_features=X.columns,
    feature=['ALA', 'ATE', 'CAR', 'CBZ', 'DCF', 'DIU', 'EE2', 'IBF', 'IBU', 'NPX', 'NXP', 'PYR', 'SIM', 'TEB']
)

size_y = 4.235
font = {'family' : 'Times New Roman',
        'weight' : 'bold',
        'size'   : 14}
x_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
plt.rc('font', **font)
fig1, axes1 = plt.subplots(figsize=(5,size_y))
axes1.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
axes1.set_xticklabels(['ALA', 'ATE', 'CAR', 'CBZ', 'DCF', 'DIU', 'EE2', 'IBF', 'IBU', 'NPX', 'NXP', 'PYR', 'SIM', 'TEB'])
plt.plot(x_data, pdp_Pollutant.pdp, color='red')
plt.ylabel('Partial dependence', fontweight='bold')
plt.xlabel('Pollutant', fontweight='bold')
plt.xticks(rotation = 90)

name = ['ALA', 'ATE', 'CAR', 'CBZ', 'DCF', 'DIU', 'EE2', 'IBF', 'IBU', 'NPX', 'NXP', 'PYR', 'SIM', 'TEB']
cnt = []
j = 0
for i in name:
    num = df.loc[:,'Pollutant'].tolist().count(i)
    cnt.append(num)
    j = j+num
axes1_2 = axes1.twinx()
axes1.set_zorder(1)
axes1_2.set_zorder(-1)
axes1.patch.set_visible(False)
plt.bar(x_data, cnt, color='0.92')
plt.ylabel('Number of data', fontweight='bold')
plt.tight_layout()
plt.show()

## 04-02. PDP plot for Adsoption type
pdp_Adsoption_type = pdp.pdp_isolate(
    model=cb_run, dataset=X_new1, model_features=X.columns,
    feature=['Competative', 'Single']
)

size_y = 4
size_x = 5.1
font = {'family' : 'Times New Roman',
        'weight' : 'bold',
        'size'   : 14}
x_data = [1, 2]
plt.rc('font', **font)
fig2, axes2 = plt.subplots(figsize=(size_x,size_y))
axes2.set_xticks([1, 2])
axes2.set_xticklabels(['Competative', 'Single'])
plt.plot(x_data, pdp_Adsoption_type.pdp, color='red')
plt.ylabel('Partial dependence', fontweight='bold')
plt.xlabel('Adsorption type', fontweight='bold')
plt.xticks(rotation = 0)

name = ['Competative', 'Single']
cnt = []
j = 0
for i in name:
    num = df.loc[:,'Adsorption type'].tolist().count(i)
    cnt.append(num)
    j = j+num
print(j)
axes2_2 = axes2.twinx()
axes2.set_zorder(1)
axes2_2.set_zorder(-1)
axes2.patch.set_visible(False)
plt.bar(x_data, cnt, color='0.92')
plt.ylabel('Number of data', fontweight='bold')
plt.tight_layout()
plt.show()

## 04-03. PDP plot for Adsorbent
pdp_Adsorbent = pdp.pdp_isolate(
    model=cb_run, dataset=X_new1, model_features=X.columns,
    feature=['AMCB','Alkali-modified SCG biochars', 'C-Biochar', 'CB', 'GCRB', 'GCRB-N',
             'MCB', 'NaOH-activated SCW biochars', 'PAC', 'PB600', 'PB800', 'PSB',
             'PSBOX-A', 'Pristine SCG biochar', 'Pristine SCW Biochar']
)

size_y = 4.95
font = {'family' : 'Times New Roman',
        'weight' : 'bold',
        'size'   : 14}
x_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
plt.rc('font', **font)
fig3, axes3 = plt.subplots(figsize=(5,size_y))
axes3.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
axes3.set_xticklabels(['AMCB', 'Alkali-SCG', 'C-Biochar', 'CB', 'GCRB', 'GCRB-N', 'MCB', 'NaOH-SCW', 'PAC', 'PB600', 'PB800', 'PSB', 'PSBOX-A', 'Pristine-SCG', 'Pristine-SCW'])
plt.plot(x_data, pdp_Adsorbent.pdp, color='red')
plt.ylabel('Partial dependence', fontweight='bold')
plt.xlabel('Adsorbent', fontweight='bold')
plt.xticks(rotation = 90)

name = ['AMCB','Alkali-modified SCG biochars', 'C-Biochar', 'CB', 'GCRB', 'GCRB-N',
        'MCB', 'NaOH-activated SCW biochars', 'PAC', 'PB600', 'PB800', 'PSB',
        'PSBOX-A', 'Pristine SCG biochar', 'Pristine SCW Biochar']
cnt = []
j = 0
for i in name:
    num = df.loc[:,'Adsorbent'].tolist().count(i)
    cnt.append(num)
    j = j+num
print(j)
axes3_2 = axes3.twinx()
axes3.set_zorder(1)
axes3_2.set_zorder(-1)
axes3.patch.set_visible(False)
plt.bar(x_data, cnt, color='0.92')
plt.ylabel('Number of data', fontweight='bold')
plt.tight_layout()
plt.show()