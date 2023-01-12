"""
This file shows application of Random Forest model to predict 'ECs adsorption capacity on Biochar materials'
"""

## 00. Clear
all = [var for var in globals() if var[0] != "_"]
for var in all:
    del globals()[var]
import warnings
warnings.filterwarnings(action='ignore')

## 01. Import library
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from sklearn import ensemble

## 02. Loading the data
df = pd.read_excel("Raw_data.xlsx")
Material_characteristics = df.loc[:,'Adsorbent':'Average pore size']
Adsorption_experiment = df.loc[:,'Pollutant':'Final concentration']
Capacity = df.loc[:,'Capacity']

## 03. Data preprocess
## 03-01. One hot encoder - Adsorbent, Pollutant, Wastewater type, Adsorption type
Adsorbent = pd.get_dummies(Material_characteristics.loc[:,'Adsorbent'])
Pollutant = pd.get_dummies(Adsorption_experiment.loc[:,'Pollutant'])
Wastewater_type = pd.get_dummies(Adsorption_experiment.loc[:,'Wastewater type'])
Adsorption_type = pd.get_dummies(Adsorption_experiment.loc[:,'Adsorption type'])

## 03-02. Preparing X and Y
Data = pd.concat([Material_characteristics, Adsorption_experiment, Adsorbent, Pollutant, Wastewater_type, Adsorption_type],axis=1)
Data.head()
Nan = Data.isnull().sum()
X = Data.drop(['Adsorbent','H', 'O', 'N', 'Pollutant', 'Wastewater type', 'Adsorption type', 'Final concentration'],axis=1)
Y = Capacity
# Check the Nan
for col in X.columns:
    df_nan = X[X[col].isin([np.nan, np.inf, -np.inf])]
    if df_nan.shape[0] != 0:
        print(col)
# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)
# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
## Type change
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

## 04. Parameter optimization (Bayesian optimization)
rf_run = ensemble.RandomForestRegressor(random_state=0)
rf_run.get_params()

def bo_params_rf(max_depth, min_samples_leaf, min_samples_split, max_samples,n_estimators):
    params = {'max_depth': int(max_depth),
              'min_samples_leaf': int(min_samples_leaf),
              'min_samples_split': int(min_samples_split),
              'max_samples': max_samples,
              'n_estimators': int(n_estimators)}
    reg = ensemble.RandomForestRegressor(random_state=0,
                                         max_depth=params['max_depth'],
                                         min_samples_leaf=params['min_samples_leaf'],
                                         min_samples_split=params['min_samples_split'],
                                         max_samples=params['max_samples'],
                                         n_estimators=params['n_estimators'])
    reg.fit(X_train, y_train)
    preds = reg.predict(X_train)
    score = np.sqrt(mean_squared_error(y_train, preds))
    return - score

rf_BO = BayesianOptimization(bo_params_rf, {'max_depth': (60,100),
                                            'min_samples_leaf': (1,4),
                                            'min_samples_split': (2,10),
                                            'max_samples': (0.5,1),
                                            'n_estimators': (100, 250)},
                              random_state=1)
results = rf_BO.maximize(n_iter=200, init_points=20,acq='ei')
params = rf_BO.max['params']
params['max_depth']= int(params['max_depth'])
params['min_samples_leaf']= int(params['min_samples_leaf'])
params['min_samples_split']= int(params['min_samples_split'])
params['n_estimators']= int(params['n_estimators'])

## 05. Model training
rf_run = ensemble.RandomForestRegressor(random_state=0,
                                        max_depth=params['max_depth'],
                                        min_samples_leaf=params['min_samples_leaf'],
                                        min_samples_split=params['min_samples_split'],
                                        max_samples=params['max_samples'],
                                        n_estimators=params['n_estimators'])
rf_run.fit(X_train, y_train)

## 06. Model evaluation
    # train
train_predict = rf_run.predict(X_train)
print("The training RMSE of the Random forest model is :\t", np.sqrt(mean_squared_error(train_predict,y_train)))
print("The training R^2 of the Random forest model is :\t", r2_score(train_predict,y_train))
print("The training MAE of theRandom forest model is :\t", mean_absolute_error(train_predict,y_train))
    # validation
test_predict = rf_run.predict(X_test)
print("The test RMSE of the Random forest model is :\t",  np.sqrt(mean_squared_error(test_predict,y_test)))
print("The test R^2 of the Random forest model is :\t",  r2_score(test_predict,y_test))
print("The test MAE of the Random forest model is :\t",  mean_absolute_error(test_predict,y_test))

## 07. Model evaluation with 1000 iteration
Tr_RMSE = []
Te_RMSE = []
Tr_R2 = []
Te_R2 = []
Tr_MAE = []
Te_MAE = []

for i in range(1, 1001):
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=i)
    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    rf_run.fit(X_train, y_train)

    # train
    train_predict = rf_run.predict(X_train)
    Tr_RMSE.append(np.sqrt(mean_squared_error(train_predict, y_train)))
    Tr_R2.append(r2_score(train_predict, y_train))
    Tr_MAE.append(mean_absolute_error(train_predict, y_train))
    # test
    test_predict = rf_run.predict(X_test)
    Te_RMSE.append(np.sqrt(mean_squared_error(test_predict, y_test)))
    Te_R2.append(r2_score(test_predict, y_test))
    Te_MAE.append(mean_absolute_error(test_predict, y_test))

    print("The iteration is \t", i)

Tr_RMSE_df = pd.DataFrame(Tr_RMSE)
Te_RMSE_df = pd.DataFrame(Te_RMSE)
Tr_R2_df = pd.DataFrame(Tr_R2)
Te_R2_df = pd.DataFrame(Te_R2)
Tr_MAE_df = pd.DataFrame(Tr_MAE)
Te_MAE_df = pd.DataFrame(Te_MAE)
model_performance = X1 = pd.concat([Tr_RMSE_df, Te_RMSE_df, Tr_R2_df, Te_R2_df, Tr_MAE_df, Te_MAE_df],axis=1)
model_performance.columns = ["Tr_RMSE", "Te_RMSE", "Tr_R2", "Te_R2", "Tr_MAE", "Te_MAE"]