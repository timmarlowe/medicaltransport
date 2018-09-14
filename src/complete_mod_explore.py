import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
plt.style.use('ggplot')

def get_completion_data(path):
    df = pd.read_pickle(path)
    X = df[['exp_Completed','weekend','Completed_p-lag','pickup_hour']]
    y = df['Completed']
    return X,y

def train_test_time(X,y,split_datetime):
    train_idx = X['pickup_hour'] <= split_datetime
    test_idx = X['pickup_hour'] > split_datetime
    X.drop('pickup_hour',axis=1, inplace=True)
    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    return X_train, X_test, y_train, y_test

def gridsearch(estimator,name,params,X,y):
    model = GridSearchCV(estimator,param_grid=params, scoring='neg_mean_squared_error',cv=2)
    model.fit(X_train, y_train)
    print(f'{name} best parameters are {model.best_params_}')
    model_best_rmse = round((model.best_score_*-1)**(1/2),2)
    print(f'{name} best rmse was {model_best_rmse}')

if __name__=='__main__':
    X,y = get_completion_data('data/hours_df_24_lag.pkl')
    split_datetime = pd.datetime(2018,8,9,23)
    X_train, X_test, y_train, y_test = train_test_time(X,y,split_datetime)
    standardizer = StandardScaler()
    X_train_std = standardizer.fit_transform(X_train)
    X_test_std = standardizer.transform(X_test)

    #Vanilla Linear Regression:
    linear_params = {'normalize':[True, False],'fit_intercept':[True,False]}
    gridsearch(LinearRegression(),'Linear Regression',linear_params,X_train,y_train)

    #Ridge Linear Regression:
    ridge_params = {'alpha':[0.01,0.1,1,10,100],'normalize':[True, False],'fit_intercept':[True,False]}
    gridsearch(Ridge(),'Ridge',ridge_params,X_train_std,y_train)

    #Lasso Linear Regression
    lasso_params = {'alpha':[0.1,1,10,100],'normalize':[True, False],'fit_intercept':[True,False]}
    gridsearch(Lasso(),'Lasso',lasso_params,X_train_std,y_train)

    #Random Forest Regression
    rf_params = {'n_estimators':[100,500,1000],'max_features':['auto','sqrt',None],'min_samples_split':[2,4], 'n_jobs':[-1]}
    gridsearch(RandomForestRegressor(),'Random Forest',rf_params,X_train,y_train)

    #Gradient Boosting Regression
    gbr_params = {'learning_rate':[.01,.1,1],'n_estimators':[100,1000],'min_samples_split':[2,4], 'max_features':['auto','sqrt',None]}
    gridsearch(GradientBoostingRegressor(),'Gradient Boosting',gbr_params,X_train,y_train)

    #Adaptive Boosting Regression
    abr_params = {'learning_rate':[.01,.1,1],'n_estimators':[100,1000]}
    gridsearch(AdaBoostRegressor(),'Adaptive Boosting',abr_params,X_train,y_train)
