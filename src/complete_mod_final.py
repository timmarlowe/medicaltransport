import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import pickle
plt.style.use('ggplot')

def get_completion_data(df):
    dummy = pd.get_dummies(df['time_of_day'])
    df = pd.concat([df,dummy],axis=1)
    X = df[['exp_Completed','weekend','pickup_hour','EarlyMorning','Morning','Afternoon','Evening']]
    y = df['Completed']
    return X,y

def train_test_time(X,y,split_datetime):
    train_idx = X['pickup_hour'] <= split_datetime
    test_idx = X['pickup_hour'] > split_datetime
    pickup_hour = X['pickup_hour'][test_idx]
    X.drop('pickup_hour',axis=1, inplace=True)
    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    return X_train, X_test, y_train, y_test, pickup_hour

def plot_preds(y_test, preds, x,save):
    fig, ax = plt.subplots(1,1,figsize=(15,8))
    ax.plot(x,y_test,label='Actual Rides Given')
    ax.plot(x,preds, label='Prediction (24 hrs in advance)')
    ax.legend()
    ax.set_title('Predicted Capacity Needed for Ride Service')
    plt.savefig(save,dpi=500)
    plt.show()
    plt.close()


class CompletionModel(object):
    def __init__(self,model):
        self._regressor = model
        self._standardizer = StandardScaler(with_mean=True)

    def fit(self,X,y):
        X_scaled = self._standardizer.fit_transform(X)
        self._regressor.fit(X_scaled,y)
        return self

    def predict(self,X):
        X_scaled = self._standardizer.transform(X)
        preds = self._regressor.predict(X_scaled)
        return preds

    def scorer(self,X,y):
        preds = self.predict(X)
        rmse = round(mean_squared_error(preds,y)**(1/2),2)
        return rmse


if __name__=='__main__':

    #Pre-processing
    completion_df = pd.read_pickle('data/hours_df_24_lag.pkl')
    X,y = get_completion_data(completion_df)
    split_date = pd.datetime(2018,8,9,23)
    X_train, X_test, y_train, y_test, test_pickup = train_test_time(X,y,split_date)

    # # #Fitting best model:
    lasso = Lasso(alpha=0.1,normalize=False, fit_intercept=True)
    cm = CompletionModel(lasso)
    cm.fit(X_train, y_train)
    preds=cm.predict(X_test)

    #RMSE of baseline prediction and real prediction
    baseline = X_test['exp_Completed']
    rmse_preds = cm.scorer(X_test, y_test)
    rmse_baseline = round(mean_squared_error(baseline, y_test)**(1/2),2)
    print(f'RMSE for predictions: {rmse_preds}')
    print(f'RMSE for baseline: {rmse_baseline}')

    #Graphing Outcome on test Set
    training = y_train
    plot_preds(y_test.reset_index(drop=True)[-336:], preds[-336:], test_pickup[-336:],
                'images/completion_preds_24_hours_warning2.png')

    #Pickling fit model
    with open('data/complete_model.pkl', 'wb') as f:
        pickle.dump(cm, f)

#Notes
#1 hr ahead best model: Lasso, alpha = .01, normalize=False, fit_intercept=True
    #RMSE for Predictions = 6.82, RMSE for Baseline = 8.15

#12 hours ahead: Lasso, alpha = .01, normalize=False, fit_intercept=True
    #RMSE for Predictions = 8.0, RMSE for Baseline = 8.09

#24 hours ahead: Lasso, alpha = 1, normalize=False, fit_intercept=True
    #RMSE for Predictions = 12.31, RMSE for Baseline = 11.21

#1 day ahead best model: Lasso, alpha = 1, normalize=False, fit_intercept=True
    #RMSE for Predictions = 118.91, RMSE for Baseline = 162.39

#2 days ahead best model: Lasso, alpha = 10, normalize=False, fit_intercept=True
    #RMSE for Predictions = 298.13, RMSE for Baseline = 282.31

#7 days ahead best model: Lasso, alpha = 10, normalize=False, fit_intercept=True
    #RMSE for Predictions = 375.43, RMSE for Baseline = 345.6
