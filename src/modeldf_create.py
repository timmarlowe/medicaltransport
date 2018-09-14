import pandas as pd
import numpy as np
import glob
import re
import math
import datetime

def get_cancel(row):
    #Home df
    df = model_df_filtered[(model_df_filtered['pickup_day']<row.pickup_day) & (model_df_filtered['patientMedicalId']==row.patientMedicalId)]
    df_counts = df.loc[:, ['Completed','CancelledDayOf','CancelledHourOf']].sum(axis=0)
    return df_counts

if __name__=='__main__':

    #Start by using original df created for eda
    cleaned = pd.read_pickle('data/rides_clean.pkl')

    model_df = cleaned.copy()

    model_df['VehicleAmb'] = 0
    model_df['VehicleAmb'][model_df['requestedVehicleType']=='AMB'] = 1

    #Filtering out any observations that aren't 'Completed, CancelledDayOf, CancelledHourOf'
    model_df_filtered = model_df.loc[model_df['status_v2'].isin(['Completed','CancelledDayOf','CancelledHourOf'])].reset_index(drop=True)

    #Turning Categoricals into dummies
    cat_list = ['create_hours_cat','time_of_day','status_v2']
    for cat in cat_list:
        model_df_filtered = pd.concat([model_df_filtered, pd.get_dummies(model_df_filtered[cat])], axis=1)

    model_df_filtered.drop(['create_hours_cat','time_of_day'], axis=1, inplace=True)

    #Getting values for previous completes, cancels of users
    model_agg = model_df_filtered.apply(get_cancel,axis=1)
    model_agg.rename(index=str,columns={'Completed':'Completed_Count',
                    'CancelledDayOf': 'CancelledDayOf_Count',
                    'CancelledHourOf':'CancelledHourOf_Count'},inplace=True)
    model_agg['CancelledDayOfPct'] = model_agg['CancelledDayOf_Count']/model_agg[['CancelledDayOf_Count','CancelledHourOf_Count','Completed_Count']].sum(axis=1)
    model_agg['CancelledHourOfPct'] = model_agg['CancelledHourOf_Count']/model_agg[['CancelledDayOf_Count','CancelledHourOf_Count','Completed_Count']].sum(axis=1)
    model_agg.drop(['CancelledDayOf_Count','CancelledHourOf_Count'],axis=1, inplace=True)
    model_agg.fillna({'CancelledHourOfPct':0,'CancelledDayOfPct':0},inplace=True)

    model_df_filtered.index=model_df_filtered.index.map(str)
    model_df_final = pd.concat([model_df_filtered, model_agg], axis=1)
    #Changing these cancellation values to percents
    model_df_final.drop(['patientMedicalId','CancelledDayOf',
                        'CancelledHourOf','Completed','Night','CreatedEarly'],axis=1, inplace=True)

    model_df_final['y']=0
    model_df_final['y'][(model_df_final['status_v2']=='CancelledHourOf') | (model_df_final['status_v2']=='CancelledDayOf')]=1
    model_df_final = model_df_final[model_df_final['pickup_day'] > pd.datetime(2018,6,14)]
    #Pickling final df
    model_df_final.to_pickle('data/model_df.pkl')
