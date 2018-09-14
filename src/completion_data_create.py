import pandas as pd
import numpy as np
import glob
import re
import math
import datetime
pd.options.mode.chained_assignment = None

def get_data(path):
    path = path
    allFiles = glob.glob(path + "/*.csv") #all files in path identified as csv are stored in var
    frame = pd.DataFrame()
    list_ = []
    for file_ in allFiles:
        f = pd.read_csv(file_,index_col=None,
                        header=0,
                        parse_dates=[['pickupDate', 'pickupTime'],
                        ['appointmentDate','appointmentTime'],
                        ['createdDate','createdTime'],
                        ['updatedDate','updatedTime'],
                        ['acceptedDate','acceptedTime'],
                        ['startedDate','startedTime'],
                        ['arrivedDate','arrivedTime'],
                        ['completedDate','completedTime'],
                        ['cancelledDate','cancelledTime']])
        list_.append(f)
    df = pd.concat(list_)

    df.rename(index=str,columns={'pickupDate_pickupTime': 'pickup_datetime',
                                'appointmentDate_appointmentTime': 'appt_datetime',
                                'createdDate_createdTime': 'created_datetime',
                                'updatedDate_updatedTime': 'updated_datetime',
                                'acceptedDate_acceptedTime': 'accepted_datetime',
                                'startedDate_startedTime': 'started_datetime',
                                'arrivedDate_arrivedTime': 'arrived_datetime',
                                'completedDate_completedTime': 'completed_datetime',
                                'cancelledDate_cancelledTime': 'cancelled_datetime',
                                'toLogitude': 'toLongitude'},
                                inplace=True)

    df.replace('nan nan',np.nan,inplace=True)
    df['cancelled_datetime'] = pd.to_datetime(df['cancelled_datetime'],format='%m/%d/%Y %H:%M')
    df['cancelled_hour'] = df['cancelled_datetime'].apply(lambda dt: datetime.datetime(dt.year, dt.month, dt.day, dt.hour) if not pd.isnull(dt) else '')
    df['cancelled_day'] = df['cancelled_datetime'].apply(lambda dt: datetime.datetime(dt.year, dt.month, dt.day) if not pd.isnull(dt) else '')
    df['pickup_hour'] = df['pickup_datetime'].apply(lambda dt: datetime.datetime(dt.year, dt.month, dt.day, dt.hour))
    df['pickup_day'] = df['pickup_datetime'].apply(lambda dt: datetime.datetime(dt.year, dt.month, dt.day))
    df['created_hour'] = df['created_datetime'].apply(lambda dt: datetime.datetime(dt.year, dt.month, dt.day, dt.hour) if not pd.isnull(dt) else '')
    df['created_day'] = df['created_datetime'].apply(lambda dt: datetime.datetime(dt.year, dt.month, dt.day) if not pd.isnull(dt) else '')
    df['day_of_week'] = df['pickup_datetime'].dt.weekday_name
    df['weekend'] = (df['day_of_week'].apply(lambda x: x in ['Saturday','Sunday'])).astype(int)
    df['Minute']= df['pickup_datetime'].dt.minute
    df['Day'] = df['pickup_datetime'].dt.day
    df['Hour'] = df['pickup_datetime'].dt.hour
    df['time_of_day'] = 'Night'
    df['time_of_day'][(df['Hour'] >= 5) &
                            (df['Hour'] < 9)] = 'EarlyMorning'
    df['time_of_day'][(df['Hour'] >= 9) &
                            (df['Hour'] < 13)] = 'Morning'
    df['time_of_day'][(df['Hour'] >= 13) &
                            (df['Hour'] < 17)] = 'Afternoon'
    df['time_of_day'][(df['Hour'] >= 17) &
                            (df['Hour'] < 21)] = 'Evening'
    df = remove_test(df)
    df = zip_field(df)
    return df

def remove_test(df):
    df = df.loc[(df['Minute']!=59) | (df['Hour']!=23)]
    df = df.loc[df['patientMedicalId']!='TEST'].reset_index(drop=True)
    return df

def zip_find(cell):
    try:
        lst = re.findall(r"\D(\d{5})\D",cell)
        return lst[len(lst)-1]
    except:
        return None

def zip_field(df):
    df['Zip']=df['fromAddress'].apply(lambda x: zip_find(x))
    df['Zip'].fillna('0',inplace=True)
    df['Zip']=df['Zip'].apply(lambda x: int(x))
    return df

def time_series_avg(row, grouped,time_lag,hd):
    if hd=='days':
        df1 = grouped[(grouped['pickup_day'] < row['pickup_day'] - pd.Timedelta(days=time_lag))]
    else:
        df1 = grouped[(grouped['pickup_hour'] < row['pickup_hour'] - pd.Timedelta(hours=time_lag))]
    return df1['Completed'].sum()/df1['Scheduled'].sum()


def time_series_days_df(df,time_lag):
    df['created_buffer'] = (df['pickup_day']-df['created_day'])/pd.Timedelta(days=1)
    df['cancelled_buffer'] = (df['pickup_day']-df['cancelled_day'])/pd.Timedelta(days=1)
    df['Scheduled'] = 0
    df['Scheduled'][(df['created_buffer']>=time_lag) & (df['cancelled_buffer']<=time_lag)] = 1
    df['Scheduled'][(df['created_buffer']>=time_lag) & (pd.isna(df['cancelled_buffer']))] = 1
    df['Completed'] = 0
    df['Completed'][df['status']=='Completed']=1
    grouped = df.groupby(['pickup_day','weekend']).agg('sum').reset_index().copy()
    grouped['exp_Completed_perc'] = grouped.apply(lambda x: time_series_avg(x, grouped,time_lag,'days'), axis=1)
    grouped['exp_Completed'] = grouped['exp_Completed_perc'] * grouped['Scheduled']
    # grouped['Completed_week_ago'] = grouped['Completed'].shift(periods=7,axis=0)
    # grouped['Completion_%_prev_week']= ((grouped['Completed'].shift(periods=7,axis=0) +
    #                                     grouped['Completed'].shift(periods=6,axis=0) +
    #                                     grouped['Completed'].shift(periods=5,axis=0)) /
    #                                     (grouped['Scheduled'].shift(periods=7,axis=0) +
    #                                     grouped['Scheduled'].shift(periods=6,axis=0) +
    #                                     grouped['Scheduled'].shift(periods=5,axis=0)))

    # grouped['exp_dow'] = ((grouped['Completed'].shift(periods=7,axis=0) /
    #                         grouped['Scheduled'].shift(periods=7,axis=0)) *
    #                         grouped['Scheduled'])
    grouped[f'Completed_p-lag'] = grouped['Completed'].shift(periods=time_lag, axis=0)
    grouped.fillna(0, inplace=True)
    grouped = grouped.iloc[time_lag+1:]
    return grouped

def time_series_hours_df(df, time_lag):
    df['created_buffer'] = (df['pickup_hour']-df['created_hour'])/pd.Timedelta(hours=1)
    df['cancelled_buffer'] = (df['pickup_hour']-df['cancelled_hour'])/pd.Timedelta(hours=1)
    df['Scheduled'] = 0
    df['Scheduled'][(df['created_buffer']>=time_lag) & (df['cancelled_buffer']<=time_lag)] = 1
    df['Scheduled'][(df['created_buffer']>=time_lag) & (pd.isna(df['cancelled_buffer']))] = 1
    df['Completed'] = 0
    df['Completed'][df['status']=='Completed']=1
    grouped = df.groupby(['pickup_hour','pickup_day','weekend','time_of_day']).agg('sum').reset_index()
    # grouped['Completed_week_ago'] = grouped['Completed'].shift(periods=168,axis=0)
    # grouped['Completion_%_prev_week']= ((grouped['Completed'].shift(periods=168,axis=0) +
    #                                     grouped['Completed'].shift(periods=167,axis=0) +
    #                                     grouped['Completed'].shift(periods=166,axis=0))/
    #                                     (grouped['Scheduled'].shift(periods=168,axis=0)+
    #                                     grouped['Scheduled'].shift(periods=167,axis=0)+
    #                                     grouped['Scheduled'].shift(periods=166,axis=0)))
    grouped['exp_Completed_perc'] = grouped.apply(lambda x: time_series_avg(x, grouped,time_lag,'hours'), axis=1)
    grouped['exp_Completed'] = grouped['exp_Completed_perc'] * grouped['Scheduled']
    # grouped['exp_Completed'] = grouped['Completion_%_prev_week'] * grouped['Scheduled']
    # grouped['exp_dow'] = ((grouped['Completed'].shift(periods=168,axis=0) /
    #                         (grouped['Scheduled'].shift(periods=168,axis=0)+1)) *
    #                         grouped['Scheduled'])
    grouped[f'Completed_p-lag'] = grouped['Completed'].shift(periods=time_lag, axis=0)
    grouped.fillna(0, inplace=True)
    grouped = grouped.iloc[336:]
    return grouped


if __name__=='__main__':
    path = '/Users/User/Documents/galvanize/capstones/medicaltransport/data'
    df = get_data(path)
    df.to_pickle('data/completion_df.pkl')
    df = pd.read_pickle('data/completion_df.pkl')
    days_df = time_series_days_df(df,7)
    days_df.to_pickle('data/days_df_7_lag.pkl')
    hours_df = time_series_hours_df(df,24)
    hours_df.to_pickle('data/hours_df_24_lag.pkl')
