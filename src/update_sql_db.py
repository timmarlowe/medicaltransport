import pickle
import requests
import json
import pandas as pd
import numpy as np
import time
from datetime import datetime
import psycopg2
from psycopg2.extensions import AsIs
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from cancel_mod_final import get_data, CancelModel
from complete_mod_final import get_completion_data, CompletionModel

def create_datatable(df, table_name,engine):
    df.to_sql(table_name,engine,if_exists='replace')
    return

def return_rows(db,table_name, datetime,host):
    conn = psycopg2.connect(f"host='{host}' dbname='{db}' user='User' password = 'listofdicts'")
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute(f"SELECT * FROM {table_name} WHERE pickup_day = timestamp '{datetime}';")
    colnames = [desc[0] for desc in cur.description]
    data = cur.fetchall()
    df = pd.DataFrame(data, columns = colnames)
    df.set_index('index',inplace=True)
    return df

def append_rows(df,table_name, engine,db,host):
    conn = psycopg2.connect(f"host='{host}' dbname='{db}' user='User' password = 'listofdicts'")
    df.to_sql(table_name,engine,if_exists='append')
    conn.commit()
    return

def drop_rows(db, table_name, datetime,host):
    conn = psycopg2.connect(f"host='{host}' dbname='{db}' user='User' password = 'listofdicts'")
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute(f"DELETE FROM {table_name} WHERE pickup_day >= timestamp '{datetime}';")
    rows_deleted = cur.rowcount
    conn.commit()
    print(f'Dropped {rows_deleted} rows from table')

if __name__=='__main__':
    #Loading Cancellation Model
    with open('data/cancel_model.pkl', 'rb') as f1:
        cancel_model = pickle.load(f1)

    #Loading Completion Model
    with open('data/complete_model.pkl', 'rb') as f2:
        complete_model = pickle.load(f2)

    #Initiating date for start of db
    day = 4
    datetime = pd.datetime(2018,7,day)

    #Engine for sql alchemy
    engine_db = create_engine('postgresql://User:listofdicts@localhost:5432/rides_db')

    #Cancellation DB
    canceldf = pd.read_pickle('data/model_df.pkl')
    cancelpreds = canceldf[canceldf['pickup_datetime'] < datetime]
    X,y = get_data(cancelpreds.copy())
    cancelpreds['prediction']=cancel_model.predict_proba(X).round(3)
    cancelpreds['outcome']= y
    cancelfeed = canceldf[canceldf['pickup_datetime'] >= datetime]
    print('Creating ride feed datatable')
    create_datatable(cancelfeed,'ride_feed',engine_db)
    print('Creating ride pred datatable')
    create_datatable(cancelpreds,'ride_preds',engine_db)
    cancelpreds = None
    cancelfeed = None
    canceldf = None

    #Completion DB
    completiondf = pd.read_pickle('data/hours_df_24_lag.pkl')
    completionpreds = completiondf[completiondf['pickup_day']<datetime]
    completionfeed = completiondf[completiondf['pickup_day']>=datetime]
    X,y = get_completion_data(completionpreds.copy())
    X.drop('pickup_hour',axis=1, inplace=True)
    completionpreds['prediction']=complete_model.predict(X)
    completionpreds['outcome']=y
    print('Creating completion feed datatable')
    create_datatable(completionfeed,'completion_feed',engine_db)
    print('Creating completion pred datatable')
    create_datatable(completionpreds,'completion_preds',engine_db)
    completiondf = None
    completionpreds = None
    completionfeed = None

    #Predict, append and return rows for both cancellation and completion db
    while True:
        drop_rows('rides_db','ride_preds',str(datetime),'localhost')
        drop_rows('rides_db','completion_preds',str(datetime),'localhost')
        while datetime < pd.datetime(2018,7,14):
            #Update cancellation datatable
            canceldata = return_rows('rides_db','ride_feed', str(datetime),'localhost')
            X,y = get_data(canceldata.copy())
            canceldata['prediction']=cancel_model.predict_proba(X).round(3)
            canceldata['outcome'] = y
            append_rows(canceldata,'ride_preds',engine_db,'rides_db','localhost')
            print(f'July {day}: {canceldata.shape[0]} rows appended to Cancellation table')

            #Update completion datatable
            completedata = return_rows('rides_db','completion_feed', str(datetime),'localhost')
            X,y = get_completion_data(completedata.copy())
            X.drop('pickup_hour',axis=1, inplace=True)
            completedata['prediction']=complete_model.predict(X)
            completedata['outcome'] = y
            append_rows(completedata, 'completion_preds',engine_db,'rides_db','localhost')
            print(f'July {day}: {completedata.shape[0]} rows appended to Completion table')


            day+=1
            datetime = pd.datetime(2018,7,day)
            time.sleep(30)
        day = 4
        datetime = pd.datetime(2018,7,day)
