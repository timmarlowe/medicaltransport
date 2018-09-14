
from flask import Flask, request, flash, redirect, url_for, render_template
import requests
from werkzeug.utils import secure_filename
import numpy as np
import sys
import os
import pdb
import pandas as pd
import pandas.io.sql as sqlio
import psycopg2
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import datetime

app = Flask(__name__,static_folder="static")
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Pull from API and convert data to pandas dataframe

def get_cancel_df():
    conn = psycopg2.connect(database='rides_db')
    sql_query = '''
            SELECT
                pickup_datetime,
                pickup_day,
                pickup_hour
                distance,
                "fromLatitude",
                "fromLongitude",
                "requestedVehicleType",
                "RideRequestReason",
                prediction,
                status,
                outcome
            FROM
                ride_preds
            WHERE
                pickup_day = (SELECT MAX(pickup_day) FROM ride_preds)
            ORDER BY
                prediction DESC;
    '''
    rides_df = sqlio.read_sql_query(sql_query, conn)
    conn = None
    return rides_df

def get_completion_df():
    conn = psycopg2.connect(database='rides_db')
    sql_query = '''
            SELECT
                pickup_hour,
                pickup_day,
                prediction,
                outcome
            FROM
                completion_preds
            ORDER BY
                pickup_hour ASC;
    '''
    completion_df = sqlio.read_sql_query(sql_query, conn)
    conn = None
    return completion_df

def define_risk_level(df):
    high_mask = df['prediction'] >= .7
    med_mask = (df['prediction'] >= .4) & (df['prediction'] < .7)
    low_mask = df['prediction'] < .4
    df.loc[high_mask, 'risk_level'] = 'High'
    df.loc[med_mask, 'risk_level'] = 'Moderate'
    df.loc[low_mask, 'risk_level'] = 'Low'
    return df

def plot_preds_html(y_test, preds,x_label,day):
    fig, ax = plt.subplots(1,1,figsize=(12,8))
    ax.plot(y_test[-48:-24],label='Actual Rides Given (Yesterday)')
    ax.plot(preds[-48:], label='Predicted Capacity Needed')
    ax.legend()
    ax.set_title('Predicted Capacity Needed for Ride Service')
    ax.set_xticks(y_test.index[-48::2])
    ax.set_xticklabels(x_label[-48::2],rotation=45,ha='right')
    plt.tight_layout()
    saved = f'static/images/predicted_rides_{day}.png'
    plt.savefig(saved,dpi=500)
    plt.close()
    return saved


@app.route('/')
def index():
    #Cancellation Data
    cancel_data = get_cancel_df()
    pickup_day = str(np.max(cancel_data['pickup_day']))[:10]
    timestamp = np.max(cancel_data['pickup_day'])
    cancel_data = define_risk_level(cancel_data)
    counts = cancel_data['risk_level'].value_counts()
    low_risk = counts['Low']
    moderate_risk = counts['Moderate']
    high_risk = counts['High']
    daily_rides = low_risk + moderate_risk + high_risk
    cols = ['pickup_datetime','requestedVehicleType','RideRequestReason','prediction','risk_level','status']
    cancel_data = cancel_data[cols]

    return render_template('index.html', table = cancel_data, pickup_day = pickup_day,
                            daily_rides = daily_rides, low_risk = low_risk,
                            moderate_risk = moderate_risk, high_risk = high_risk,
                            timestamp = timestamp)

@app.route('/utilization', methods=['GET'])
def utilization():
    #Completion Data
    cancel_data = get_cancel_df()
    count = cancel_data.agg('count')[0]
    completion_data = get_completion_df()
    comp_pickup_day = str(np.max(completion_data['pickup_day']))[:10]
    timestamp = np.max(completion_data['pickup_day'])
    grouped = completion_data.groupby('pickup_day').agg('sum')
    predicted_rides = int(round(grouped.iloc[-1]['prediction'],0))
    pickup_hours = completion_data['pickup_hour']
    pickup_hours = pickup_hours.apply(lambda x: x.strftime('%h %d at %H:00'))
    saved = plot_preds_html(completion_data['outcome'],completion_data['prediction'],pickup_hours,comp_pickup_day)

    return render_template('utilization.html', predicted_rides = predicted_rides,
                            timestamp = timestamp, saved=saved, pickup_day = comp_pickup_day,
                            count = count)


@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True, debug=False)
