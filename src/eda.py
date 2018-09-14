import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import datetime
import glob as glob
import plotly.plotly as py
import plotly.tools as tls
import plotly.offline as plo
import os
import matplotlib as mpl
mpl.rcParams.update({
    'font.size'           : 20.0,
    'axes.titlesize'      : 'large',
    'axes.labelsize'      : 'medium',
    'xtick.labelsize'     : 'medium',
    'ytick.labelsize'     : 'medium',
    'legend.fontsize'     : 'large',
})
apikey = os.environ.get('PLOTLY_API_KEY')
tls.set_credentials_file(username='timmarlowe', api_key=apikey)
pd.options.mode.chained_assignment = None

def simple_hist(ax,x,num_bins,title, x_label,cumulative=False):
    ax.hist(x,bins=num_bins,cumulative=cumulative)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    return ax

def pivot(df,index, column, values, agg):
    group = df.groupby([index,column]).agg(agg).reset_index()
    pivot = group.pivot(index=index, columns=column, values=values)
    return pivot

def cancelled_completed(df,lst,save,title):
    fig, ax = plt.subplots(figsize=(12,6))
    for i in lst:
        ax.plot(df[i])
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Rides')
    plt.legend()
    plt.savefig('images/{}.png'.format(save),dpi=500)
    # plotly_fig = tls.mpl_to_plotly( fig )
    # plot_url = py.plot(plotly_fig, filename=save, auto_open=False)
    plt.close()

def simple_timeline(ax,y,xlabel,ylabel,title):
    ax.plot(y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return ax

def simple_scatter(x,y,xlabel,ylabel,title,save):
    fig, ax = plt.subplots(1,1,figsize=(8,8))
    plt.scatter(x,y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(save,dpi=500)
    plt.close

def filter_df(df,rows,columns,other_row):
    if other_row:
        df.loc['Other'] = df.sum(axis=0) - df.loc[rows].sum(axis=0)
        rows.append('Other')
        return df.loc[rows,columns]
    else:
        return df.loc[rows,columns]


def stacked_bar_side_by_side(df,num,cols,cat,save,other_row=True,column_total=False):
    if column_total:
        df['total']=df[cols].sum(axis=1)
    else:
        df['total']=df.sum(axis=1)
    df.sort_values(by='total',ascending=False,inplace=True)
    df_filtered = filter_df(df,list(df.index.values[:num]),cols,other_row)
    pct_df = df_filtered.div(df_filtered.sum(1), axis = 0)
    # import pdb; pdb.set_trace()
    dflist = [df_filtered, pct_df]
    fig,axes = plt.subplots(1,2,figsize=(16,8))
    counter=0
    for df, ax in zip(dflist,axes):
        if counter == 1:
            legend=False
            df.plot(ax=ax,kind='bar',stacked=True, legend=legend)
            ax.set_title('Percent of All Rides of {}'.format(cat))
            ax.set_ylabel('% of Rides')
        else:
            legend=True
            df.plot(ax=ax,kind='bar',stacked=True, legend=legend)
            ax.set_title('Total Number of Rides')
            ax.set_ylabel('Total Rides')
        if cat == 'patientMedicalId':
            ax.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                labelbottom=False)
        counter +=1
    # plt.suptitle('{} and Cancellation Rates'.format(cat),fontsize=18)
    plt.tight_layout()
    plt.savefig(save,dpi=500)
    plt.close()

if __name__ == '__main__':

    plt.style.use('ggplot')
    cleaned = pd.read_pickle('data/rides_clean.pkl')
    cleaned = cleaned[cleaned['pickup_day'] < pd.datetime(2018,9,1)]
    status_pivot = pd.pivot_table(cleaned, index='status',values='rideId',aggfunc='count')
    status_pivot.sort_values(by='rideId',ascending=False)
    print(status_pivot)

    #Creation of Expected DB - for calculations of actual vs. expected
    expected = cleaned.loc[cleaned['status_v2'].isin(['Completed','CancelledDayOf','CancelledHourOf'])]
    expected['expected_status']=expected['status_v2']
    expected['expected_status'][(expected['create_hours_cat'].isin(['CreatedDayOf','CreatedHourOf']))
                                & (expected['status_v2'].isin(['CancelledDayOf','CancelledHourOf']))] = 'CreatedCancelledDayOf'
    expected['expected_status'][expected['expected_status'].isin(['CancelledDayOf','CancelledHourOf'])] = 'CreatedEarlyCancelledDayOf'
    expected['expected_status'][(expected['status_v2'] == 'Completed') & (expected['create_hours_cat'].isin(['CreatedDayOf','CreatedHourOf']))] = 'CreatedDayOfCompleted'
    expected['expected_status'][expected['expected_status']=='Completed'] = 'CreatedEarlyCompleted'

    expected_pivot = pd.pivot_table(expected,index='expected_status',values='rideId',aggfunc='count')
    expected_pivot.sort_values(by='rideId',ascending=False)
    expected_pivot['daily_avg']=round(expected_pivot['rideId']/92,1)
    print(expected_pivot)

    #Datetime explorations
    #Plotting Histograms for Day, Hour, Minute
    fig, axes = plt.subplots(1,2,figsize=(16,6))
    simple_hist(axes[0],cleaned['Hour'],24,'Hour of Scheduled Pickup by Vendor','Hour of Pickup')
    simple_hist(axes[1],cleaned['Minute'],60,'Minute of Scheduled Pickup by Vendor','Minute of Pickup')
    plt.savefig('images/time_hists.png',dpi=500)
    plt.close()

    #Creating qhdf pivot
    qh_df = pivot(cleaned,'pickup_quarter_hour','status','rideId','count')
    h_df = pivot(cleaned,'pickup_hour','status','rideId','count')
    d_df = pivot(cleaned,'pickup_day','status','rideId','count')

    #Plotting qh_df for cancelled and completed
    cancelled_completed(qh_df,['Completed','Cancelled'],'quarterhour_rides','Cancelled and Completed Rides per Quarter-Hour')
    #Plotting qh_df for cancelled and completed
    cancelled_completed(h_df,['Completed','Cancelled'],'hour_rides','Cancelled and Completed Rides per Hour')
    #plotting d_df for cancelled and completed
    cancelled_completed(d_df,['Completed','Cancelled'],'day_rides','Cancelled and Completed Rides per Day')


    h_df2 = pivot(cleaned,'pickup_hour','status_v2','rideId','count')
    cancelled_completed(h_df2,['Completed','CancelledDayOf','CancelledHourOf'],'hour_rides_late_cancel','Cancelled and Completed Rides per Hour by Cancellation Timing')
    d_df2 = pivot(cleaned,'pickup_day','status_v2','rideId','count')
    cancelled_completed(d_df2,['Completed','CancelledDayOf','CancelledHourOf'],'day_rides_late_cancel', 'Cancelled and Completed Rides per Day by Cancellation Timing')

    d_df2['LateCancelled']=d_df2['CancelledDayOf']+d_df2['CancelledHourOf']
    cancelled_completed(d_df2,['Completed','LateCancelled'],'day_rides_all_late_cancel', 'Completed and Late-Cancelled Rides by Day')

    h_df3 = pivot(expected,'pickup_hour', 'expected_status','rideId','count')
    cancelled_completed(h_df3,['CreatedEarlyCompleted','CreatedEarlyCancelledDayOf','CreatedDayOfCompleted'],'hour_rides_cancel_ondemand','Completed v. Expected Hourly Rides')
    d_df3 = pivot(expected,'pickup_day', 'expected_status','rideId','count')
    cancelled_completed(d_df3,['CreatedEarlyCompleted','CreatedEarlyCancelledDayOf','CreatedDayOfCompleted','CreatedCancelledDayOf'],'day_rides_cancel_ondemand','Completed v. Expected Daily Rides')
    d_df3['Expected']=d_df3['CreatedEarlyCompleted']+d_df3['CreatedEarlyCancelledDayOf']
    d_df3['Actual']=d_df3['CreatedEarlyCompleted']+d_df3['CreatedDayOfCompleted']
    cancelled_completed(d_df3,['Expected','Actual'],'day_rides_expected_actual','Expected v. Actual Rides Provided')

    tot_pivot = pd.pivot_table(cleaned,index='status_v2',values='rideId',aggfunc='count')
    tot_pivot.sort_values(by='rideId',inplace=True,ascending=False)
    x=np.arange(4)
    fig, ax = plt.subplots(1,1,figsize=(12,6))
    plt.bar(x,tot_pivot['rideId'][:4])
    plt.title('Total Rides by Cancellation Timing')
    plt.xticks(x,tot_pivot.index.values[:4])
    plt.savefig('images/totalrides.png',dpi=500)
    plt.close()

    #Percent Cancelled
    df_lst = [h_df2, d_df2]
    labels = ['Hourly','Daily']
    for df, label in zip(df_lst,labels):
        df.fillna(0,inplace=True)
        df['Total Rides Scheduled'] = df[['CancelledDayOf','CancelledHourOf','Completed']].sum(axis=1)
        df['pct_late_cancelled']=round((df['CancelledDayOf']+df['CancelledHourOf'])/df['Total Rides Scheduled'],3)
        fig, ax = plt.subplots(1,1,figsize=(12,6))
        simple_timeline(ax,df['pct_late_cancelled'],'Date','Percent of Rides Late-Cancelled','{} Percent of Rides Late-Cancelled Over Time'.format(label))
        plt.savefig('images/{}_pct_late_cancelled.png'.format(label),dpi=500)
        # plotly_fig = tls.mpl_to_plotly( fig )
        # plot_url = py.plot(plotly_fig, filename='{}_pct_cancelled'.format(label), auto_open=False)
        plt.close()

    #Plot daily rides and cancellation rate on same graph
    fig1, ax1 = plt.subplots(1,1,figsize=(12,6))
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Total Rides Scheduled (As of Date)', color=color)
    ax1.plot(d_df2['Total Rides Scheduled'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    color = 'tab:blue'
    ax2.set_ylabel('Percent Late-Cancelled', color=color)
    ax2.plot(d_df2['pct_late_cancelled'], color=color)
    ax2.grid(False)
    ax2.tick_params(axis='y', labelcolor=color)
    # fig1.tight_layout()
    plt.title('Total Rides Requested v. Percent Cancelled')
    plt.figlegend(loc=1)
    plt.savefig('images/daily_rides_v_cancellations.png',dpi=500)
    plt.close()

    #Cancelled date/time vs pickup datetime
    cancelled = cleaned[(cleaned['status']=='Cancelled') | (cleaned['status']=='DriverCancelled')]
    fig, axes=plt.subplots(1,3,figsize=(12,6))
    bins=[50,48,40]
    ranges = [(-1000,50),(-24,24),(-2,2)]
    for ax,bin,range in zip(axes,bins,ranges):
        ax.hist(cancelled['hours_warning'],bins=bin,range=range)
        ax.axvline(x=0,color="#000099")
        ax.set_xlabel('Hours')
    plt.suptitle('Cancellation Time - Pickup Time (Hours)')
    plt.savefig('images/cancellation_warning.png',dpi=500)
    plt.close()

    #Cancelled by Time of Day
    tod_df = pivot(cleaned,'time_of_day','status_v2','rideId','count')
    stacked_bar_side_by_side(tod_df,tod_df.shape[0],['Completed','CancelledDayOf','CancelledHourOf'],'Pickup Time of Day','images/timing_pickup_time.png',False, column_total=True)

    #Cancelled by Time of Day
    dow_df = pivot(cleaned,'day_of_week','status_v2','rideId','count')
    stacked_bar_side_by_side(dow_df,dow_df.shape[0],['Completed','CancelledDayOf','CancelledHourOf'],'Pickup Day of Week','images/timing_pickup_day.png',False, column_total=True)

    #Cancellation by Vehicle Type - by status
    vehicle_df1 = pivot(cleaned,'requestedVehicleType','status','rideId','count')
    stacked_bar_side_by_side(vehicle_df1,4,['Completed','Cancelled','DriverCancelled'],'Vehicle Type','images/status_vehicle_type.png')
    #Cancellation by Vehicle Type - by timing of cancellation
    vehicle_df2 = pivot(cleaned,'requestedVehicleType','status_v2','rideId','count')
    stacked_bar_side_by_side(vehicle_df2,4,['Completed','CancelledDayOf','CancelledHourOf'],'Vehicle Type','images/timing_vehicle_type.png',column_total=True)

    #Cancellation by Vehicle Company Name - by status
    company_df1 = pivot(cleaned,'vehicleCompanyName','status','rideId','count')
    stacked_bar_side_by_side(company_df1,10,['Completed','Cancelled','DriverCancelled'],'Company','images/status_company_name.png')
    #Cancellation by Vehicle Company Name - by timing of cancellation
    company_df2 = pivot(cleaned,'vehicleCompanyName','status_v2','rideId','count')
    stacked_bar_side_by_side(company_df2,10,['Completed','CancelledDayOf','CancelledHourOf'],'Company','images/timing_company_name.png',column_total=True)

    #Cancellation by User
    user_df1 = pivot(cleaned,'patientMedicalId','status','rideId','count')
    user_df1.fillna(0,inplace=True)
    stacked_bar_side_by_side(user_df1,50,['Completed','Cancelled','DriverCancelled'],'patientMedicalId','images/status_user.png',False)
    user_df1['CancelRate'] = user_df1['Cancelled']/user_df1['total']
    simple_scatter(user_df1['total'],user_df1['CancelRate'],'Total Requests','Cancellation Rate','Cancellation Rate vs. Total Requests','images/cancel_v_requests.png')

    #Cancellation by User - by timing of cancellation
    user_df2 = pivot(cleaned,'patientMedicalId','status_v2','rideId','count')
    user_df2.fillna(0,inplace=True)
    stacked_bar_side_by_side(user_df2,50,['Completed','CancelledDayOf','CancelledHourOf'],'patientMedicalId','images/timing_user.png',False,column_total=True)
    user_df2['LateCancelRate'] = (user_df2['CancelledDayOf'] + user_df2['CancelledHourOf'])/user_df2['total']
    user_df2['LateCancelRate'].fillna(0,inplace=True)
    simple_scatter(user_df2['total'],user_df2['LateCancelRate'],'Total Requests','Late Cancellation Rate','Late Cancellation Rate vs. Total Requests by Passenger','images/latecancel_v_requests.png')

    #Histogram of Late Cancellation Rate
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    simple_hist(ax,user_df2['LateCancelRate'],50,'Late Cancellation Rate (Day-of) by Passenger','Late Cancellation Rate')
    plt.savefig('images/cancelrate_hist.png',dpi=500)
    plt.close()

    #Zip Code Analysis
    zip_df = pivot(cleaned,'Zip','status_v2','rideId','count')
    stacked_bar_side_by_side(zip_df,20,['Completed','CancelledDayOf','CancelledHourOf'],'Zip Code','images/status_zip_code.png',column_total=True)

    #Created Buffer Analysis
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    simple_hist(ax,cleaned['created_hours_buffer'],50,'Hours Between Appt. Created and Pickup','Created Time Minus Pickup Time (Hours)')
    plt.savefig('images/createdbuffer_hist.png',dpi=500)
    plt.close()
    created_df = pivot(cleaned,'create_hours_cat','status_v2','rideId','count')
    stacked_bar_side_by_side(created_df,created_df.shape[0],['Completed','CancelledEarly','CancelledDayOf','CancelledHourOf'],'Created Time Category','images/status_createdtime.png',False,column_total=True)

    #On-Demand Type Analysis
    ondemand_df1 = pivot(cleaned,'create_hours_cat','on_demand','rideId','count')
    stacked_bar_side_by_side(ondemand_df1,ondemand_df1.shape[0],[0,1],'On-Demand Category','images/createdtime_v_ondemand.png',False,column_total=True)
    ondemand_df2 = pivot(cleaned, 'on_demand','status_v2','rideId','count')
    stacked_bar_side_by_side(ondemand_df2, ondemand_df2.shape[0],['Completed','CancelledEarly','CancelledDayOf','CancelledHourOf'],'On Demand Status','images/status_ondemand.png',False,column_total=True)

    #Cancelled reason type analysis
    cancelled_reasondf1 = pivot(cleaned,'cancelledReasonType','status','rideId','count')
    stacked_bar_side_by_side(cancelled_reasondf1,cancelled_reasondf1.shape[0],['Completed','Cancelled','DriverCancelled'],'Cancelled Reason Type','images/status_cancelled.png',False,column_total=True)
    cancelled_reasondf2 = pivot(cleaned,'cancelledReasonType','status_v2','rideId','count')
    stacked_bar_side_by_side(cancelled_reasondf2,cancelled_reasondf2.shape[0],['Completed','CancelledEarly','CancelledDayOf','CancelledHourOf'],'Cancelled Reason Type','images/status_cancelled2.png',False,column_total=True)
