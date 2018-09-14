import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import folium
from folium import plugins
from folium import features
import datetime

def heat_map(df,to_from,save,gradient=None):
    map_metro = folium.Map(location=[45.5122, -122.6587],zoom_start = 11,max_zoom=14, tiles='Stamen', attr='Toner')
    heat_df = df[['{}Latitude'.format(to_from), '{}Longitude'.format(to_from),'pickup_day']]
    heat_df.dropna(inplace=True)
    heat_array = [[row['{}Latitude'.format(to_from)],row['{}Longitude'.format(to_from)]] for index, row in heat_df.iterrows()]
    if gradient is None:
        hm = plugins.HeatMap(heat_array, radius=10)
    else:
        hm = plugins.HeatMap(heat_array, radius=10,gradient=gradient)
    hm.add_to(map_metro)
    map_metro.save(save)

def heat_map_with_time(df,to_from,by,data_index,save):
    map_metro = folium.Map(location=[45.5122, -122.6587],zoom_start = 11,max_zoom=14,tiles='Stamen', attr='Toner')
    heat_df = df[['{}Latitude'.format(to_from), '{}Longitude'.format(to_from),by]]
    heat_df.dropna(inplace=True)
    heat_array_time = [[[row['fromLatitude'],row['fromLongitude']] for index, row in heat_df[heat_df[by] == i].iterrows()] for i in data_index]
    if data_index == pickup:
        idx = [str(i).split(' ')[0] for i in data_index]
    else:
        idx = data_index
    hmt = plugins.HeatMapWithTime(heat_array_time, index=idx, auto_play=False, radius=10)
    hmt.add_to(map_metro)
    map_metro.save(save)

def zip_code_map(df,columns,legend_name,save,fill_color):
    map_metro = folium.Map(location=[45.5122, -122.6587],zoom_start = 9,max_zoom=14, tiles='Stamen', attr='Toner')
    zip_code = 'data/Zip_Code_Boundaries.geojson'
    map_metro.choropleth(
        geo_data=zip_code,
        name='choropleth',
        data=df,
        columns=columns,
        key_on='feature.properties.ZIPCODE',
        fill_color= fill_color,
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=legend_name,
        highlight=True
        )
    map_metro.save(save)

def pivot(df,index, column, values, agg):
    group = df.groupby([index,column]).agg(agg).reset_index()
    pivot = group.pivot(index=index, columns=column, values=values)
    return pivot

if __name__ == '__main__':
    #Creating necessary DFs
    cleaned = pd.read_pickle('data/rides_clean.pkl')
    cleaned['pickup_day'] = cleaned['pickup_datetime'].apply(lambda dt: datetime.datetime(dt.year, dt.month, dt.day))
    pickup_day = list(set(cleaned['pickup_day']))
    pickup_day = [i.to_pydatetime() for i in pickup_day]
    pickup = sorted(pickup_day)
    picks = [str(i).split(' ')[0] for i in pickup]
    hour = list(set(cleaned['Hour']))
    hour = sorted(hour)
    completed = cleaned[cleaned['status']=='Completed']
    cancelled = cleaned[(cleaned['status_v2']=='CancelledDayOf') | (cleaned['status_v2']=='CancelledHourOf')]

    #Mapping Overall Rides Completed and over days and over hours of day
    heat_map(completed,'from','images/heat_map_completed.html',gradient={.4: 'blue', .65: 'lime', 1: 'pink'})
    heat_map_with_time(completed,'from','pickup_day',pickup,'images/heat_map_days_completed.html')
    heat_map_with_time(completed,'from','Hour',hour,'images/heat_map_hours_completed.html')

    #Mapping Overall Rides Cancelled and over time
    heat_map(cancelled,'from','images/heat_map_cancelled.html')
    heat_map_with_time(cancelled,'from','pickup_day',pickup,'images/heat_map_days_cancelled.html')
    heat_map_with_time(cancelled,'from','Hour',hour,'images/heat_map_hours_cancelled.html')

    df_zip = pivot(cleaned,'Zip','status_v2','rideId','count')
    df_zip.fillna(0,inplace=True)
    df_zip['AvgRidesPerDay'] = df_zip['Completed']/len(pickup)
    df_zip['TotalRidesScheduled'] = df_zip[['CancelledDayOf','CancelledHourOf','Completed']].sum(axis=1)
    df_zip['LateCancelledRate']=(df_zip['CancelledDayOf']+df_zip['CancelledHourOf'])/df_zip['TotalRidesScheduled']
    df_zip.reset_index(inplace=True)
    df_zip.dropna(inplace=True)
    zip_code_map(df_zip,['Zip','LateCancelledRate'],'Late Cancellation Rate (%)','images/zip_cloro_cancel.html','YlOrRd')
    zip_code_map(df_zip,['Zip','Completed'],'Completed Rides from Zip Code','images/zip_cloro_complete.html','YlGn')
    zip_code_map(df_zip,['Zip','AvgRidesPerDay'],'Average Daily Rides from Zip Code','images/zip_cloro_avgrides.html','Blues')
