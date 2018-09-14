import pandas as pd
import numpy as np
import glob
import re
import math
import datetime

def distance(lat1, lon1, lat2, lon2):
    radius = 3959

    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c

    return d

def zip_find(cell):
    try:
        lst = re.findall(r"\D(\d{5})\D",cell)
        return lst[len(lst)-1]
    except:
        return None

#Import Data and rename
path =r'/Users/User/Documents/galvanize/capstones/medicaltransport/data'
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

#Drop columns for fname and lname
df.drop(['patientFirstName', 'patientLastName','saferideFee', 'creditCardFee',
        'passengerCost','careCost','vehiclePlateNumber','lyftRideId',
        'lyftReferenceCode','ExtraTime','hospitalPhoneNumber',
        'patientPhoneNumber'], axis=1, inplace=True)

#Creating Day, hour, minute vars for pickup
df['Minute']= df['pickup_datetime'].dt.minute
df['Hour'] = df['pickup_datetime'].dt.hour
df['Day'] = df['pickup_datetime'].dt.day
df['Zip']=df['fromAddress'].apply(lambda x: zip_find(x))
df['Zip'].fillna('0',inplace=True)
df['Zip']=df['Zip'].apply(lambda x: int(x))

df.to_pickle('data/rides.pkl')

#Removing test rides at 11:59 PM
interim = df.loc[(df['Minute']!=59) | (df['Hour']!=23)]
#Replacing values in interim
interim.replace('nan nan',np.nan,inplace=True)
#Creating new cleaned df
cleaned = interim.loc[interim['patientMedicalId']!='TEST'].reset_index(drop=True)
#Reformatting cancelled_datetime
cleaned['cancelled_datetime'] = pd.to_datetime(cleaned['cancelled_datetime'],format='%m/%d/%Y %H:%M')
#Looking at and transforming hours of buffer
cleaned['created_hours_buffer'] = (cleaned['created_datetime']-cleaned['pickup_datetime'])/pd.Timedelta(hours=1)
cleaned['create_hours_cat'] = 'CreatedEarly'
cleaned['create_hours_cat'][(cleaned['created_datetime'].dt.day ==
                            cleaned['pickup_datetime'].dt.day) &
                            (cleaned['created_datetime'].dt.month ==
                            cleaned['pickup_datetime'].dt.month)] = 'CreatedDayOf'
cleaned['create_hours_cat'][cleaned['created_hours_buffer'] >= -1] = 'CreatedHourOf'

#Replacing values in ondemand
cleaned['rideType'].replace(to_replace='ondemand',value='on-demand',inplace=True)
cleaned['on_demand'] = (cleaned['rideType'] == 'on-demand').astype(int)
cleaned.drop('rideType',axis=1,inplace=True)
cleaned['hours_warning'] = (cleaned['cancelled_datetime']-cleaned['pickup_datetime'])/pd.Timedelta(hours=1)
#Creating status_v2 with timing of cancellation
cleaned['status_v2'] = cleaned['status']
cleaned['status_v2'][cleaned['status']=='Cancelled'] = 'CancelledEarly'
cleaned['status_v2'][(cleaned['cancelled_datetime'].dt.day ==
                            cleaned['pickup_datetime'].dt.day) &
                            (cleaned['cancelled_datetime'].dt.month ==
                            cleaned['pickup_datetime'].dt.month)] = 'CancelledDayOf'
cleaned['status_v2'][cleaned['hours_warning'] >= -1] = 'CancelledHourOf'
#Creating ride distance (haversine - point to point, not manhattan)
cleaned['to_from_distance']=cleaned.apply(lambda row: distance(row['fromLatitude'],
                                                            row['fromLongitude'],
                                                            row['toLatitude'],
                                                            row['toLongitude']),axis=1)
#Creating distance from center city (haversine - point to point, not manhattan)
cleaned['center_city_distance']= cleaned.apply(lambda row: distance(row['fromLatitude'],
                                                            row['fromLongitude'],
                                                            45.5122, -122.6587),axis=1)
#Creating day of week variable
cleaned['day_of_week'] = cleaned['pickup_datetime'].dt.weekday_name
cleaned['weekend'] = (cleaned['day_of_week'].apply(lambda x: x in ['Saturday','Sunday'])).astype(int)

#Grouping by time periods
cleaned['pickup_quarter_hour'] = cleaned['pickup_datetime'].apply(lambda dt: datetime.datetime(dt.year, dt.month, dt.day, dt.hour,15*(dt.minute // 15)))
cleaned['pickup_hour'] = cleaned['pickup_datetime'].apply(lambda dt: datetime.datetime(dt.year, dt.month, dt.day, dt.hour))
cleaned['pickup_day'] = cleaned['pickup_datetime'].apply(lambda dt: datetime.datetime(dt.year, dt.month, dt.day))

#Grouping by Time of Day
cleaned['time_of_day'] = 'Night'
cleaned['time_of_day'][(cleaned['Hour'] >= 5) &
                        (cleaned['Hour'] < 9)] = 'EarlyMorning'
cleaned['time_of_day'][(cleaned['Hour'] >= 9) &
                        (cleaned['Hour'] < 13)] = 'Morning'
cleaned['time_of_day'][(cleaned['Hour'] >= 13) &
                        (cleaned['Hour'] < 17)] = 'Afternoon'
cleaned['time_of_day'][(cleaned['Hour'] >= 17) &
                        (cleaned['Hour'] < 21)] = 'Evening'
cleaned = cleaned[cleaned['pickup_day'] < pd.datetime(2018,9,1)]

cleaned.to_pickle('data/rides_clean.pkl')
########## This is EDA Dataframe ###################
