#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import datetime

pd.options.display.max_rows = 1000
pd.options.display.max_columns = 1000


# In[3]:


airlines = pd.read_csv('/Users/Jesse/Desktop/DS4A_workspace/Final 11/airlines.csv')
airports = pd.read_csv('/Users/Jesse/Desktop/DS4A_workspace/Final 11/airports.csv', encoding='latin-1')
events = pd.read_csv('/Users/Jesse/Desktop/DS4A_workspace/Final 11/events_US.csv', encoding='latin-1')      
flight_traffic = pd.read_csv('/Users/Jesse/Desktop/DS4A_workspace/Final 11/flight_traffic.csv')
stock_prices = pd.read_csv('/Users/Jesse/Desktop/DS4A_workspace/Final 11/stock_prices.csv')
weather = pd.read_csv('/Users/Jesse/Desktop/DS4A_workspace/Final 11/weather.csv')


# In[4]:


#format dates
dates=pd.to_datetime(flight_traffic[['year', 'month', 'day']])
flight_traffic.insert(0, 'date', dates)
flight_traffic = flight_traffic.drop(columns=['year','month','day'])
flight_traffic['date'] = flight_traffic['date'].apply(lambda x:x.date())


# In[5]:


#format other dates
weather['datetime'] = pd.to_datetime(weather['datetime'], format='%Y%m%dT%H:%M:%S')
weather['date']=weather['datetime'].apply(lambda x: x.date())
weather['time']=weather['datetime'].apply(lambda x: x.time())


# In[6]:


#format HHMM timestamps
time_fields = ['scheduled_departure','actual_departure','wheels_off','wheels_on','scheduled_arrival','actual_arrival']
for time_field in time_fields:
    flight_traffic[time_field] = flight_traffic[time_field].fillna(0).astype('int32').astype(str).    apply(lambda x: x.rjust(4,'0')).apply(lambda x: x[0:5])
    flight_traffic[time_field] = flight_traffic[time_field].apply(lambda x: '2359' if x=='2400' else x)
    flight_traffic[time_field] = pd.to_datetime(flight_traffic[time_field], format= "%H%M")
    flight_traffic[time_field] = flight_traffic[time_field].apply(lambda x: x.time())


# In[7]:


#get a list of airlines that have stock information
relevant_airlines = []
for airline in stock_prices.columns:
    if airline != "timestamp":
        relevant_airlines.append(airline)
relevant_airlines


# In[8]:


#pull out data relevant to these airlines
airlines = airlines[airlines['airline_id'].isin(relevant_airlines)]
flight_traffic = flight_traffic[flight_traffic['airline_id'].isin(relevant_airlines)]

#get airline names onto flight_traffic df 
flight_traffic = pd.merge(flight_traffic,airlines,how='left',on='airline_id')


# In[9]:


#calculate total delay time for each flight
flight_traffic['total_delay'] = flight_traffic['airline_delay']+flight_traffic['weather_delay']+flight_traffic['air_system_delay']+flight_traffic['security_delay']+flight_traffic['aircraft_delay']


# In[73]:


palette=sns.color_palette("YlOrRd", 5)

#visualize breakdown of delays by delay type
delays = ['airline_delay','weather_delay','air_system_delay','security_delay','aircraft_delay']
delay_breakdown = []
total_delays = 0
for delay_type in delays:
    num_delays = len(flight_traffic[flight_traffic[delay_type]>0])
    delay_breakdown.append(num_delays)
    total_delays+=num_delays

pie = []
for delay_type in delay_breakdown:
    pie.append((delay_type/total_delays)*100)

print(pie)

fig1, ax1 = plt.subplots()
ax1.pie(pie, labels=delays, autopct='%1.1f%%', shadow=True, startangle=90, explode = (0, 0.1, 0, 0,0),colors = ['#FAE3C0', '#D75D44'])


# In[11]:


#based on time of weather measurement, label it q1/q2/q3/q4 -- in other words, in what quarter of the day was the weather assessed? 
weather['quarter'] = weather['time'].                    apply(lambda x: 'q1' if x<datetime.time(6, 0) else                    ('q2' if (x>=datetime.time(6, 0) and x<datetime.time(12,0)) else                     ('q3' if (x>=datetime.time(12, 0) and x<datetime.time(18,0)) else 'q4')))

#do the same for flight departure/arrival timestamps
flight_traffic['departure_quarter'] = flight_traffic['scheduled_departure'].                    apply(lambda x: 'q1' if x<datetime.time(6, 0) else                    ('q2' if (x>=datetime.time(6, 0) and x<datetime.time(12,0)) else                     ('q3' if (x>=datetime.time(12, 0) and x<datetime.time(18,0)) else 'q4')))

flight_traffic['arrival_quarter'] = flight_traffic['scheduled_arrival'].                    apply(lambda x: 'q1' if x<datetime.time(6, 0) else                    ('q2' if (x>=datetime.time(6, 0) and x<datetime.time(12,0)) else                     ('q3' if (x>=datetime.time(12, 0) and x<datetime.time(18,0)) else 'q4')))


# In[12]:


flight_traffic = flight_traffic.drop(columns=['taxi_out','taxi_in','diverted','wheels_off','wheels_on'])


# In[13]:


flight_traffic.head()


# In[14]:


test = pd.melt(flight_traffic, id_vars=['date','airline_id','airline_name','origin_airport',                                        'destination_airport','scheduled_departure','actual_departure','departure_quarter',                                       'scheduled_arrival','actual_arrival','arrival_quarter','cancelled','scheduled_elapsed',                                       'actual_elapsed','distance','total_delay'], value_vars=['airline_delay','weather_delay',                                        'air_system_delay','security_delay','aircraft_delay'],var_name='delay_type', value_name='delay_amt')


# In[15]:


test['delayed'] = test['total_delay'].                    apply(lambda x: 1 if x>=15 else 0)


# In[16]:


#create dataframes joining the individual flights with their respective weather conditions during departure / arrival
departure_weather = pd.merge(test,weather, how='inner', left_on=['origin_airport','date','departure_quarter'], right_on=['airport_id','date','quarter'])
arrival_weather = pd.merge(test,weather, how='inner', left_on=['destination_airport','date','arrival_quarter'], right_on=['airport_id','date','quarter'])


# # begin with departure weather

# In[17]:


departure_weather.describe()


# In[18]:


cancelled = departure_weather[departure_weather['cancelled']==1]
notcancelled = departure_weather[departure_weather['cancelled']==0]


# In[19]:


cancelled.describe()


# In[20]:


notcancelled.describe()


# ## wind analysis

# In[21]:


departure_weather.boxplot(column='wind_speed')


# In[22]:


#are extreme values only the case for cancelled flights? check boxplots
cancelled_dw = departure_weather[departure_weather['cancelled']==1]
noncancelled_dw = departure_weather[departure_weather['cancelled']==0]

cancelled_dw.boxplot(column='wind_speed')


# In[23]:


#ans: nope -- drop these outliers (value = 999)
noncancelled_dw.boxplot(column='wind_speed')


# In[24]:


#create a new df without wind speed irregularities
normal_wind = departure_weather[departure_weather['wind_speed']<100]
cancelled = normal_wind[normal_wind['cancelled']==1]
noncancelled = normal_wind[normal_wind['cancelled']==0]


# In[25]:


cancelled.wind_speed.describe()


# In[26]:


noncancelled.wind_speed.describe()


# In[34]:


#break down wind speeds into 4 levels
normal_wind['speed_level'] = normal_wind['wind_speed'].                    apply(lambda x: 'lvl1' if (x>0 and x<5) else                    ('lvl2' if (x>=5 and x<10) else                    ('lvl3' if (x>=10 and x<15) else                    ('lvl4' if x>=15 else 'none'))))


# ### looking at cancellations

# In[35]:


cancelled = []
not_cancelled = []

speeds = sorted(normal_wind['speed_level'].unique())
for speed in speeds:
    cancelled.append(len(normal_wind[(normal_wind['speed_level']==speed)&(normal_wind['cancelled']==1)]))
    not_cancelled.append(len(normal_wind[(normal_wind['speed_level']==speed)&(normal_wind['cancelled']==0)]))
raw_data = {'cancelled':cancelled,'not_cancelled':not_cancelled}
df=pd.DataFrame(raw_data)

totals = [i+j for i,j in zip(df['cancelled'], df['not_cancelled'])]
cancelled_bar = [i / j * 100 for i,j in zip(df['cancelled'], totals)]
not_cancelled_bar = [i / j * 100 for i,j in zip(df['not_cancelled'], totals)]

barWidth = 0.85

names = speeds

r=[]
for i in range(len(speeds)):
    r.append(i)

plt.bar(r, cancelled_bar, color='#D75D44', edgecolor='white', width=barWidth)
plt.bar(r, not_cancelled_bar, bottom=cancelled_bar, color='#FAE3C0', edgecolor='white', width=barWidth)

plt.xticks(r, names)
plt.xlabel("speed bucket")
plt.ylabel("percent cancelled (red)")

plt.show()


# insight: notable spike in cancellation rate for lvl4 wind speeds

# In[36]:


palette=sns.color_palette("YlOrRd", 5)
sns.set(style="ticks", palette=('#FAE3C0','#D75D44'))
sns.boxplot(x="speed_level", y="wind_speed",hue="cancelled",
            data=normal_wind,order=['none','lvl1','lvl2','lvl3','lvl4'])
sns.despine(offset=10, trim=True)


# insight: notable disparity in cancelled vs not cancelled for wind speed with lvl4 -- let's dive into lvl4 (wind speed >= 15mps)

# In[38]:


high_wind_speed = normal_wind[normal_wind['wind_speed']>=15]
high_wind_speed['wind_speed'] = round(high_wind_speed['wind_speed'])


# In[39]:


cancelled = []
not_cancelled = []

speeds = sorted(high_wind_speed['wind_speed'].unique())
for speed in speeds:
    cancelled.append(len(high_wind_speed[(high_wind_speed['wind_speed']==speed)&(high_wind_speed['cancelled']==1)]))
    not_cancelled.append(len(high_wind_speed[(high_wind_speed['wind_speed']==speed)&(high_wind_speed['cancelled']==0)]))
raw_data = {'cancelled':cancelled,'not_cancelled':not_cancelled}
df=pd.DataFrame(raw_data)

totals = [i+j for i,j in zip(df['cancelled'], df['not_cancelled'])]
cancelled_bar = [i / j * 100 for i,j in zip(df['cancelled'], totals)]
not_cancelled_bar = [i / j * 100 for i,j in zip(df['not_cancelled'], totals)]

barWidth = 0.85

names = (speeds)

r=[]
for i in range(len(speeds)):
    r.append(i)

plt.bar(r, cancelled_bar, color='#D75D44', edgecolor='white', width=barWidth)
plt.bar(r, not_cancelled_bar, bottom=cancelled_bar, color='#FAE3C0', edgecolor='white', width=barWidth)

plt.xticks(r, names)
plt.xlabel("speed bucket")
plt.ylabel("percent cancelled (red)")

plt.show()


# insight: if wind speed >= 19 mps, flight will most likely be cancelled

# ### looking at delays

# In[40]:


delayed = []
not_delayed = []

speeds = sorted(normal_wind['speed_level'].unique())
for speed in speeds:
    delayed.append(len(normal_wind[(normal_wind['speed_level']==speed)&(normal_wind['delayed']==1)]))
    not_delayed.append(len(normal_wind[(normal_wind['speed_level']==speed)&(normal_wind['delayed']==0)]))
raw_data = {'delayed':delayed,'not_delayed':not_delayed}
df=pd.DataFrame(raw_data)

totals = [i+j for i,j in zip(df['delayed'], df['not_delayed'])]
delayed_bar = [i / j * 100 for i,j in zip(df['delayed'], totals)]
not_delayed_bar = [i / j * 100 for i,j in zip(df['not_delayed'], totals)]

barWidth = 0.85

names = speeds

r=[]
for i in range(len(speeds)):
    r.append(i)

plt.bar(r, delayed_bar, color='#D75D44', edgecolor='white', width=barWidth)
plt.bar(r, not_delayed_bar, bottom=delayed_bar, color='#FAE3C0', edgecolor='white', width=barWidth)

plt.xticks(r, names)
plt.xlabel("speed bucket")
plt.ylabel("percent delayed (red)")

plt.show()


# insight: delay rate increases marginally with wind speed

# In[41]:


sns.set(style="ticks", palette="pastel")
sns.boxplot(x="speed_level", y="wind_speed",hue="delayed",
            data=normal_wind)
sns.despine(offset=10, trim=True)


# insight: notable disparity in delayed vs not delayed for wind speed with lvl4 reaffirms previous conclusion about cancellations -- wind speed in upper end of lvl4 speed bucket will most likely result in cancellation (and thus not a delay)

# In[42]:


#get all non-cancelled flights to analyze delay time
noncancelled_wind = normal_wind[normal_wind['cancelled']==0]

#adjust delay times to account for definition of delayed = delay time > 15
subset = noncancelled_wind[noncancelled_wind['delayed'] == 0]
subset['total_delay'] = np.nan
noncancelled_wind.update(subset)


# In[43]:


speed_buckets = noncancelled_wind.groupby("speed_level")['total_delay'].agg(np.mean)
noncancelled_wind.groupby("speed_level")['total_delay'].agg(np.mean).plot(yerr = speed_buckets, kind = 'bar', title = 'delay amount by wind speed level')
sns.mpl.pyplot.ylabel('avg delay')


# insight: when flight is not cancelled, higher wind speed will marginally imply greater delay

# ### summary:
# wind speed categories:
# - 'lvl1' if (x>0 and x<5)
# - 'lvl2' if (x>=5 and x<10)
# - 'lvl3' if (x>=10 and x<15)
# - 'lvl4' elsewise
# 
# insights:
# - insight: notable spike in cancellation rate for lvl4 wind speeds
# - insight: notable disparity in wind speed distribution for cancelled vs not cancelled in lvl4
# - insight: if wind speed >= 19 mps, flight will most likely be cancelled
# - insight: delay rate increases marginally with wind speed 
# - insight: notable disparity in delayed vs not delayed for wind speed with lvl4 reaffirms previous conclusion about cancellations -- wind speed in upper end of lvl4 speed bucket will most likely result in cancellation (and thus not a delay)
# - insight: when flight is not cancelled, higher wind speed will generally imply greater delay
# 
# key conclusions:
# - if wind speed >= 19 mps, flight will most likely be cancelled (>60% chance)
# - delay rate increases marginally with wind speed 
# - when flight is not cancelled, higher wind speed will marginally imply greater delay

# ## snow analysis

# In[44]:


departure_weather.boxplot(column='snow_depth')


# In[45]:


cancelled_snow = departure_weather[departure_weather['cancelled']==1]
noncancelled_snow = departure_weather[departure_weather['cancelled']==0]


# In[46]:


cancelled_snow.snow_depth.describe()


# In[47]:


noncancelled_snow.snow_depth.describe()


# In[48]:


departure_weather['snow_level'] = departure_weather['snow_depth'].                    apply(lambda x: 'lvl1' if (x>0 and x<4) else                    ('lvl2' if (x>=4 and x<8) else                    ('lvl3' if (x>=8 and x<12) else                    ('lvl4' if x>=12 else 'none'))))


# ### looking at cancellations

# In[49]:


cancelled = []
not_cancelled = []

speeds = sorted(departure_weather['snow_level'].unique())
for speed in speeds:
    cancelled.append(len(departure_weather[(departure_weather['snow_level']==speed)&(departure_weather['cancelled']==1)]))
    not_cancelled.append(len(departure_weather[(departure_weather['snow_level']==speed)&(departure_weather['cancelled']==0)]))
raw_data = {'cancelled':cancelled,'not_cancelled':not_cancelled}
df=pd.DataFrame(raw_data)

totals = [i+j for i,j in zip(df['cancelled'], df['not_cancelled'])]
cancelled_bar = [i / j * 100 for i,j in zip(df['cancelled'], totals)]
not_cancelled_bar = [i / j * 100 for i,j in zip(df['not_cancelled'], totals)]

barWidth = 0.85

names = speeds

r=[]
for i in range(len(speeds)):
    r.append(i)

plt.bar(r, cancelled_bar, color='#D75D44', edgecolor='white', width=barWidth)
plt.bar(r, not_cancelled_bar, bottom=cancelled_bar, color='#FAE3C0', edgecolor='white', width=barWidth)

plt.xticks(r, names)
plt.xlabel("snow depth bucket")
plt.ylabel("percent cancelled (red)")

plt.show()


# insight: cancellation rate highest for lvl3 (8-12cm) and lvl4 (>12cm)

# In[50]:


#take a look at upper levels
mid_snow = departure_weather[(departure_weather['snow_level']=='lvl2')|(departure_weather['snow_level']=='lvl3')|(departure_weather['snow_level']=='lvl4')]
mid_snow['snow_depth'] = round(mid_snow['snow_depth'])


# In[51]:


cancelled = []
not_cancelled = []

snows = sorted(mid_snow['snow_depth'].unique())
for snow in snows:
    cancelled.append(len(mid_snow[(mid_snow['snow_depth']==snow)&(mid_snow['cancelled']==1)]))
    not_cancelled.append(len(mid_snow[(mid_snow['snow_depth']==snow)&(mid_snow['cancelled']==0)]))
raw_data = {'cancelled':cancelled,'not_cancelled':not_cancelled}
df=pd.DataFrame(raw_data)

totals = [i+j for i,j in zip(df['cancelled'], df['not_cancelled'])]
cancelled_bar = [i / j * 100 for i,j in zip(df['cancelled'], totals)]
not_cancelled_bar = [i / j * 100 for i,j in zip(df['not_cancelled'], totals)]

barWidth = 0.85

names = snows

r=[]
for i in range(len(snows)):
    r.append(i)

plt.bar(r, cancelled_bar, color='#D75D44', edgecolor='white', width=barWidth)
plt.bar(r, not_cancelled_bar, bottom=cancelled_bar, color='#FAE3C0', edgecolor='white', width=barWidth)

plt.xticks(r, names)
plt.xlabel("snow depth bucket")
plt.ylabel("percent cancelled (red)")

plt.show()


# In[52]:


cancelled_bar


# insight: high cancellation rate once snow depth exceeds 15cm, about 20% once depth exceeds 4cm

# ### looking at delays

# In[53]:


delayed = []
not_delayed = []

speeds = sorted(departure_weather['snow_level'].unique())
for speed in speeds:
    delayed.append(len(departure_weather[(departure_weather['snow_level']==speed)&(departure_weather['delayed']==1)]))
    not_delayed.append(len(departure_weather[(departure_weather['snow_level']==speed)&(departure_weather['delayed']==0)]))
raw_data = {'delayed':delayed,'not_delayed':not_delayed}
df=pd.DataFrame(raw_data)

totals = [i+j for i,j in zip(df['delayed'], df['not_delayed'])]
delayed_bar = [i / j * 100 for i,j in zip(df['delayed'], totals)]
not_delayed_bar = [i / j * 100 for i,j in zip(df['not_delayed'], totals)]

barWidth = 0.85

names = speeds

r=[]
for i in range(len(speeds)):
    r.append(i)

plt.bar(r, delayed_bar, color='#D75D44', edgecolor='white', width=barWidth)
plt.bar(r, not_delayed_bar, bottom=delayed_bar, color='#FAE3C0', edgecolor='white', width=barWidth)

plt.xticks(r, names)
plt.xlabel("snow depth bucket")
plt.ylabel("percent delayed (red)")

plt.show()


# In[54]:


delayed_bar


# insight: ~50% chance of delay when if snow just begins to pile up at departure; decreases thorugh lvl4 (as cancellation rate increases)

# In[55]:


#get all non-cancelled flights to analyze delay time
noncancelled_snow = departure_weather[departure_weather['cancelled']==0]

#adjust delay times to account for definition of delayed = delay time > 15
subset = noncancelled_snow[noncancelled_snow['delayed'] == 0]
subset['total_delay'] = np.nan
noncancelled_snow.update(subset)


# In[56]:


snow_buckets = noncancelled_snow.groupby("snow_level")['total_delay'].agg(np.mean)
noncancelled_snow.groupby("snow_level")['total_delay'].agg(np.mean).plot(yerr = snow_buckets, kind = 'bar', title = 'delay amount by snow depth level',color='#FAE3C0')
sns.mpl.pyplot.ylabel('avg delay')


# insight: longest delay for snow depth lvl2-lvl3 (4-12cm)

# ### summary:
# 
# snow depth categories:
# - 'lvl1' if (x<4)
# - 'lvl2' if (x>=4 and x<8)
# - 'lvl3' if (x>=8 and x<12)
# - 'lvl4' elsewise
# 
# insights:
# - insight: cancellation rate highest for lvl3 (8-12cm) and lvl4 (>12cm)
# - insight: high cancellation rate once snow depth exceeds 15cm, about 20% once depth exceeds 4cm
# - insight: ~50% chance of delay when if snow just begins to pile up at departure; decreases thorugh lvl4 (as cancellation rate increases)
# - insight: longest delay for snow depth lvl2-lvl3 (4-12cm)
# 
# key conclusions:
# - cancellation rate highest if during departure time snow depth is (8+ cm)
# - ~50% chance of delay when if snow just begins to pile up at departure; decreases thorugh lvl4 (as cancellation rate increases)
# - longest average delay time when snow depth lvl2-lvl3 (4-12cm)

# ## wind x snow analysis

# In[57]:


#reset
departure_weather = pd.merge(test,weather, how='inner', left_on=['origin_airport','date','departure_quarter'], right_on=['airport_id','date','quarter'])
departure_weather = departure_weather[departure_weather['wind_speed']<100]


# In[58]:


departure_weather['wind_speed_level'] = departure_weather['wind_speed'].                    apply(lambda x: 'lvl1' if (x>0 and x<5) else                    ('lvl2' if (x>=5 and x<10) else                    ('lvl3' if (x>=10 and x<15) else                    ('lvl4' if x>=15 else 'none'))))

departure_weather['snow_depth_level'] = departure_weather['snow_depth'].                    apply(lambda x: 'lvl1' if (x>0 and x<4) else                    ('lvl2' if (x>=4 and x<8) else                    ('lvl3' if (x>=8 and x<12) else                    ('lvl4' if x>=12 else 'none'))))


# In[59]:


#adjust delay times to account for definition of delayed = delay time > 15
subset = departure_weather[departure_weather['delayed'] == 0]
subset['total_delay'] = np.nan
departure_weather.update(subset)


# In[60]:


#get all possible combinatinos of wind speed level and snow depth level
level_index = departure_weather.groupby(['wind_speed_level','snow_depth_level']).size().reset_index()
reference = level_index
counts = list(level_index[0])
level_index = level_index.drop(columns=0)
level_index['combo'] = np.arange(len(level_index))
dw = pd.merge(departure_weather,level_index,how='left',on=['wind_speed_level','snow_depth_level'])


# In[61]:


reference


# In[63]:


#calculate aggregate stats
cancellation_combo_groupings = list(dw.groupby("combo")['cancelled'].agg(np.sum))
delay_combo_groupings = list(dw.groupby("combo")['delayed'].agg(np.sum))
delay_time_groupings = list(dw.groupby("combo")['total_delay'].agg(np.mean))

cancellation_df = pd.DataFrame({'combo_id':np.arange(len(level_index)),'count':counts,'num_cancelled':cancellation_combo_groupings})
cancellation_df['cancellation_rate'] = cancellation_df['num_cancelled']/cancellation_df['count']

delay_df = pd.DataFrame({'combo_id':np.arange(len(level_index)),'count':counts,'num_delayed':delay_combo_groupings,'avg_delay_time':delay_time_groupings})
delay_df['delay_rate'] = delay_df['num_delayed']/delay_df['count']
delay_df['expected_delay'] = delay_df['delay_rate']*delay_df['avg_delay_time']


# In[64]:


sns.barplot(x="combo_id", y="cancellation_rate", data=cancellation_df,palette=sns.color_palette("YlOrRd", 5))


# In[65]:


reference['cancellation_rate'] = cancellation_df['cancellation_rate']


# In[66]:


reference.sort_values(by='cancellation_rate',ascending=False)


# In[67]:


sns.barplot(x="combo_id", y="delay_rate", data=delay_df)


# In[68]:


sns.barplot(x="combo_id", y="avg_delay_time", data=delay_df)


# In[69]:


sns.barplot(x="combo_id", y="expected_delay", data=delay_df,palette=sns.color_palette("YlOrRd", 5))


# In[61]:


reference['expected_delay'] = delay_df['expected_delay']


# In[72]:


reference['delay_rate'] = delay_df['delay_rate']


# In[83]:


reference.sort_values(by=['cancellation_rate','expected_delay'],ascending=False)


# In[84]:


reference


# ## bad weather ids:

# In[76]:


dw['bad_weather'] = np.nan


# In[85]:


#label traffic/weather dataframe with bad_weather status
bad_weather_ids = [0,1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
bad_weather = dw[(dw['combo'].isin(bad_weather_ids))|(dw['wind_speed']>=19)]
bad_weather['bad_weather'] = "Y"
dw.update(bad_weather)
dw=dw.fillna({'bad_weather':"N"})


# In[86]:


dw.head()


# In[91]:


dw.to_csv('expanded_new_weather.csv')

