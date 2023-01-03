#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from pandas import json_normalize
import requests


# In[9]:


from time import sleep

datas = pd.DataFrame()
a = 'https://api.weather.com/v1/location/KMDW:9:US/observations/historical.json?apiKey=e1f10a1e78da46f5b10a1e78da96f525&units=e&startDate='

years = ['2021', '2022']
months = ['08', '09', '10', '11', '12']
days = [31, 30, 31, 30, 31]


months1 = ['01', '02', '03', '04', '05', '06',
              '07', '08']

days1 = [31, 28 , 31, 30, 31, 30, 31, 31]

seconds = 7

for y in range(len(years)):  
    if years[y] == '2021':
        for m in range(len(months)):
            for d in range(1, days[m] + 1):
                sleep(seconds)
                weather_data = requests.get(a + years[y] + months[m] + str(d).zfill(2)).json();
                df= json_normalize(weather_data, 'observations')
                datas= datas.append(df)
                print(m)
                print(d)
                #print(df)
        pass
    else:
        for m1 in range(len(months1)):
            for d1 in range(1, days1[m1] + 1):
                sleep(seconds)
                weather_data = requests.get(a + years[y] + months1[m1] + str(d1).zfill(2)).json();
                df= json_normalize(weather_data, 'observations')
                data = datas.append(df)
                print(m)
                print(d)
                #print(df)
                
print(datas)


# In[68]:


from time import sleep

data_together = pd.DataFrame()
a = 'https://api.weather.com/v1/location/KMDW:9:US/observations/historical.json?apiKey=e1f10a1e78da46f5b10a1e78da96f525&units=e&startDate=2021'

#years = ['2021', '2022']
months = ['08', '09', '10', '11', '12']
days = [31, 30, 31, 30, 31]


#months1 = ['01', '02', '03', '04', '05', '06','07', '08']

#days1 = [31, 28 , 31, 30, 31, 30, 31, 31]

seconds = 7

for m in range(len(months)):
    for d in range(1, days[m] + 1):
        sleep(seconds)
        weather_data = requests.get(a + months[m] + str(d).zfill(2)).json();
        df= json_normalize(weather_data, 'observations')
        data_together = data_together.append(df)
        print(m)
        print(d)
        #print(df)

                
print(data_together)


# In[72]:


a = 'https://api.weather.com/v1/location/KMDW:9:US/observations/historical.json?apiKey=e1f10a1e78da46f5b10a1e78da96f525&units=e&startDate=2022'

#years = ['2021', '2022']
#months = ['08', '09', '10', '11', '12']
#days = [31, 30, 31, 30, 31]


months1 = ['01', '02', '03', '04', '05', '06','07', '08']

days1 = [31, 28 , 31, 30, 31, 30, 31, 31]

seconds = 7

for m1 in range(len(months1)):
    for d1 in range(1, days1[m1] + 1):
        sleep(seconds)
        weather_data = requests.get(a + months1[m1] + str(d1).zfill(2)).json();
        df= json_normalize(weather_data, 'observations')
        data_together = data_together.append(df)
        print(m1)
        print(d1)
        #print(df)

                
print(data_together)


# In[5]:


data_together


# In[75]:


data_together['expire_time_gmt'] = pd.to_datetime(data_together['expire_time_gmt'],unit='s') #convert unix to datetime
data_together['valid_time_gmt'] = pd.to_datetime(data_together['valid_time_gmt'],unit='s') #convert unix to datetime


# In[76]:


data_together


# In[80]:


data_together=data_together.drop(['key', 'class','expire_time_gmt', 'obs_id', 'day_ind', 'icon_extd', 'rh', 'vis', 'wc', 'wdir_cardinal', 'gust', 'qualifier', 'qualifier_svrty', 'blunt_phrase', 'terse_phrase', 'primary_wave_period', 'primary_wave_height', 'primary_swell_period', 'primary_swell_height', 'primary_swell_direction' ,'secondary_swell_period', 'secondary_swell_height', 'secondary_swell_direction' ], axis=1)

data_together


# In[4]:


data_together.to_excel('C:\\Users\\Public\\bikesharing_Chicago\\weather_data1.xlsx')


# In[3]:


a = 'https://api.weather.com/v1/location/KMDW:9:US/observations/historical.json?apiKey=e1f10a1e78da46f5b10a1e78da96f525&units=e&startDate=20220101'


weather_data = requests.get(a).json();
df1= json_normalize(weather_data, 'observations')
df1['expire_time_gmt'] = pd.to_datetime(df1['expire_time_gmt'],unit='s')
df1


# In[19]:


#print(weather_data)
df = json_normalize(weather_data, 'observations') #convert json format to DataFrame
#df['expire_time_gmt'] = pd.to_datetime(df['expire_time_gmt'],unit='s') #convert unix to datetime
#df['valid_time_gmt'] = pd.to_datetime(df['valid_time_gmt'],unit='s') #convert unix to datetime

#print(df(2))
#df=df.drop(['key', 'class','expire_time_gmt', 'obs_id', 'day_ind', 'icon_extd', 'rh', 'vis', 'wc', 'wdir_cardinal', 'gust', 'qualifier', 'qualifier_svrty', 'blunt_phrase', 'terse_phrase', 'primary_wave_period', 'primary_wave_height', 'primary_swell_period', 'primary_swell_height', 'primary_swell_direction' ,'secondary_swell_period', 'secondary_swell_height', 'secondary_swell_direction' ], axis=1)

print(df)


# In[35]:


import pandas as pd
import sys
input_file = 'C:\\Users\\Public\\bikesharing_Chicago\\Bikesharing_v4.xlsx'
#output_file = 'C:\\Users\\Public\\bikesharing_Chicago\\Bikesharing_v3.xlsx'
df = pd.read_excel(input_file, None)
all_df = []
for key in df.keys():
    all_df.append(df[key])
data_concatenated = pd.concat(all_df,axis=0,ignore_index=True)
#writer = pd.ExcelWriter(output_file)
#data_concatenated.to_excel(writer,sheet_name='merged',index=False)
#writer.save()
data_concatenated


# In[40]:


data_concatenated.dtypes


# In[41]:


data_concatenated['Started Time'] = data_concatenated['Started Time'].astype('string')


# In[43]:


data_concatenated['Hour'] = data_concatenated['Started Time'].str[0:2]


# In[44]:


data_concatenated


# In[50]:


data_counted= data_concatenated.groupby(['Started Date', 'Hour']).count().reset_index()


# In[51]:


data_counted


# In[52]:


data_counted = data_counted.rename(columns = {"Started Time":"Count"})


# In[53]:


data_counted


# In[54]:


data_counted.to_excel('C:\\Users\\Public\\bikesharing_Chicago\\Bikesharing_perhour.xlsx')


# In[ ]:




