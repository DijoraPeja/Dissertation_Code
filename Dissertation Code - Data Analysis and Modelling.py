#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import datetime
import numpy as np
import seaborn as sns
from scipy.stats import boxcox
from scipy import stats
import matplotlib.pyplot as plt


# **Transformation of Data**

# In[96]:


complete = pd.read_excel('C:\\Users\\Public\\bikesharing_Chicago\\datasetcomplete_v6.xlsx')


# In[97]:


complete['Year']=complete['Date'].dt.year
complete['Month']=complete['Date'].dt.month
complete['Day']=complete['Date'].dt.day
complete['Week-day'] = complete['Date'].dt.dayofweek


# In[98]:


weekend = []
for value in complete["Week-day"]:
    if value >= 5:
        weekend.append("Weekend")
    else:
        weekend.append("Workday")
      
complete["Weekend"] = weekend  


# In[99]:


seasonnr={'Winter':1,'Spring':2,'Summer':3,'Autumn':4}
complete['Seasonnr']=complete['Season'].map(seasonnr)

weekend={'Weekend':1,'Workday':0}
complete['Is_weekend']=complete['Weekend'].map(weekend)

holiday={'Holiday':1,'No holiday':0}
complete['Is_holiday']=complete['Holiday'].map(holiday)


# In[100]:


clouds={'CLR':1,'FEW':2,'SCT':3,'BKN':4, 'OVC':5}
complete['Clouds']=complete['clds'].map(clouds)


# In[101]:


weatherc={'Fair':1,'Fair / Windy':1,'Cloudy':2,'Cloudy / Windy':2, 'Partly Cloudy':2, 'Fog': 2, 'Haze':2, 'Haze / Windy': 2, 'Heavy Rain':4, 'Heavy Rain / Windy': 4, 'Heavy Snow / Windy': 4, 'Heavy T-Storm':4, 'Heavy T-Storm / Windy':4, 'Light Drizzle':3, 'Light Drizzle / Windy': 3, 'Light Freezing Drizzle': 3, 'Light Rain': 3, 'Light Rain / Windy': 3, 'Light Rain with Thunder': 3, 'Light Snow':3, 'Light Snow / Windy':3, 'Mostly Cloudy':2,'Mostly Cloudy / Windy':2,'Partly Cloudy / Windy':2, 'Rain':3, 'Rain / Windy':3, 'Snow':3, 'Thunder':4, 'Thunder / Windy': 4, 'Thunder in the Vicinity' :4, 'T-Storm':4, 'T-Storm / Windy':4, 'Wintry Mix':4 }
complete['Weather_condition']=complete['wx_phrase'].map(weatherc)


# In[102]:


complete = complete.rename(columns = {"Unnamed: 0":"Id"})


# In[103]:


complete.info()


# In[104]:


print("Duplicate entry:",len(complete[complete.duplicated()])) 


# In[108]:


complete['Date1']=complete['Date'].dt.date


# In[109]:


complete_onlydate = complete.groupby(['Date1']).count().reset_index()
#complete_onlydate = complete_onlydate.select['Date1']


# In[110]:


complete_onlydate


# In[111]:


complete_onlydate = complete_onlydate[complete_onlydate.columns[0:2]]


# In[112]:


complete_onlydate["day1"] = complete_onlydate.index + 1


# In[113]:


complete_onlydate


# In[114]:


complete_onlydate = complete_onlydate.drop(['Id'], axis=1)


# In[115]:


complete = complete.merge(complete_onlydate, on='Date1', how='outer')


# In[116]:


complete


# In[117]:


complete_onlydate


# In[118]:


complete['Date1']


# In[636]:


complete2 = complete.drop(['Date1', 'Month_name', 'Seasonnr'], axis=1)


# In[642]:


complete2 = complete2.drop(['Day'], axis=1)


# In[643]:


complete2 = complete2.rename(columns={'day1':'Day'})


# In[644]:


def completeinfo():
    compl = pd.DataFrame(index=complete2.columns)
    compl['DataType'] = complete.dtypes
    compl["Non-null_Values"] = complete.count()
    compl['Unique_Values'] = complete.nunique()
    compl['NaN_Values'] = complete.isnull().sum()
    return compl
        


# In[645]:


completeinfo()


# In[646]:


complete2.describe().T 


# **Distribution of Data**

# In[419]:



sns.set(style="darkgrid")
f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.20, .85)})
sns.boxplot(data=complete, x='Count', ax=ax_box, color='#6daa9f')
sns.histplot(data=complete, x="Count", ax=ax_hist,kde=True, edgecolor="#4c766f", color='#62998f')
ax_box.set(xlabel='')
plt.show()


# In[121]:


plt.savefig("output.jpg")


# In[122]:


plt.figure(figsize=(10,5))
plt.boxplot(complete['Count'],vert=False)
plt.show()


# Checking for outliers

# In[31]:


z = np.abs(stats.zscore(complete['Count']))
print(z)


# In[32]:


print(np.where(z > 3))


# In[33]:


CompleteWithoutOutliers = complete[np.abs(complete["Count"] - complete["Count"].mean()) <= (3*complete["Count"].std())]

print(complete.shape)
print(CompleteWithoutOutliers.shape)


# Distribution of Temp

# In[34]:


sns.set(style="darkgrid")
f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
sns.boxplot(data=complete, x='temp', ax=ax_box)
sns.histplot(data=complete, x="temp", ax=ax_hist,kde=True)
ax_box.set(xlabel='')
plt.show()


# For the Box-Cox transformation, a λ value of 1 is equivalent to using the original data. Therefore, if the confidence interval for the optimal λ includes 1, then no transformation is necessary. The llamba in our case is 0.243

# In[401]:


transformed_data, best_lambda = boxcox(complete['Count']) 


# In[402]:


sns.displot(transformed_data, kde=True) 


# In[403]:


print(best_lambda)


# In[38]:


fig,axes = plt.subplots(2,2,figsize=(20,13))
sns.histplot((complete['Count']),ax=axes[0,0],color='brown').set_title("Count")
sns.histplot(np.log1p(complete['Count']+0.0000001),ax=axes[0,1],color='red').set_title("Count - Log Count") 
sns.histplot(np.sqrt(complete['Count']),ax=axes[1,0], color='blue').set_title("Count - Square root")
sns.histplot(transformed_data, color='blue').set_title("Count - Box-cox")


# In[408]:


fig,axes = plt.subplots(1,2, figsize=(20,5))
sns.histplot((complete['Count']), ax=axes[0],edgecolor="#f6685e",linewidth=2, color='#f8958e').set_title(" Input Data - Count")
stats.probplot((complete['Count']), dist='norm',plot=plt)


# In[444]:


fig,axes = plt.subplots(1,2, figsize=(20,5))
font = {'size'   : 15}
sns.histplot(np.log1p(complete['Count']+0.0000001),ax=axes[0], edgecolor="#ac4841",linewidth=2,color='#f6685e').set_title("Log - Count") 
stats.probplot(np.log1p(complete['Count']+0.0000001), dist='norm',plot=plt)


# In[445]:


fig,axes = plt.subplots(1,2, figsize=(20,5))
sns.histplot(np.sqrt(complete['Count']),ax=axes[0], edgecolor="#4c766f",linewidth=2,color='#b6d4cf').set_title("Square root - Count")
stats.probplot(np.sqrt(complete['Count']), dist='norm',plot=plt)


# In[443]:


fig,axes = plt.subplots(1,2, figsize=(20,5))
sns.histplot(transformed_data, ax=axes[0], edgecolor="#4c766f",linewidth=2,color="#6daa9f").set_title("Box-cox - Count")
stats.probplot(transformed_data, dist='norm',plot=plt)


# **Graphs about the data in general**

# In[43]:


font = {'size'   : 12}
plt.rc('font', **font)
complete.groupby(complete.Date.dt.date).agg({'Count':'sum'}).reset_index().plot(x='Date',
                                                                       y='Count',
                                                                       figsize=(20,8),
                                                                       legend=False, 
                                                                       title='Sum of bike count daily')
plt.show()


# In[446]:


font = {'size'   : 17}
complete.groupby(complete.Date.dt.date).agg({'Count':'mean'}).reset_index().plot(x='Date',
                                                                       y='Count',
                                                                       figsize=(20,8),
                                                                       legend=False, 
                                                                       color = '#f44336')
plt.title('Average daily count of bikes ', **font)
plt.xlabel('Date', **font)
plt.ylabel('Count', **font)
plt.rc('font', **font)
plt.show()


# In[438]:


font = {'size'   : 17}
complete.groupby(complete.temp).agg({'Count':'mean'}).reset_index().plot(x='temp',
                                                                       y='Count',
                                                                       figsize=(20,8),
                                                                       legend=False, 
                                                                        color = '#f44336')
plt.title('Relation of Avg. Bike Count and Temperature', **font)
plt.xlabel('Temperature', **font)
plt.ylabel('Count', **font)
plt.rc('font', **font)
plt.show()


# In[439]:


font = {'size'   : 17}
complete.groupby(complete.pressure).agg({'Count':'mean'}).reset_index().plot(x='pressure',
                                                                       y='Count',
                                                                       figsize=(20,8),
                                                                       legend=False, 
                                                                       color= '#f44336',

                                                                            )
plt.title('Relation of Avg. Bike Count and Pressure ', **font)
plt.xlabel('Pressure', **font)
plt.ylabel('Count', **font)
plt.rc('font', **font)
plt.rc('font', **font)
plt.show()


# In[340]:


cols = ['#f6685e',"#FFD966",  '#CFE2F3', "#6daa9f"]

a,b = plt.subplots(1,1, figsize=(10,5))
sns.boxplot(data=complete, y = 'Count', x = 'Season', palette = cols)


# In[429]:


cols = ['#f6685e',"#FFD966",  '#CFE2F3', "#6daa9f"]

col1=  ['#e7f0f9', '#CFE2F3', '#a7ccc5', '#8abbb2', '#6daa9f', '#f8958e', '#f6685e','#f6776e', '#f9a49e', '#ffd966', '#ffe084', '#ffe8a3']

        
fig,axes = plt.subplots(1,2,figsize=(15,4))
sns.boxplot(data=complete,ax=axes[0], y = 'Count', x = 'Month', palette=col1)
sns.boxplot(data=complete, y = 'Count', x = 'Season',  palette=cols)


# In[355]:


col1=  ['#e7f0f9', '#CFE2F3', '#a7ccc5', '#8abbb2', '#6daa9f', '#f8958e', '#f6685e','#f6776e', '#f9a49e', '#ffd966', '#ffe084', '#ffe8a3']

a,b = plt.subplots(1,1, figsize=(10,5))
sns.boxplot(data=complete, y = 'Count', x = 'Hour', palette=col1)


# *Problem with this graph, you do not know how many days are rainy days, how many are stormy, maybe this could be the reason why heavy storm has a higher count than rain for instance*

# In[432]:


cols = ['#f6685e', "#6daa9f"]
cols1 = ['#f6685e',"#6daa9f",'#a7ccc5','#CFE2F3', '#ffe8a3','#FFD966','#f8958e']
fig,axes = plt.subplots(1,2,figsize=(20,5))
sns.boxplot(data=complete,ax=axes[0], y = 'Count', x = 'Weekday-name', palette=cols1)
figure = sns.boxplot(data=complete, y = 'Count', x = 'Weekend', palette=cols)
figure.savefig("output.png")


# In[106]:


cols= ["#6daa9f","#774571"]
a,b = plt.subplots(1,1, figsize=(10,5))
sns.boxplot(data=complete, y = 'Count', x = 'Weather_condition', palette = cols)


# In[28]:


complete1 = complete.copy() 
weather={1:1,2:1, 3:2, 4:2}
complete1['W_conditions']=complete['Weather_condition'].map(weather)


# In[61]:


a,b = plt.subplots(1,1, figsize=(10,5))
sns.lineplot(data=complete, x="Hour", y="Count", 
              hue="Season", marker="x",markeredgecolor="black")
#plt.tight_layout()


# In[433]:


cols = [ "#6daa9f",'#f6685e']
a,b = plt.subplots(1,1, figsize=(10,5))
sns.lineplot(data=complete, x="Hour", y="Count", 
              hue="Holiday", marker="x",markeredgecolor="black", palette=cols)


# In[64]:


a,b = plt.subplots(1,1, figsize=(10,5))
sns.boxplot(data=complete, y = 'Count', x = 'clds')


# In[65]:


pivot = pd.pivot_table(complete, values='Count', index='Hour', columns='Weekday-name', aggfunc='median')
pivot = pivot[['Mon', 'Tues', 'Weds', 'Thurs', 'Fri', 'Sat', 'Sun',]]

a,b = plt.subplots(1,1, figsize=(10,5))
sns.heatmap(pivot, cmap='viridis')
plt.title('Heatmap of bike count during the days each hour')
plt.show()      


# In[66]:


pivot = pd.pivot_table(complete, values='Count', index='Hour', columns='Weekday-name', aggfunc='mean')
pivot = pivot[['Mon', 'Tues', 'Weds', 'Thurs', 'Fri', 'Sat', 'Sun',]]

a,b = plt.subplots(1,1, figsize=(10,5))
sns.heatmap(pivot, cmap='viridis')
plt.title('Heatmap of bike count during the days each hour')
plt.show()    


# In[448]:


# can also be visulaized using histograms for all the continuous variables.

cols = ['#f6685e',"#FFD966",  '#CFE2F3', "#6daa9f"]


#sns.countplot(x= data["DEATH_EVENT"], palette= cols)
#complete.temp.unique()
fig,axes=plt.subplots(2,2)
axes[0,0].hist(x="temp",data=complete,edgecolor="#f6685e",linewidth=2, color='#f8958e')
axes[0,0].set_title("Temp")
axes[0,1].hist(x="dewPt",data=complete,edgecolor="#4c766f",linewidth=2,color='#b6d4cf')
axes[0,1].set_title("Dew Point")
axes[1,0].hist(x="uv_index",data=complete,edgecolor="#4c766f",linewidth=2,color="#6daa9f")
axes[1,0].set_title("UV Index")
axes[1,1].hist(x="pressure",data=complete,edgecolor="#ac4841",linewidth=2,color="#f6685e")
axes[1,1].set_title("Pressure")
fig.set_size_inches(10,10)


# In[123]:


complete = complete.drop(['Id', 'Unnamed: 0.1', 'Id'], axis=1)


# In[551]:


import seaborn as sns


# In[580]:


complete1 = complete.drop(['Id', 'Day'], axis=1)


# In[584]:


complete1 = complete1.rename(columns={'day1':'Day'})


# In[585]:


complete1 = complete1.rename(columns={'Seasonnr':'Season'})


# In[599]:


colors = ['#6daa9f','#8abbb2', '#a7ccc5', '#d3e5e2', '#fcd1ce', '#fab3ae','#f9a49e', '#f8958e', '#f7867e', '#f6685e']
corrmat = complete1[:].corr().round(2)
cmap = sns.blend_palette(colors, n_colors=10, as_cmap=False, input='rgb')
#mask = np.array(corrmat)
#mask[np.tril_indices_from(mask)]=False
plt.subplots(figsize=(18,18))
corrplot = sns.heatmap(corrmat,cmap= cmap,annot=True, square=True, cbar=True)
corrplot1 = corrplot.figure
corrplot1.savefig('C:\\Users\\Public\\bikesharing_Chicago\\corrplot.png')


# In[70]:


complete.columns


# In[133]:


complete.insert(loc=0, column='Id', value=np.arange(len(complete)))


# In[135]:


complete.columns


# **Two dataframes, one with month and one with season, try both of them for predictions

# In[136]:


compl_mont=complete.loc[:, complete.columns.drop(['Id','Date1', 'Year','Date', 'wdir', 'wx_phrase', 'dewPt', 'clds', 'Holiday','feels_like', 'heat_index', 'Week-day', 'Weekend', 'Weekday-name', 'Season', 'Seasonnr'])]


# In[137]:


compl_mont


# In[138]:


compl_mont= compl_mont.drop(['Unnamed: 0.1'], axis=1)


# In[139]:


compl_mont['Count_log']= np.log1p(compl_mont['Count'])


# In[140]:


compl_mont= compl_mont.drop(['Count'], axis=1)


# In[141]:


compl_mont= compl_mont.drop(['Day'], axis=1)


# In[142]:


compl_mont= compl_mont.rename(columns={"day1":"Day"})


# In[143]:


compl_mont


# In[144]:


compl_monthw = compl_mont.drop(['Clouds'], axis=1)


# In[163]:


compl_monthw.isna().sum()


# In[145]:


compl_monthc = compl_mont.drop(['Weather_condition'], axis=1)


# In[146]:


compl_monthc


# In[164]:


compl_month3= compl_monthw.copy()


# In[166]:


weather={1:1,2:1, 3:2, 4:2}
compl_month3['W_conditions']=compl_monthw['Weather_condition'].map(weather)


# In[167]:


compl_month3


# In[168]:


compl_month3 = compl_month3.drop(['Weather_condition'], axis=1)


# In[169]:


compl_month3.columns


# Final dataset with month as dummy (with clouds without w_conditions)

# In[170]:


compl_month_fin=pd.get_dummies(compl_monthc, columns=['Month'], drop_first=True)


# In[171]:


compl_month_fin


# In[172]:


compl_month_fin.columns


# In[173]:


compl_month_finh=pd.get_dummies(compl_month_fin, columns=['Hour'], drop_first=True)


# Final models with w_conditions, without clds

# In[522]:


compl_month_try=pd.get_dummies(compl_month3, columns=['Month'], drop_first=False)


# In[526]:


compl_month3.to_excel('C:\\Users\\Public\\bikesharing_Chicago\\compl_month3.xlsx')


# In[174]:


compl_month4=pd.get_dummies(compl_month3, columns=['Month'], drop_first=True)


# In[175]:


compl_month4.columns


# In[176]:


compl_month5=pd.get_dummies(compl_month4, columns=['Hour'], drop_first=True)


# In[177]:


compl_month5


# Trying out with seasons

# In[178]:


compl_season=complete.loc[:, complete.columns.drop(['Date1','Year','Date', 'wdir','wx_phrase', 'dewPt', 'clds', 'Holiday','feels_like', 'heat_index', 'Week-day', 'Weekend', 'Weekday-name', 'Month', 'Season', 'WeekdayName'])]


# In[59]:


compl_season


# In[60]:


compl_season['Count_log']= np.log1p(compl_season['Count'])


# In[61]:


compl_season= compl_season.drop(['Count'], axis=1)


# In[62]:


compl_season= compl_season.drop(['Day'], axis=1)


# In[63]:


#corelation matrix.
cor_mat= compl_season[:].corr()
mask = np.array(cor_mat)
mask[np.tril_indices_from(mask)] = False
fig=plt.gcf()
fig.set_size_inches(30,12)
sns.heatmap(data=cor_mat,mask=mask,square=True,annot=True,cbar=True)


# In[64]:


compl_seasonc= compl_season.drop(['Weather_condition'], axis=1)


# In[90]:


compl_seasonw= compl_season.drop(['Clouds'], axis=1)


# In[91]:


weather={1:1,2:1, 3:2, 4:2}
compl_seasonw['W_conditions']=compl_seasonw['Weather_condition'].map(weather)


# In[92]:


compl_seasonw= compl_seasonw.drop(['Weather_condition'], axis=1)


# In[106]:


compl_seasonw


# In[107]:


#complete.to_excel('C:\\Users\\Public\\bikesharing_Chicago\\datasetcomplete_v3.xlsx')


# In[108]:


#complete_v6 = pd.read_excel('C:\\Users\\Public\\bikesharing_Chicago\\datasetcomplete_v6.xlsx')


# **Train and testing models**
# 
# 

# In[625]:


from sklearn.model_selection import train_test_split, GridSearchCV,  cross_val_score
from sklearn import preprocessing, linear_model
from sklearn.preprocessing import  LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler 
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, mean_absolute_error
from sklearn.linear_model import Ridge, Lasso, LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor


# In[110]:


compl_month3


# In[111]:


X=compl_month3.drop('Count_log',axis=1)
y=compl_month3['Count_log']


# In[112]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# Robust Scaler or Minmax scaler

# In[451]:


def predict(ml_model,X,y):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=10)
    rob =RobustScaler()

    X_train = rob.fit_transform(X_train)
    X_test = rob.transform(X_test)

    model=ml_model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test,y_pred)
    plt.scatter(y_pred,y_test,color='#6daa9f')
    #cols = ['#f6685e', "#6daa9f"]

    plt.title('Gradient Boosting Model')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    print(f'R^2 is {model.score(X_test,y_test)}\n Adj R^2 is {1-(1-model.score(X_test,y_test))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)}\n RMSE is: {mean_squared_error(y_test,y_pred,squared=False)}')
    print("The mean aboslute error (MAE) on test set: {:.4f}".format(mae))
    print("The mean squared error (MSE) on test set: {:.4f}".format(mse))


# In[452]:


model_lm= predict(LinearRegression(),X,y)


# In[204]:


X=compl_month3.drop('Count_log',axis=1)
y=compl_month3['Count_log']


# In[213]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=10)
mm =MinMaxScaler()

X_train_std = mm.fit_transform(X_train)
X_test_std = mm.transform(X_test)

model = LinearRegression()
#
# Fit the model
#
model.fit(X_train_std, y_train)


importances = pd.DataFrame(data={
    'Attribute': X_train.columns,
    'Importance': model.coef_[0]
})
importances = importances.sort_values(by='Importance', ascending=False)


# In[116]:


predict(RandomForestRegressor(),X,y)


# In[117]:


predict(GradientBoostingRegressor(),X,y)


# **Try it with season!!**

# In[459]:


X1=compl_month3.drop('Count_log',axis=1)
y1=compl_month3['Count_log']


# In[119]:


compl_seasonw


# In[120]:


X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.20, random_state=42)


# Robust Scaler or Minmax scaler

# In[460]:


def predict1(ml_model,X1,y1):
    X_train,X_test,y_train,y_test=train_test_split(X1,y1,test_size=0.20,random_state=10)
    rob =RobustScaler()

    X_train = rob.fit_transform(X_train)
    X_test = rob.transform(X_test)

    model=ml_model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test,y_pred)
    plt.scatter(y_pred,y_test,color='#6daa9f')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    print(f'R^2 is {model.score(X_test,y_test)}\n Adj R^2 is {1-(1-model.score(X_test,y_test))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)}\n RMSE is: {mean_squared_error(y_test,y_pred,squared=False)}')
    print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
    print("The mean aboslute error (MAE) on test set: {:.4f}".format(mae))


# In[467]:


predict1(LinearRegression(),X1,y1)


# In[475]:


X_train,X_test,y_train,y_test=train_test_split(X1,y1,test_size=0.20,random_state=10)
mm =MinMaxScaler()

X_train = mm.fit_transform(X_train)
X_test = mm.transform(X_test)

model=GradientBoostingRegressor()
model=model.fit(X_train,y_train)
y_pred=model.predict(X_test)
plt.scatter(y_pred,y_test,color='#6daa9f')
plt.title('Gradient Boosting Model')
plt.xlabel('Predicted')
plt.ylabel('Actual')
print(f'R^2 is {model.score(X_test,y_test)}\n Adj R^2 is {1-(1-model.score(X_test,y_test))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)}\n RMSE is: {mean_squared_error(y_test,y_pred,squared=False)}')


# In[462]:


predict1(RandomForestRegressor(),X1,y1)


# In[463]:


predict1(GradientBoostingRegressor(),X,y)


# **Try it without holiday, with month**

# In[126]:


compl_month1= compl_month.drop(['Is_holiday'], axis=1)


# In[ ]:


compl_month1


# In[ ]:


X2=compl_month1.drop('Count_log',axis=1)
y2=compl_month1['Count_log']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.20, random_state=42)


# Robust Scaler or Minmax scaler

# In[ ]:


predict(LinearRegression(),X2,y2)


# In[ ]:


predict(RandomForestRegressor(),X2,y2)


# In[ ]:


predict_mm(RandomForestRegressor(),X2,y2)


# **Try without weather_condition!!**

# In[ ]:


compl_month2= compl_month.drop(['Weather_condition'], axis=1)


# In[ ]:


compl_month2


# In[ ]:


X3=compl_month2.drop('Count_log',axis=1)
y3=compl_month2['Count_log']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X3, y3, test_size=0.20, random_state=42)


# Robust Scaler or Minmax scaler

# In[ ]:


def predict1(ml_model,X3,y3):
    X_train,X_test,y_train,y_test=train_test_split(X3,y3,test_size=0.20,random_state=10)
    rob =RobustScaler()

    X_train = rob.fit_transform(X_train)
    X_test = rob.transform(X_test)

    model=ml_model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    plt.scatter(y_pred,y_test,color='b')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    print(f'R^2 is {model.score(X_test,y_test)}\n Adj R^2 is {1-(1-model.score(X_test,y_test))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)}\n RMSE is: {mean_squared_error(y_test,y_pred,squared=False)}')


# In[ ]:


predict(LinearRegression(),X3,y3)


# In[ ]:


predict(RandomForestRegressor(),X3,y3)


# In[ ]:


predict_mm(RandomForestRegressor(),X3,y3)


# **Try with weather condition with only two variables**

# In[ ]:


compl_month3


# In[ ]:


compl_month3['Count_log']= np.log1p(compl_month3['Count'])


# In[ ]:


#compl_month3=compl_month3.drop('Count',axis=1)


# In[510]:


X13=compl_month3.drop('Count_log',axis=1)
y13=compl_month3['Count_log']


# In[511]:


X_train, X_test, y_train, y_test = train_test_split(X13, y13, test_size=0.20, random_state=42)


# Robust Scaler or Minmax scaler

# In[512]:


def predict1(ml_model,X13,y13):
    X_train,X_test,y_train,y_test=train_test_split(X3,y3,test_size=0.20,random_state=10)
    rob =RobustScaler()

    X_train = rob.fit_transform(X_train)
    X_test = rob.transform(X_test)

    model=ml_model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    plt.scatter(y_pred,y_test,color='b')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    print(f'R^2 is {model.score(X_test,y_test)}\n Adj R^2 is {1-(1-model.score(X_test,y_test))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)}\n RMSE is: {mean_squared_error(y_test,y_pred,squared=False)}')


# In[ ]:


def predict_mm1(ml_model,X3,y3):
    X_train,X_test,y_train,y_test=train_test_split(X3,y3,test_size=0.20,random_state=10)
    mm =MinMaxScaler()

    X_train = mm.fit_transform(X_train)
    X_test = mm.transform(X_test)

    model=ml_model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    plt.scatter(y_pred,y_test,color='g')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    print(f'R^2 is {model.score(X_test,y_test)}\n Adj R^2 is {1-(1-model.score(X_test,y_test))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)}\n RMSE is: {mean_squared_error(y_test,y_pred,squared=False)}')


# In[513]:


predict(LinearRegression(),X13,y13)


# In[ ]:


predict(RandomForestRegressor(),X13,y13)


# In[ ]:


predict_mm(RandomForestRegressor(),X13,y13)


# **Try without is_weekend**

# In[ ]:


compl_month1= compl_month.drop(['Is_weekend'], axis=1)


# In[ ]:


compl_month1


# With month as dummy

# In[514]:


X4=compl_month4.drop('Count_log',axis=1)
y4=compl_month4['Count_log']


# In[515]:


X_train, X_test, y_train, y_test = train_test_split(X4, y4, test_size=0.20, random_state=42)


# Robust Scaler or Minmax scaler

# In[516]:


def predict1(ml_model,X4,y4):
    X_train,X_test,y_train,y_test=train_test_split(X4,y4,test_size=0.20,random_state=10)
    rob =RobustScaler()

    X_train = rob.fit_transform(X_train)
    X_test = rob.transform(X_test)

    model=ml_model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test,y_pred)
    plt.scatter(y_pred,y_test,color='b')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    print(f'R^2 is {model.score(X_test,y_test)}\n Adj R^2 is {1-(1-model.score(X_test,y_test))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)}\n RMSE is: {mean_squared_error(y_test,y_pred,squared=False)}')
    print("The mean aboslute error (MAE) on test set: {:.4f}".format(mae))
    print("The mean squared error (MSE) on test set: {:.4f}".format(mse))


# In[ ]:


def predict_mm1(ml_model,X4,y4):
    X_train,X_test,y_train,y_test=train_test_split(X4,y4,test_size=0.20,random_state=10)
    mm =MinMaxScaler()

    X_train = mm.fit_transform(X_train)
    X_test = mm.transform(X_test)

    model=ml_model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    plt.scatter(y_pred,y_test,color='g')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    print(f'R^2 is {model.score(X_test,y_test)}\n Adj R^2 is {1-(1-model.score(X_test,y_test))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)}\n RMSE is: {mean_squared_error(y_test,y_pred,squared=False)}')


# In[517]:


predict(LinearRegression(),X4,y4)


# In[ ]:


predict1(RandomForestRegressor(),X4,y4)


# In[ ]:


predict1(GradientBoostingRegressor(),X4,y4)


# with month and hour as dummy

# In[518]:


X4=compl_month5.drop('Count_log',axis=1)
y4=compl_month5['Count_log']


# In[519]:


X_train, X_test, y_train, y_test = train_test_split(X4, y4, test_size=0.20, random_state=42)


# Robust Scaler or Minmax scaler

# In[520]:


def predict1(ml_model,X4,y4):
    X_train,X_test,y_train,y_test=train_test_split(X4,y4,test_size=0.20,random_state=10)
    rob =RobustScaler()

    X_train = rob.fit_transform(X_train)
    X_test = rob.transform(X_test)

    model=ml_model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test,y_pred)
    plt.scatter(y_pred,y_test,color='b')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    print(f'R^2 is {model.score(X_test,y_test)}\n Adj R^2 is {1-(1-model.score(X_test,y_test))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)}\n RMSE is: {mean_squared_error(y_test,y_pred,squared=False)}')
    print("The mean aboslute error (MAE) on test set: {:.4f}".format(mae))
    print("The mean squared error (MSE) on test set: {:.4f}".format(mse))


# In[521]:


predict1(LinearRegression(),X4,y4)


# In[ ]:


predict1(RandomForestRegressor(),X4,y4)


# In[ ]:


predict1(GradientBoostingRegressor(),X4,y4)


# **Linear Regression**

# In[127]:


import statsmodels.api as sm
from stargazer.stargazer import Stargazer


# In[179]:


import statsmodels.api as sm

X8=compl_monthc.drop('Count_log',axis=1)
y8=compl_monthc['Count_log']

X8 = sm.add_constant(X8)

model1 = sm.OLS(y8, X8).fit()

#view model summary
print(model1.summary())


# Linear regression with the month as dummies

# In[180]:


X1=compl_month3.drop('Count_log',axis=1)
y1=compl_month3['Count_log']

X1 = sm.add_constant(X1)

model2 = sm.OLS(y1, X1).fit()

#view model summary
print(model2.summary())


# In[131]:


compl_month4.isna().sum()
#x_multi_cons.isna().sum()


# In[181]:


X1=compl_month4.drop('Count_log',axis=1)
y1=compl_month4['Count_log']

X1 = sm.add_constant(X1)

model3 = sm.OLS(y1, X1).fit()

#view model summary
print(model3.summary())


# In[182]:


X1=compl_month5.drop('Count_log',axis=1)
y1=compl_month5['Count_log']

X1 = sm.add_constant(X1)

model4 = sm.OLS(y1, X1).fit()

#view model summary
print(model4.summary())


# In[183]:


stargazer = Stargazer([model2, model3, model4])


# In[327]:


stargazer.title('OLS Regression Results')
stargazer.custom_columns(['Model 1', 'Model 2', 'Model 3'], [1, 1, 1])
stargazer.show_model_numbers(False)
stargazer.covariate_order(['const','Hour', 'Month',  'temp','Is_holiday', 'Is_weekend', 'W_conditions', 'precip_hrly', 'pressure', 'uv_index', 'wspd', 'Day'])

stargazer.add_custom_notes(['Model 2 - Omit: Month_dummy', 'Model 3 - Omit: Month_dummy + Hour_dummy'])

stargazer


# **GradientBoosting Parameters**

# In[ ]:


gbr = GradientBoostingRegressor()
gbr_params = {
    "n_estimators":[250,500,1000],
    "max_depth":[2,4,6],
    "learning_rate":[0.01,0.1,1],
    "loss": ['ls','huber','quantile'],
}


# In[ ]:


regressor = GridSearchCV(gbr, gbr_params, verbose=1,cv=3,n_jobs=-1) 
regressor.fit(X_train,y_train) 
#how to do your out of sample validation stategy? 
#what is cv= cross calidation, what should you put there? 


# In[ ]:


regressor.best_params_


# In[ ]:


regressor.best_estimator_     


# In[ ]:


predictions = regressor.predict(X_test) 


# In[244]:


from sklearn.metrics import mean_absolute_error, mean_squared_error

print('Root Mean Squared Error is {:.4f} '.format(np.sqrt(mean_squared_error(y_test,
                                                    regressor.predict(X_test)))) )
 


# In[ ]:


X5=compl_month4.drop(['Count_log'],axis=1)
y5=compl_month4['Count_log']

X_train, X_test, y_train, y_test = train_test_split(X5,y5, random_state=42, test_size=0.2)
#
# Standardize the dataset
#
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)
#
# Hyperparameters for GradientBoostingRegressor
#
gbr_params = {'n_estimators': 1000,
          'max_depth': 6,
          'min_samples_split': 10,
          'learning_rate': 0.1,
          'loss': 'ls'}
#
# Create an instance of gradient boosting regressor
#
gbr = GradientBoostingRegressor(**gbr_params)
#
# Fit the model
#
model = gbr.fit(X_train_std, y_train)

y_pred=model.predict(X_test_std)

#
# Print Coefficient of determination R^2
#
print("Model Accuracy: %.3f" % gbr.score(X_test_std, y_test))
#
# Create the mean squared error
#
mse = mean_squared_error(y_test, gbr.predict(X_test_std))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
print('Root Mean Squared Error is {:.4f} '.format(np.sqrt(mean_squared_error(y_test, y_pred, squared=False))))
                                                    


# The dataset with the month as dummies, it increases for 0.01

# In[ ]:


#with the dataset where weather conditon has only two categories
X11=compl_month4.drop(['Count_log'],axis=1)
y11=compl_month4['Count_log']

X_train, X_test, y_train, y_test = train_test_split(X11,y11, random_state=18, test_size=0.2)
#
# Standardize the dataset
#
mm =MinMaxScaler()

X_train = mm.fit_transform(X_train)
X_test = mm.transform(X_test)
#
# Hyperparameters for GradientBoostingRegressor
#
gbr_params = {'n_estimators': 1000,
          'max_depth': 7,
             'random_state' : 18}
#
# Create an instance of gradient boosting regressor
#
gbr = GradientBoostingRegressor(**gbr_params)
#
# Fit the model
#
model = gbr.fit(X_train, y_train)

y_pred=model.predict(X_test)

#
# Print Coefficient of determination R^2
#
print("Model Accuracy: %.3f" % gbr.score(X_test, y_test))
#
# Create the mean squared error
#
mse = mean_squared_error(y_test, gbr.predict(X_test))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
print('Root Mean Squared Error is {:.4f} '.format(np.sqrt(mean_squared_error(y_test, y_pred, squared=False))))
                                                    


# In[ ]:


#with the dataset where weather conditon has only two categories
X11=compl_month5.drop(['Count_log'],axis=1)
y11=compl_month5['Count_log']

X_train, X_test, y_train, y_test = train_test_split(X11,y11, random_state=18, test_size=0.2)
#
# Standardize the dataset
#
mm =MinMaxScaler()

X_train = mm.fit_transform(X_train)
X_test = mm.transform(X_test)
#
# Hyperparameters for GradientBoostingRegressor
#
gbr_params = {'n_estimators': 1000,
          'max_depth': 7,
             'random_state' : 18}
#
# Create an instance of gradient boosting regressor
#
gbr = GradientBoostingRegressor(**gbr_params)
#
# Fit the model
#
model = gbr.fit(X_train, y_train)

y_pred=model.predict(X_test)

#
# Print Coefficient of determination R^2
#
print("Model Accuracy: %.3f" % gbr.score(X_test, y_test))
#
# Create the mean squared error
#
mse = mean_squared_error(y_test, gbr.predict(X_test))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
print('Root Mean Squared Error is {:.4f} '.format(np.sqrt(mean_squared_error(y_test, y_pred, squared=False))))
                                                    


# In[295]:


#with the dataset where weather conditon has only two categories
X11=compl_month3.drop(['Count_log'],axis=1)
y11=compl_month3['Count_log']

X_train, X_test, y_train, y_test = train_test_split(X11,y11, random_state=18, test_size=0.2)
#
# Standardize the dataset
#
mm =StandardScaler()

X_train = mm.fit_transform(X_train)
X_test = mm.transform(X_test)
#
# Hyperparameters for GradientBoostingRegressor
#
gbr_params = {'n_estimators': 1000,
          'max_depth': 7,
             'random_state' : 18,
              'loss': 'ls'}
#
# Create an instance of gradient boosting regressor
#
gbr = GradientBoostingRegressor(**gbr_params)
#
# Fit the model
#
model = gbr.fit(X_train, y_train)

y_pred=model.predict(X_test)

#
# Print Coefficient of determination R^2
#
print("Model Accuracy: %.3f" % gbr.score(X_test, y_test))
#
# Create the mean squared error
#
mse = mean_squared_error(y_test, gbr.predict(X_test))
mae = mean_absolute_error(y_test, gbr.predict(X_test))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
print('Root Mean Squared Error is {:.4f} '.format(np.sqrt(mean_squared_error(y_test, y_pred, squared=False))))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
print("The mean aboslute error (MAE) on test set: {:.4f}".format(mae))                                                    
print(1-(1-model.score(X_test,y_test))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))
feature_importance = model.feature_importances_


# In[ ]:


#with the dataset where weather conditon has only two categories
X11=compl_month_finh.drop(['Count_log'],axis=1)
y11=compl_month_finh['Count_log']

X_train, X_test, y_train, y_test = train_test_split(X11,y11, random_state=18, test_size=0.2)
#
# Standardize the dataset
#
mm =MinMaxScaler()

X_train = mm.fit_transform(X_train)
X_test = mm.transform(X_test)
#
# Hyperparameters for GradientBoostingRegressor
#
gbr_params = {'n_estimators': 1000,
          'max_depth': 7,
             'random_state' : 18}
#
# Create an instance of gradient boosting regressor
#
gbr = GradientBoostingRegressor(**gbr_params)
#
# Fit the model
#
model = gbr.fit(X_train, y_train)

y_pred=model.predict(X_test)

#
# Print Coefficient of determination R^2
#
print("Model Accuracy: %.3f" % gbr.score(X_test, y_test))
#
# Create the mean squared error
#
mse = mean_squared_error(y_test, gbr.predict(X_test))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
print('Root Mean Squared Error is {:.4f} '.format(np.sqrt(mean_squared_error(y_test, y_pred, squared=False))))


# In[ ]:


gbr = GradientBoostingRegressor()
gbr_params = {
    "n_estimators":[250,500,1000],
    "max_depth":[2,4,6],
    "learning_rate":[0.01,0.1,1],
    "loss": ['ls','huber','quantile'],
}


# In[ ]:


regressor = GridSearchCV(gbr, gbr_params, verbose=1,cv=10,n_jobs=-1) 
regressor.fit(X_train,y_train) 
#how to do your out of sample validation stategy? 
#what is cv= cross calidation, what should you put there? 


# In[ ]:


regressor.best_params_


# In[ ]:


regressor.best_estimator_     


# In[ ]:


#with the dataset where weather conditon has only two categories
X20=compl_month3.drop(['Count_log'],axis=1)
y20=compl_month3['Count_log']

X_train, X_test, y_train, y_test = train_test_split(X20,y20, random_state=18, test_size=0.2)
#
# Standardize the dataset
#
mm =MinMaxScaler()

X_train = mm.fit_transform(X_train)
X_test = mm.transform(X_test)
#
# Hyperparameters for GradientBoostingRegressor
#
gbr_params = {'n_estimators': 1000,
          'max_depth': 7,
             'random_state' : 18,
                'loss': 'ls'}

#
# Create an instance of gradient boosting regressor
#
gbr = GradientBoostingRegressor(**gbr_params)
#
# Fit the model
#
model = gbr.fit(X_train, y_train)

y_pred=model.predict(X_test)

#
# Print Coefficient of determination R^2
#
print("Model Accuracy: %.3f" % gbr.score(X_test, y_test))
#
# Create the mean squared error
#
mse = mean_squared_error(y_test, gbr.predict(X_test))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
print('Root Mean Squared Error is {:.4f} '.format(np.sqrt(mean_squared_error(y_test, y_pred))))
                                                    

feature_importance = model.feature_importances_


predictions = model.predict(X_test[:5])
print("Predicted values are: ", predictions)
print("Real values are:", y_test[:5])


# In[ ]:


feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, compl_month.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()


# **Random Forest Parameters**

# In[ ]:


rm = RandomForestRegressor()

rm_params = { 
    'n_estimators': [200,300,500,1000],
    'max_depth' : [3,4,5,6,7],
    'random_state' : [0,18,42]
}


# In[ ]:


regressor_rm = GridSearchCV(rm, rm_params, verbose=1,cv=3,n_jobs=-1) 
regressor_rm.fit(X_train,y_train) 


# In[ ]:


regressor_rm.best_params_


# In[ ]:


regressor_rm.best_estimator_     


# In[ ]:


predictions_rm = regressor_rm.predict(X_test) 


# In[ ]:


print('Root Mean Squared Error is {:.4f} '.format(np.sqrt(mean_squared_error(y_test,
                                                    predictions_rm, squared=False)))) 
 


# In[ ]:


X10=compl_month.drop(['Count_log'],axis=1)
y10=compl_month['Count_log']

X_train, X_test, y_train, y_test = train_test_split(X10,y10, random_state=18, test_size=0.2)
#
# Standardize the dataset
#
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)
#
# Hyperparameters for GradientBoostingRegressor
#
rm_params = {'n_estimators': 500,
          'max_depth': 7,
             'random_state' : 18}
#
# Create an instance of gradient boosting regressor
#
rm = RandomForestRegressor(**rm_params)
#
# Fit the model
#
model1 = rm.fit(X_train_std, y_train)

y_pred=model1.predict(X_test)

#
# Print Coefficient of determination R^2
#
print("Model Accuracy: %.3f" % rm.score(X_test_std, y_test))
#
# Create the mean squared error
#
mse = mean_squared_error(y_test, rm.predict(X_test_std))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
print('Root Mean Squared Error is {:.4f} '.format(np.sqrt(mean_squared_error(y_test, y_pred, squared=False))))

feature_importance = model1.feature_importances_
                                                    


# In[226]:


X21=compl_month4.drop(['Count_log'],axis=1)
y21=compl_month4

['Count_log']

X_train, X_test, y_train, y_test = train_test_split(X21,y21, random_state=18, test_size=0.2)
#
# Standardize the dataset
#
mm =MinMaxScaler()

X_train = mm.fit_transform(X_train)
X_test = mm.transform(X_test)
#
# Hyperparameters for GradientBoostingRegressor
#
rm_params = {'n_estimators': 500,
             'random_state' : 18}
#
# Create an instance of gradient boosting regressor
#
rm = RandomForestRegressor(**rm_params)
#
# Fit the model
#
model = rm.fit(X_train, y_train)

y_pred=model.predict(X_test)

#
# Print Coefficient of determination R^2
#
print("Model Accuracy: %.3f" % rm.score(X_test, y_test))
#
# Create the mean squared error
#
mse = mean_squared_error(y_test, rm.predict(X_test))
mae = mean_absolute_error(y_test, rm.predict(X_test))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
print("The mean squared error (MSE) on test set: {:.4f}".format(mae))
print('Root Mean Squared Error is {:.4f} '.format(np.sqrt(mean_squared_error(y_test, y_pred, squared=False))))

print(1-(1-model.score(X_test,y_test))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))
                                                    
feature_importance = model.feature_importances_


# In[ ]:


X12=compl_month3.drop(['Count_log'],axis=1)
y12=compl_month3['Count_log']

X_train, X_test, y_train, y_test = train_test_split(X12,y12, random_state=42, test_size=0.2)
#
# Standardize the dataset
#
mm =MinMaxScaler()

X_train = mm.fit_transform(X_train)
X_test = mm.transform(X_test)
#
# Hyperparameters for GradientBoostingRegressor
#
rm_params = {'n_estimators': 1000}
#
# Create an instance of gradient boosting regressor
#
rm = RandomForestRegressor(**rm_params)
#
# Fit the model
#
model = rm.fit(X_train, y_train)

y_pred=model.predict(X_test)

#
# Print Coefficient of determination R^2
#
print("Model Accuracy: %.3f" % rm.score(X_test, y_test))
#
# Create the mean squared error
#
mse = mean_squared_error(y_test, rm.predict(X_test))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
print('Root Mean Squared Error is {:.4f} '.format(np.sqrt(mean_squared_error(y_test, y_pred, squared=False))))
                                                    


# In[ ]:


feature_importance2 = 100.0 * (feature_importance2 / feature_importance2.max())
sorted_idx = np.argsort(feature_importance2)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.barh(pos, feature_importance2[sorted_idx], align='center')
plt.yticks(pos, compl_month.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()


# In[133]:


#with month as dummy
X26=compl_month4.drop(['Count_log'],axis=1)
y26=compl_month4['Count_log']

X_train, X_test, y_train, y_test = train_test_split(X26,y26, random_state=10, test_size=0.2)
#
# Standardize the dataset

rob =RobustScaler()

X_train = rob.fit_transform(X_train)
X_test = rob.transform(X_test)

#
rm = RandomForestRegressor()
#
# Fit the model
#
model2 = rm.fit(X_train, y_train)

y_pred=model2.predict(X_test)

#
# Print Coefficient of determination R^2
#
print("Model Accuracy: %.3f" % rm.score(X_test, y_test))
#
# Create the mean squared error
#
mse = mean_squared_error(y_test, rm.predict(X_test))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
print('Root Mean Squared Error is {:.4f} '.format(np.sqrt(mean_squared_error(y_test, y_pred, squared=False))))
                                                    


# In[ ]:


#with month as dummy
X26=compl_month5.drop(['Count_log'],axis=1)
y26=compl_month5['Count_log']

X_train, X_test, y_train, y_test = train_test_split(X26,y26, random_state=10, test_size=0.2)
#
# Standardize the dataset

rob =RobustScaler()

X_train = rob.fit_transform(X_train)
X_test = rob.transform(X_test)

#
rm = RandomForestRegressor()
#
# Fit the model
#
model2 = rm.fit(X_train, y_train)

y_pred=model2.predict(X_test)

#
# Print Coefficient of determination R^2
#
print("Model Accuracy: %.3f" % rm.score(X_test, y_test))
#
# Create the mean squared error
#
mse = mean_squared_error(y_test, rm.predict(X_test))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
print('Root Mean Squared Error is {:.4f} '.format(np.sqrt(mean_squared_error(y_test, y_pred, squared=False))))
                                                    


# In[503]:


import lightgbm as lgb
from lightgbm import LGBMRegressor


# In[137]:


predict_mm(LGBMRegressor(),X,y)


# In[499]:


import xgboost as xgb
from xgboost import XGBRegressor


# In[139]:


predict_mm(XGBRegressor(),X,y)


# In[279]:


X12=compl_month4.drop(['Count_log'],axis=1)
y12=compl_month4['Count_log']

X_train, X_test, y_train, y_test = train_test_split(X12,y12, random_state=42, test_size=0.2)
#
# Standardize the dataset
#
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#
# Hyperparameters for GradientBoostingRegressor
#
lgbr_params = {'n_estimators': 1000,
          'max_depth': 6,
           #    'max_bin': 255,
              'num_leaves': 31,
          'learning_rate': 0.1}
     #     'num_iterations' : 100
      #    'loss': 'ls'}
#
# Create an instance of gradient boosting regressor
#
lgbr = LGBMRegressor(**lgbr_params)
#
# Fit the model
#
model = lgbr.fit(X_train_std, y_train)

y_pred=model.predict(X_test)

#
# Print Coefficient of determination R^2
#
print("Model Accuracy: %.3f" % lgbr.score(X_test, y_test))
#
# Create the mean squared error
#
mse = mean_squared_error(y_test, lgbr.predict(X_test))
mae = mean_absolute_error(y_test, lgbr.predict(X_test))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
print("The mean squared error (MSE) on test set: {:.4f}".format(mae))
print('Root Mean Squared Error is {:.4f} '.format(np.sqrt(mean_squared_error(y_test, y_pred, squared=False))))

print(1-(1-lgbr.score(X_test,y_test))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))
                                                    
feature_importance = lgbr.feature_importances_                                                   


# In[280]:


feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, X_train.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('LGBM- Feature Importance', size='12')
plt.show()


# **XGBM**

# In[ ]:


xgbr = XGBRegressor()
xgbr_params = {
    "n_estimators":[250,500,1000],
    "max_depth":[2,4,6],
    "learning_rate":[0.01,0.1,1],
    "loss": ['ls','huber','quantile'],
}


# In[ ]:


regressor = GridSearchCV(xgbr, xgbr_params, verbose=1,cv=3,n_jobs=-1) 
regressor.fit(X_train,y_train) 


# In[ ]:


regressor.best_params_


# In[ ]:


regressor.best_estimator_     


# In[ ]:


predictions = regressor.predict(X_test) 


# In[294]:


X14=compl_month3.drop(['Count_log'],axis=1)
y14=compl_month3['Count_log']

X_train, X_test, y_train, y_test = train_test_split(X14,y14, random_state=42, test_size=0.2)
#
# Standardize the dataset
#
mm =MinMaxScaler()

X_train_std = mm.fit_transform(X_train)
X_test = mm.transform(X_test)

#
# Hyperparameters for GradientBoostingRegressor
#
xgbr_params = {'n_estimators': 1000,
          'max_depth': 6,
              # 'max_bin': 255,
               #'num_leaves': 31,
          'learning_rate': 0.1}
     #     'num_iterations' : 100
      #    'loss': 'ls'}
#
# Create an instance of gradient boosting regressor
#
xgbr = XGBRegressor(**xgbr_params)
#
# Fit the model
#
model = xgbr.fit(X_train_std, y_train)

y_pred=model.predict(X_test)

#
# Print Coefficient of determination R^2
#
print("Model Accuracy: %.3f" % xgbr.score(X_test, y_test))
#
# Create the mean squared error
#
mse = mean_squared_error(y_test, xgbr.predict(X_test))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
print('Root Mean Squared Error is {:.4f} '.format(np.sqrt(mean_squared_error(y_test, y_pred, squared=False))))
                                                    


# In[623]:


X24=compl_month4.drop(['Count_log'],axis=1)
y24=compl_month4['Count_log']

X_train, X_test, y_train, y_test = train_test_split(X24,y24, random_state=42, test_size=0.2)
#
# Standardize the dataset
#
mm =StandardScaler()

X_train = mm.fit_transform(X_train)
X_test = mm.transform(X_test)

#
# Hyperparameters for GradientBoostingRegressor
#
xgbr_params = {'n_estimators': 1000,
          'max_depth': 6,
              # 'max_bin': 255,
               #'num_leaves': 31,
          'learning_rate': 0.1}
     #     'num_iterations' : 100
      #  'loss': 'ls'}
#
# Create an instance of gradient boosting regressor
#
xgbr = XGBRegressor(**xgbr_params)
#
# Fit the model
#
model = xgbr.fit(X_train, y_train)

y_pred=model.predict(X_test)

#
# Print Coefficient of determination R^2
#
print("Model Accuracy: %.3f" % xgbr.score(X_test, y_test))
#
# Create the mean squared error
#
mse = mean_squared_error(y_test, xgbr.predict(X_test))
mae = mean_absolute_error(y_test, xgbr.predict(X_test))

print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
print("The mean squared error (MSE) on test set: {:.4f}".format(mae))
print('Root Mean Squared Error is {:.4f} '.format(np.sqrt(mean_squared_error(y_test, y_pred, squared=False))))
print(1-(1-xgbr.score(X_test,y_test))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))

accuracy_score(y_test, y_pred)


# **Artifical Neural Network for Regression**

# In[500]:


import os 
import tensorflow as tf
from tensorflow import keras
from sklearn import metrics


# In[501]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, BatchNormalization, Dropout
from tensorflow.keras import optimizers
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline


# In[502]:


from tensorflow.keras.callbacks import EarlyStopping


# In[ ]:


X25=compl_month3.drop(['Count_log'],axis=1)
y25=compl_month3['Count_log']

X_train, X_test, y_train, y_test = train_test_split(X25, y25, test_size=0.3, random_state=42)

# Quick sanity check with the shapes of Training and testing datasets
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[ ]:


#Predictions=model3.predict(x_val)


# In[ ]:


#pd.DataFrame(model.history.history)


# In[ ]:


#plt.style.use("ggplot")
#pd.DataFrame(model.history.history).plot(figsize=(12,10))


# In[245]:


#metrics.explained_variance_score(y_val, Predictions)


# In[253]:


print(Predictions1.shape)
print(y_test.shape)


# Trying models with only last 20 percent of the data

# In[ ]:


X=compl_month_fin.drop(['Count_log'],axis=1)
y=compl_month_fin['Count_log']

X_train = X.head(7603)
X_test = X.tail(1901)
y_train = y.head(7603)
y_test = y.tail(1901)


mm =MinMaxScaler()

X_train = mm.fit_transform(X_train)
X_test = mm.transform(X_test)
#
# Hyperparameters for GradientBoostingRegressor
#
gbr_params = {'n_estimators': 1000,
          'max_depth': 7,
             'random_state' : 18}
#
# Create an instance of gradient boosting regressor
#
gbr = GradientBoostingRegressor(**gbr_params)
#
# Fit the model
#
model = gbr.fit(X_train, y_train)

y_pred=model.predict(X_test)

#
# Print Coefficient of determination R^2
#
print("Model Accuracy: %.3f" % gbr.score(X_test, y_test))
#
# Create the mean squared error
#
mse = mean_squared_error(y_test, gbr.predict(X_test))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
print('Root Mean Squared Error is {:.4f} '.format(np.sqrt(mean_squared_error(y_test, y_pred, squared=False))))


# In[ ]:


compl_month_fin


# In[ ]:


X=compl_month_fin.drop(['Count_log'],axis=1)
y=compl_month_fin['Count_log']

X_train = X.head(7603)
X_test = X.tail(1901)
y_train = y.head(7603)
y_test = y.tail(1901)

# create ANN model
model3 = Sequential()
 
# Defining the Input layer and FIRST hidden layer, both are same!
model3.add(Dense(128, input_dim=22, activation='relu'))
 
# Defining the Second layer of the model
# after the first layer we don't have to specify input_dim as keras configure it automatically
model3.add(Dense(74, activation='relu'))
 
# The output neuron is a single fully connected node 
# Since we will be predicting a single number
model3.add(Dense(1,  kernel_initializer='normal'))
 
# Compiling the model
model3.compile(loss='mean_squared_error', optimizer='adam')
 
# Fitting the ANN to the Training set
model3.fit(X_train, y_train, batch_size = 20, epochs = 50, verbose=1)

Predictions1=model3.predict(X_test)

metrics.explained_variance_score(y_test, Predictions1)


# **FINAL Models**

# In[488]:


X=compl_month4.drop(['Count_log'],axis=1)
y=compl_month4['Count_log']

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42, test_size=0.2)


# In[489]:


mm =RobustScaler()

X_train_std = mm.fit_transform(X_train)
X_test = mm.transform(X_test)


# In[485]:




# Create an instance of gradient boosting regressor
#
lm = LinearRegression()
#
# Fit the model
#
model = lm.fit(X_train_std, y_train)

y_pred=model.predict(X_test)

#
# Print Coefficient of determination R^2
#
print("Model Accuracy: %.3f" % lm.score(X_test, y_test))
#
# Create the mean squared error
#
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test,y_pred)
print("The mean aboslute error (MAE) on test set: {:.4f}".format(mae))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))

print('Root Mean Squared Error is {:.4f} '.format(np.sqrt(mean_squared_error(y_test, y_pred, squared=False))))

feature_importance = lm.coef_ 


# In[486]:


feature_importance= abs(feature_importance)
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.barh(pos, feature_importance[sorted_idx], align='center', color='#b6d4cf')
plt.yticks(pos, X_train.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('LM- Feature Importance', size='12')
plt.show()


# In[487]:



feature_importance= abs(feature_importance)
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.barh(pos, feature_importance[sorted_idx], align='center', color='#98c3bb')
plt.yticks(pos, X_train.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('LM- Feature Importance', size='12')
plt.show()


# In[492]:



rm = RandomForestRegressor()
#
# Fit the model
#
model2 = rm.fit(X_train_std, y_train)

y_pred=model2.predict(X_test)

#
# Print Coefficient of determination R^2
#
print("Model Accuracy: %.3f" % rm.score(X_test, y_test))
#
# Create the mean squared error
#
mse = mean_squared_error(y_test, rm.predict(X_test))
mae = mean_absolute_error(y_test, rm.predict(X_test))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
print("The mean aboslute error (MAE) on test set: {:.4f}".format(mae))                                                    
#print(1-(1-model.score(X_test,y_test))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))
print('Root Mean Squared Error is {:.4f} '.format(np.sqrt(mean_squared_error(y_test, y_pred, squared=False))))


feature_importance2 = model2.feature_importances_


# In[495]:


feature_importance2 = 100.0 * (feature_importance2 / feature_importance2.max())
sorted_idx = np.argsort(feature_importance2)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.barh(pos, feature_importance2[sorted_idx], align='center', color='#f9a49e')
plt.yticks(pos, X_train.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('RF- Feature Importance', size='12')
plt.show()


# In[496]:



# Hyperparameters for GradientBoostingRegressor
#
gbr_params = {'n_estimators': 1000,
          'max_depth': 7,
             'random_state' : 18}
#
# Create an instance of gradient boosting regressor
#
gbr = GradientBoostingRegressor(**gbr_params)
#
# Fit the model
#
model = gbr.fit(X_train_std, y_train)

y_pred=model.predict(X_test)

#
# Print Coefficient of determination R^2
#
print("Model Accuracy: %.3f" % gbr.score(X_test, y_test))
#
mse = mean_squared_error(y_test, gbr.predict(X_test))
mae = mean_absolute_error(y_test, gbr.predict(X_test))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
print('Root Mean Squared Error is {:.4f} '.format(np.sqrt(mean_squared_error(y_test, y_pred, squared=False))))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
print("The mean aboslute error (MAE) on test set: {:.4f}".format(mae))                                                    
print(1-(1-model.score(X_test,y_test))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))

GBM_importance = gbr.feature_importances_


# In[497]:


GBM_importance = 100.0 * (GBM_importance / GBM_importance.max())
sorted_idx = np.argsort(GBM_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.barh(pos, GBM_importance[sorted_idx], align='center', color='#f7867e')
plt.yticks(pos, X_train.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('GBM- Feature Importance', size='12')
plt.show()


# In[504]:



lgbr_params = {'n_estimators': 1000,
          'max_depth': 6,
           #    'max_bin': 255,
              'num_leaves': 31,
          'learning_rate': 0.1}
     #     'num_iterations' : 100
      #    'loss': 'ls'}
#
# Create an instance of gradient boosting regressor
#
lgbr = LGBMRegressor(**lgbr_params)
#
# Fit the model
#
model = lgbr.fit(X_train_std, y_train)

y_pred=model.predict(X_test)

#
# Print Coefficient of determination R^2
#
print("Model Accuracy: %.3f" % lgbr.score(X_test, y_test))
#
# Create the mean squared error
#
mse = mean_squared_error(y_test, lgbr.predict(X_test))
mae = mean_absolute_error(y_test, lgbr.predict(X_test))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
print("The mean squared error (MSE) on test set: {:.4f}".format(mae))
print('Root Mean Squared Error is {:.4f} '.format(np.sqrt(mean_squared_error(y_test, y_pred, squared=False))))

print(1-(1-lgbr.score(X_test,y_test))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))
                                                    
feature_importance = lgbr.feature_importances_                                                   


# In[505]:


feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.barh(pos, feature_importance[sorted_idx], align='center', color='#8abbb2')
plt.yticks(pos, X_train.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('LGBM- Feature Importance', size='12')
plt.show()


# In[ ]:





# In[507]:



xgbr_params = {'n_estimators': 1000,
          'max_depth': 6,
              # 'max_bin': 255,
               #'num_leaves': 31,
          'learning_rate': 0.1}
     #     'num_iterations' : 100
      #  'loss': 'ls'}
#
# Create an instance of gradient boosting regressor
#
xgbr = XGBRegressor(**xgbr_params)
#
# Fit the model
#
model = xgbr.fit(X_train_std, y_train)

y_pred=model.predict(X_test)

#
# Print Coefficient of determination R^2
#
print("Model Accuracy: %.3f" % xgbr.score(X_test, y_test))
#
# Create the mean squared error
#
mse = mean_squared_error(y_test, xgbr.predict(X_test))
mae = mean_absolute_error(y_test, xgbr.predict(X_test))

print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
print("The mean squared error (MSE) on test set: {:.4f}".format(mae))
print('Root Mean Squared Error is {:.4f} '.format(np.sqrt(mean_squared_error(y_test, y_pred, squared=False))))
print(1-(1-xgbr.score(X_test,y_test))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))

feature_importance = xgbr.feature_importances_                                                   


# In[508]:


feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.barh(pos, feature_importance[sorted_idx], align='center', color='#6daa9f')
plt.yticks(pos, X_train.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('XGBM- Feature Importance', size='12')
plt.show()


# In[ ]:





# In[312]:


X=compl_month4.drop(['Count_log'],axis=1)
y=compl_month4['Count_log']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dtypes = list(zip(X.dtypes.index, map(str, X.dtypes)))
for k,dtype in dtypes:
    if dtype == "float32":
        X[k] -= X[k].mean()
        X[k] /= X[k].std()

# Quick sanity check with the shapes of Training and testing datasets
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)



# In[236]:


X.dtypes


# In[238]:


from sklearn.neural_network import MLPRegressor


# In[ ]:


import shap


# In[242]:


model5 = MLPRegressor()
model5.fit(X_train, y_train)
print(model5)

expected_y  = y_test
predicted_y = model.predict(X_test)


# In[247]:


explainer = shap.KernelExplainer(model5.predict, X_train_summary)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)


# In[509]:


vals= np.abs(shap_values).mean(0)
feature_importance = pd.DataFrame(list(zip(X_train.columns,vals)),columns=['col_name','feature_importance_vals'])
feature_importance['feature_importance_vals'] = 100.0 * (feature_importance['feature_importance_vals'] / feature_importance['feature_importance_vals'].max())
#feature_importance.sort_values(by=['feature_importance_vals'],ascending=False,inplace=True)
#feature_importance
feature_importance = feature_importance.sort_values('feature_importance_vals', axis=0, ascending=True)

#sorted_idx = np.argsort(feature_importance)
#pos = np.arange(sorted_idx.shape[0]) + .5
plt.barh(feature_importance['col_name'], feature_importance['feature_importance_vals'], color= '#f6685e')
plt.xlabel('Relative Importance', fontsize=12)
plt.title('MLP Regressor- Feature Importance', size='12')
plt.grid(True)
plt.show()


# In[255]:


feature_importance([feature_importance_vals]) #= 100.0 * (feature_importance[feature_importance_vals] / feature_importance[feature_importance_vals].max())


# In[ ]:


sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, X_train.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('XGBM- Feature Importance', size='12')
plt.show()


# In[239]:


nn = MLPRegressor(solver='lbfgs', alpha=1e-1, hidden_layer_sizes=(5, 2), random_state=0)
nn.fit(X_train, y_train)


# In[269]:



# create ANN model
#model3 = Sequential()
model3 = MLPRegressor()
#model5.fit(X_train, y_train)
#print(model5)

expected_y  = y_test
predicted_y = model.predict(X_test)
 
# Defining the Input layer and FIRST hidden layer, both are same!
model3.add(Dense(128, input_dim=21, activation='relu'))
 
# Defining the Second layer of the model
# after the first layer we don't have to specify input_dim as keras configure it automatically
model3.add(Dense(74, activation='relu'))
 
# The output neuron is a single fully connected node 
# Since we will be predicting a single number
model3.add(Dense(1,  kernel_initializer='normal'))
 
# Compiling the model
model3.compile(loss='mean_squared_error', optimizer='adam')
 
# Fitting the ANN to the Training set
model3.fit(X_train, y_train, batch_size = 20, epochs = 50, verbose=1)

#y_pred=model.predict(X_test)
#ann.fit(x=X_train, y=y_train, epochs=100, batch_size=32,validation_data=(X_test,y_test),callbacks=EarlyStopping(monitor='val_loss',patience=4))


# In[318]:


model3 = MLPRegressor(hidden_layer_sizes=128, activation='tanh', solver='adam',
                         max_iter=100, learning_rate_init=0.001, batch_size=30)
model3.fit(X_train, y_train)



# In[319]:


Predictions1=model3.predict(X_test)


# In[320]:


print("Accuracy:", metrics.explained_variance_score(y_test, Predictions1))
print("MAE:",metrics.mean_absolute_error(y_test,Predictions1))
print ("MSE:",metrics.mean_squared_error(y_test,Predictions1))
print("RMSE:",np.sqrt(metrics.mean_squared_error(y_test,Predictions1)))


# In[ ]:


final_score = metrics.explained_variance_score(y_test, Predictions1)
MAE = metrics.mean_absolute_error(y_test,Predictions1)


# **Predict the last 20 percent of the data**

# In[537]:


X=compl_month4.drop(['Count_log'],axis=1)
y=compl_month4['Count_log']

X_train = X.head(7603)
X_test = X.tail(1901)
y_train = y.head(7603)
y_test = y.tail(1901)


# In[538]:




def predict(ml_model,X,y):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=10)
    rob =RobustScaler()

    X_train = rob.fit_transform(X_train)
    X_test = rob.transform(X_test)

    model=ml_model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test,y_pred)
    plt.scatter(y_pred,y_test,color='#6daa9f')
    #cols = ['#f6685e', "#6daa9f"]

    plt.title('Gradient Boosting Model')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    print(f'R^2 is {model.score(X_test,y_test)}\n Adj R^2 is {1-(1-model.score(X_test,y_test))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)}\n RMSE is: {mean_squared_error(y_test,y_pred,squared=False)}')
    print("The mean aboslute error (MAE) on test set: {:.4f}".format(mae))
    print("The mean squared error (MSE) on test set: {:.4f}".format(mse))


# In[539]:


predict(LinearRegression(),X,y)


# In[352]:


X=compl_month4.drop(['Count_log'],axis=1)
y=compl_month4['Count_log']

X_train = X.head(7603)
X_test = X.tail(1901)
y_train = y.head(7603)
y_test = y.tail(1901)

mm =RobustScaler()

X_train = mm.fit_transform(X_train)
X_test = mm.transform(X_test)

rm = RandomForestRegressor()
#
# Fit the model
#
model2 = rm.fit(X_train, y_train)

y_pred=model2.predict(X_test)

#
# Print Coefficient of determination R^2
#
print("Model Accuracy: %.3f" % rm.score(X_test, y_test))
#
# Create the mean squared error
#
mse = mean_squared_error(y_test, rm.predict(X_test))
mae = mean_absolute_error(y_test, rm.predict(X_test))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
print("The mean aboslute error (MAE) on test set: {:.4f}".format(mae))                                                    
#print(1-(1-model.score(X_test,y_test))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))
print('Root Mean Squared Error is {:.4f} '.format(np.sqrt(mean_squared_error(y_test, y_pred, squared=False))))


feature_importance2 = model2.feature_importances_


# In[361]:


X_train = X.head(7603)
X_test = X.tail(1901)
y_train = y.head(7603)
y_test = y.tail(1901)

mm =RobustScaler()

X_train = mm.fit_transform(X_train)
X_test = mm.transform(X_test)


rm_params = {'n_estimators': 500,
             'random_state' : 18}
#
# Create an instance of gradient boosting regressor
#
rm = RandomForestRegressor(**rm_params)

#
# Fit the model
#
model2 = rm.fit(X_train, y_train)

y_pred=model2.predict(X_test)

#
# Print Coefficient of determination R^2
#
print("Model Accuracy: %.3f" % rm.score(X_test, y_test))
#
# Create the mean squared error
#
mse = mean_squared_error(y_test, rm.predict(X_test))
mae = mean_absolute_error(y_test, rm.predict(X_test))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
print("The mean aboslute error (MAE) on test set: {:.4f}".format(mae))                                                    
print(1-(1-model2.score(X_test,y_test))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))
print('Root Mean Squared Error is {:.4f} '.format(np.sqrt(mean_squared_error(y_test, y_pred, squared=False))))


# In[629]:


X=compl_month4.drop(['Count_log'],axis=1)
y=compl_month4['Count_log']

X_train = X.head(7603)
X_test = X.tail(1901)
y_train = y.head(7603)
y_test = y.tail(1901)

rob =RobustScaler()

X_train = rob.fit_transform(X_train)
X_test = rob.transform(X_test)
# Hyperparameters for GradientBoostingRegressor
#
gbr_params = {'n_estimators': 1000,
          'max_depth': 7,
             'random_state' : 18}
#
# Create an instance of gradient boosting regressor
#
gbr = GradientBoostingRegressor(**gbr_params)
#
# Fit the model
#
model = gbr.fit(X_train, y_train)

y_pred=model.predict(X_test)

#
# Print Coefficient of determination R^2
#
print("Model Accuracy: %.3f" % gbr.score(X_test, y_test))
#
mse = mean_squared_error(y_test, gbr.predict(X_test))
mae = mean_absolute_error(y_test, gbr.predict(X_test))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
print('Root Mean Squared Error is {:.4f} '.format(np.sqrt(mean_squared_error(y_test, y_pred, squared=False))))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
print("The mean aboslute error (MAE) on test set: {:.4f}".format(mae))                                                    
print(1-(1-model.score(X_test,y_test))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))

r2_score(y_test, y_pred)


# In[357]:


X=compl_month4.drop(['Count_log'],axis=1)
y=compl_month4['Count_log']

X_train = X.head(7603)
X_test = X.tail(1901)
y_train = y.head(7603)
y_test = y.tail(1901)

rob =RobustScaler()

X_train = rob.fit_transform(X_train)
X_test = rob.transform(X_test)

lgbr_params = {'n_estimators': 1000,
          'max_depth': 6,
           #    'max_bin': 255,
              'num_leaves': 31,
          'learning_rate': 0.1}
     #     'num_iterations' : 100
      #    'loss': 'ls'}
#
# Create an instance of gradient boosting regressor
#
lgbr = LGBMRegressor(**lgbr_params)
#
# Fit the model
#
model = lgbr.fit(X_train, y_train)

y_pred=model.predict(X_test)

#
# Print Coefficient of determination R^2
#
print("Model Accuracy: %.3f" % lgbr.score(X_test, y_test))
#
# Create the mean squared error
#
mse = mean_squared_error(y_test, lgbr.predict(X_test))
mae = mean_absolute_error(y_test, lgbr.predict(X_test))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
print("The mean squared error (MSE) on test set: {:.4f}".format(mae))
print('Root Mean Squared Error is {:.4f} '.format(np.sqrt(mean_squared_error(y_test, y_pred, squared=False))))

print(1-(1-lgbr.score(X_test,y_test))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))
                                                    
feature_importance = lgbr.feature_importances_                                                   


# In[608]:


X=compl_month4.drop(['Count_log'],axis=1)
y=compl_month4['Count_log']

X_train = X.head(7603)
X_test = X.tail(1901)
y_train = y.head(7603)
y_test = y.tail(1901)

rob =RobustScaler()

X_train = rob.fit_transform(X_train)
X_test = rob.transform(X_test)
xgbr_params = {'n_estimators': 1000,
          'max_depth': 6,
              # 'max_bin': 255,
               #'num_leaves': 31,
          'learning_rate': 0.1}
     #     'num_iterations' : 100
      #  'loss': 'ls'}
#
# Create an instance of gradient boosting regressor
#
xgbr = XGBRegressor(**xgbr_params)
#
# Fit the model
#
model = xgbr.fit(X_train, y_train)

y_pred=model.predict(X_test)

#
# Print Coefficient of determination R^2
#
print("Model Accuracy: %.3f" % xgbr.score(X_test, y_test))
#
# Create the mean squared error
#
mse = mean_squared_error(y_test, xgbr.predict(X_test))
mae = mean_absolute_error(y_test, xgbr.predict(X_test))

print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
print("The mean squared error (MSE) on test set: {:.4f}".format(mae))
print('Root Mean Squared Error is {:.4f} '.format(np.sqrt(mean_squared_error(y_test, y_pred, squared=False))))
print(1-(1-xgbr.score(X_test,y_test))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))

print("Model Accuracy: %.3f" % xgbr.score(X_test, y_test))


feature_importance = xgbr.feature_importances_       

accuracy_score(y_test, xgbr.predict(X_test))


# In[606]:


from sklearn.metrics import accuracy_score


# In[321]:


X=compl_month4.drop(['Count_log'],axis=1)
y=compl_month4['Count_log']

X_train = X.head(7603)
X_test = X.tail(1901)
y_train = y.head(7603)
y_test = y.tail(1901)

# Quick sanity check with the shapes of Training and testing datasets
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)



# In[380]:



# create ANN model
model3 = Sequential()
 
# Defining the Input layer and FIRST hidden layer, both are same!
model3.add(Dense(128, input_dim=21, activation='relu'))
 
# Defining the Second layer of the model
# after the first layer we don't have to specify input_dim as keras configure it automatically
model3.add(Dense(74, activation='relu'))
 
# The output neuron is a single fully connected node 
# Since we will be predicting a single number
model3.add(Dense(1,  kernel_initializer='normal'))
 
# Compiling the model
model3.compile(loss='mean_squared_error', optimizer='adam')
 
# Fitting the ANN to the Training set
model3.fit(X_train, y_train, batch_size = 20, epochs = 50, verbose=1)

#y_pred=model.predict(X_test)
#ann.fit(x=X_train, y=y_train, epochs=100, batch_size=32,validation_data=(X_test,y_test),callbacks=EarlyStopping(monitor='val_loss',patience=4))


# In[322]:


model3 = MLPRegressor(hidden_layer_sizes=128, activation='tanh', solver='adam',
                         max_iter=100, learning_rate_init=0.001, batch_size=30)
model3.fit(X_train, y_train)


# In[323]:


Predictions1=model3.predict(X_test)


# In[630]:


print("Accuracy:", metrics.explained_variance_score(y_test, Predictions1))
print("MAE:",metrics.mean_absolute_error(y_test,Predictions1))
print ("MSE:",metrics.mean_squared_error(y_test,Predictions1))
print("RMSE:",np.sqrt(metrics.mean_squared_error(y_test,Predictions1)))

r2_score(Predictions1,y_test)


# In[ ]:




