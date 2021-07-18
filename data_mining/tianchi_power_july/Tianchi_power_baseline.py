#!/usr/bin/env python
# coding: utf-8

# ### 加载库，载入数据

# In[51]:


import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# In[52]:


df = pd.read_csv("./Tianchi_power.csv")
df.head()


# 转换为标准日期格式
df['record_date'] = pd.to_datetime(df['record_date'])
df.loc[(df.user_id==1416) & (df.record_date>'20160731'),'power_consumption'] = 1216933.0
df.head()

# baseline 1：直接统计平均

base_df = df[['record_date','power_consumption']].groupby(by='record_date').agg('sum')
base_df = base_df.reset_index()
base_df.head()

import matplotlib.pyplot as plt
#trend = base_df[(base_df.record_date>='2015-07-01')&(base_df.record_date<'2015-10-01')]
trend = base_df[(base_df.record_date>='2015-06-01')&(base_df.record_date<'2015-09-01')]
plt.plot(trend['record_date'], trend['power_consumption'])
plt.title('2016 power consumption')
plt.show()
trend2 = base_df[(base_df.record_date>='2016-06-01')&(base_df.record_date<'2016-09-01')]
plt.plot(trend2['record_date'], trend2['power_consumption'])
plt.title('2017 power consumption')
plt.show()


# In[6]:


num1 = base_df[(base_df.record_date<='2015-06-30')&(base_df.record_date>='2015-06-24')]['power_consumption'].sum()
num1 = num1/7

num2 = base_df[(base_df.record_date<='2015-07-31')&(base_df.record_date>='2015-07-25')]['power_consumption'].sum()
num2 = num2/7

num3 = base_df[(base_df.record_date<='2015-08-31')&(base_df.record_date>='2015-08-25')]['power_consumption'].sum()
num3 = num3/7


# In[7]:


diff = (num1+num2-2*num3)/2
diff


# In[8]:


commit_tmp_df = base_df[(base_df.record_date>='2016-07-01') & (base_df.record_date<'2016-09-01')].copy()
commit_tmp_df['day'] = base_df['record_date'].apply(lambda x: x.day)
commit_tmp_df = commit_tmp_df[['power_consumption','day']].groupby(by='day').agg('sum')
commit_tmp_df = pd.DataFrame(commit_tmp_df.reset_index())

predict_df = pd.date_range('2016/9/1', periods=31, freq='D')
predict_df = pd.DataFrame(predict_df)
predict_df.columns = ['predict_date']
predict_df['predict_power_consumption'] = commit_tmp_df['power_consumption']/2
predict_df['predict_power_consumption'] = predict_df['predict_power_consumption'].astype('int')
predict_df['predict_power_consumption'] -= diff
predict_df = predict_df[(predict_df.predict_date<'2016-10-01')]
predict_df['predict_date'] = predict_df['predict_date'].astype(str).apply(lambda x: x.replace("-",""))
predict_df.to_csv('Tianchi_power_predict_table.csv',index=False)


# ### 统计方差和均值

# In[56]:


df2 = df[['user_id','power_consumption']].groupby(by='user_id').agg(['mean','median','var'])


# In[57]:


df2.reset_index(col_level=1)
df2.columns = df2.columns.get_level_values(1)
df2 = df2.reset_index()
df2.head()


# ## baseline 2：特征工程+lightGBM

# ### baseline 2.1 直接对每天的总量进行回归拟合

# In[58]:


df_test = base_df[(base_df.record_date>='2016-08-01')&(base_df.record_date<='2016-08-30')]
df_test['record_date'] = pd.DataFrame(df_test['record_date']+pd.Timedelta('31 days'))
df_test.head()


# In[59]:


base_df = pd.concat([base_df, df_test]).sort_values(['record_date'])


# In[60]:


base_df['dow'] = base_df['record_date'].apply(lambda x: x.dayofweek)
base_df['doy'] = base_df['record_date'].apply(lambda x: x.dayofyear)
base_df['day'] = base_df['record_date'].apply(lambda x: x.day)
base_df['month'] = base_df['record_date'].apply(lambda x: x.month)
base_df['year'] = base_df['record_date'].apply(lambda x: x.year)


# In[61]:


def map_season(month):
    month_dic = {1:1, 2:1, 3:2, 4:2, 5:3, 6:3, 7:3, 8:3, 9:3, 10:4, 11:4, 12:1}
    return month_dic[month]

base_df['season'] = base_df['month'].apply(lambda x: map_season(x))


# In[62]:


base_df.head()


# In[63]:


base_df_stats = new_df = base_df[['power_consumption','year','month']].groupby(by=['year', 'month']).agg(['mean', 'std'])
base_df_stats.head()


# In[64]:


base_df_stats.columns = base_df_stats.columns.droplevel(0)
base_df_stats = base_df_stats.reset_index()
base_df_stats.head()


# In[65]:


base_df_stats['1_m_mean'] = base_df_stats['mean'].shift(1)
base_df_stats['2_m_mean'] = base_df_stats['mean'].shift(2)
base_df_stats['1_m_std'] = base_df_stats['std'].shift(1)
base_df_stats['2_m_std'] = base_df_stats['std'].shift(2)
base_df_stats.head()


# In[66]:


data_df = pd.merge(base_df, base_df_stats[['year', 'month', '1_m_mean', '2_m_mean', '1_m_std', '2_m_std']], how='inner', on=['year', 'month'])
data_df = data_df[~pd.isnull(data_df['2_m_mean'])]


# In[67]:


data_df.tail()


# In[68]:


data_df.to_csv('./data_all_20170524.csv', index=False)


# In[47]:


train_data = data_df[data_df.record_date<'2016-09-01'][['dow','doy','day','month','year','season','1_m_mean','2_m_mean','1_m_std','2_m_std']]

test_data = data_df[data_df.record_date>='2016-09-01'][['dow','doy','day','month','year','season','1_m_mean','2_m_mean','1_m_std','2_m_std']]

train_target = data_df[data_df.record_date<'2016-09-01'][['power_consumption']]


# In[48]:


train_lgb = train_data.copy()
train_lgb[['dow','doy','day','month','year','season']] = train_lgb[['dow','doy','day','month','year','season']].astype(str)
test_lgb = test_data.copy()
test_lgb[['dow','doy','day','month','year','season']] = test_lgb[['dow','doy','day','month','year','season']].astype(str)


# In[49]:


X_lgb = train_lgb.values
y_lgb = train_target.values.reshape(train_target.values.shape[0],)


# In[50]:


X_lgb[0,:]


# In[25]:


import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

estimator = lgb.LGBMRegressor(colsample_bytree=0.8, subsample=0.9, subsample_freq=5)

param_grid = {
    'learning_rate': [0.01, 0.02, 0.05, 0.1],
    'n_estimators': [100, 200, 400, 800, 1000, 1200, 1500, 2000],
    'num_leaves':[128, 1024, 4096]
}

#fit_params = {'sample_weight':, 'early_stopping_rounds':5, 'categorical_feature':[0,1,2,3,4,5]}
#fit_params = {'early_stopping_rounds':5, 'categorical_feature':[0,1,2,3,4,5]}
fit_params = {'categorical_feature':[0,1,2,3,4,5]}

gbm = GridSearchCV(estimator, param_grid)#, fit_params=fit_params)

gbm.fit(X_lgb, y_lgb)

print("----------------------cv results--------------------------")
print(gbm.cv_results_)

print("----------------------------cv------------------------------")
print(gbm.cv)

print('Best parameters found by grid search are:', gbm.best_params_)


# ### 选择loss function和调整权重

# In[26]:


weights = 10000000.0/train_target.values.reshape(train_target.values.shape[0],)


# In[ ]:


import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

estimator = lgb.LGBMRegressor(colsample_bytree=0.8, subsample=0.9, subsample_freq=5, objective='regression_l1')

param_grid = {
    'learning_rate': [0.01, 0.02, 0.05, 0.1],
    'n_estimators': [800, 1000, 1500, 2000, 2500],
    'num_leaves':[128, 1024, 4096]
}

#fit_params = {'sample_weight':, 'early_stopping_rounds':5, 'categorical_feature':[0,1,2,3,4,5]}
#fit_params = {'early_stopping_rounds':5, 'categorical_feature':[0,1,2,3,4,5]}
fit_params = {'sample_weight': weights, 'categorical_feature':[0,1,2,3,4,5]}

gbm = GridSearchCV(estimator, param_grid)#, fit_params=fit_params)

gbm.fit(X_lgb, y_lgb)

print ("----------------------cv results--------------------------")
print (gbm.cv_results_)

print ("----------------------------cv------------------------------")
print (gbm.cv)

print('Best parameters found by grid search are:', gbm.best_params_)


# In[27]:


#随机敲定一组参数跑模型
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
# create dataset for lightgbm
lgb_train = lgb.Dataset(X_lgb, y_lgb)
# specify your configurations as a dict
params = {
    'num_leaves': 128,
    'learning_rate':0.01,
    'n_estimators':800,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'weight': weights,
    'application':'regression_l1'
}

print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=800,
                feature_name=['f' + str(i + 1) for i in range(X_lgb.shape[1])],
                categorical_feature=[0,1,2,3,4,5])


# In[28]:


#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
print('Plot feature importances...')
ax = lgb.plot_importance(gbm, max_num_features=50)
plt.show()


# In[38]:


commit_df = pd.date_range('2016/9/1', periods=30, freq='D')
commit_df = pd.DataFrame(commit_df)
commit_df.columns = ['predict_date']
y_predict = gbm.predict(test_lgb.values)
commit_df['predict_power_consumption'] = pd.DataFrame(y_predict).astype('int')
commit_df.head()


# In[39]:


commit_df['predict_date'] = commit_df['predict_date'].astype(str).apply(lambda x: x.replace("-",""))
commit_df.to_csv('Tianchi_power_predict_table_20170519_v2.csv',index=False)

