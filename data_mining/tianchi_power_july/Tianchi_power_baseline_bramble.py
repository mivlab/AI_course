#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import lightgbm as lgb


# In[13]:


# 直接导入之前保存过的处理好的文件
data_df = pd.read_csv("./dataset/data_all_20170524.csv")


# In[14]:


train_data = data_df[data_df.record_date<'2016-09-01'][['dow','doy','day','month','year','season','1_m_mean','2_m_mean','1_m_std','2_m_std']]

test_data = data_df[data_df.record_date>='2016-09-01'][['dow','doy','day','month','year','season','1_m_mean','2_m_mean','1_m_std','2_m_std']]

train_target = data_df[data_df.record_date<'2016-09-01'][['power_consumption']]

# 添加测试集的target
test_target = data_df[data_df.record_date>='2016-09-01'][['power_consumption']]
test_target = test_target.reset_index(drop=True)


# In[15]:


# 运行valid_sets=lgb_test2出错
# 提示：TypeError: Wrong type(ndarray) for label, should be list or numpy array
# 看了label不能输入dataframe, 要变为array

y_train = train_target.values.reshape(train_target.values.shape[0],)
# print y_train.shape
y_test = test_target.values.reshape(test_target.values.shape[0],)
# print y_test.shape


# ### lgb.Dataset可能导入有问题
# 导致最后预测的结果全是平均值，我重新按照lightGBM的文档把对应的参数修改一下。
# 
# 这里直接通过pandas的dataframe制作数据，下面这句是官网给的例子。
# 
#     train_data = lgb.Dataset(data, label=label, feature_name=['c1', 'c2', 'c3'], categorical_feature=['c3'])
# 

# In[16]:


# 制作lgb.dataset
weights = 10000000.0/train_target.values.reshape(train_target.values.shape[0],)

lgb_train2 = lgb.Dataset(train_data, label=y_train, weight=weights, feature_name=['dow', 'doy', 'day', 'year', 'month', 'season', '1_m_mean', '2_m_mean', '1_m_std', '2_m_std'], categorical_feature=['dow', 'doy', 'day', 'year', 'month', 'season'])
lgb_test2 = lgb.Dataset(test_data, label=y_test, feature_name=['dow', 'doy', 'day', 'year', 'month', 'season', '1_m_mean', '2_m_mean', '1_m_std', '2_m_std'], categorical_feature=['dow', 'doy', 'day', 'year', 'month', 'season'])


# In[17]:


# training!
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'auc'},
    'num_leaves': 128,
    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train2,
                num_boost_round=800,
#                early_stopping_rounds=5,
#                 valid_sets=lgb_test2,
                verbose_eval=False)


# In[18]:


commit_df = pd.date_range('2016/9/1', periods=30, freq='D')
commit_df = pd.DataFrame(commit_df)
commit_df.columns = ['predict_date']
y_predict = gbm.predict(test_data.values)
commit_df['predict_power_consumption'] = pd.DataFrame(y_predict).astype('int')
commit_df


# In[ ]:


commit_df['predict_date'] = commit_df['predict_date'].astype(str).apply(lambda x: x.replace("-",""))
commit_df.to_csv('Tianchi_power_predict_table_20170519_v4.csv',index=False)

