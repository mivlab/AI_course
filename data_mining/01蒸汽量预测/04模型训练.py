#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_cell_magic('html', '', "<style>\n.dataframe td,.dataframe thead th { \n    note:'pandas表格属性';\n    white-space: auto;\n    text-align:left;\n    border:1px solid;\n    font-size:12px\n}\n.input_prompt{\n    note:'隐藏cell左边的提示如 In[12]以便于截图';\n#     display:none;\n}\ndiv.output_text {\n    note:'输出内容的高度';\n    max-height: 500px;\n}\ndiv.output_area img{\n    note:'输出图片的宽度';\n    max-width:100%\n}\ndiv.output_scroll{\n    note:'禁用输出的阴影';\n    box-shadow: none;\n}\n</style>\n<h5>!!以上是作者为了排版而修改的排版效果，请注意是否需要使用!!</h5>")


# In[3]:


# 修改pandas默认的现实设置
import numpy as np
import pandas as pd
pd.set_option('display.max_columns',10)  
pd.set_option('display.max_rows',20)  
#禁用科学计数法
np.set_printoptions(suppress=True,   precision=10,  threshold=2000,  linewidth=150)  
pd.set_option('display.float_format',lambda x : '%.2f' % x)


# ## 商品销售模型示意
# $$sales =\beta_0+\beta_1*TV+\beta_2* radio+\beta_3*newspaper$$

# ## 导入工具和读取数据

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore") 
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


train_data_file = "./zhengqi_train.txt"
test_data_file =  "./zhengqi_test.txt"

train_data = pd.read_csv(train_data_file, sep='\t', encoding='utf-8')
test_data = pd.read_csv(test_data_file, sep='\t', encoding='utf-8')


# ## 特征工程

# In[6]:


##删除异常值
train_data = train_data[train_data['V9']>-7.5]
test_data = test_data[test_data['V9']>-7.5]

##归一化数据
from sklearn import preprocessing
features_columns = [col for col in train_data.columns if col not in ['target']]
min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler = min_max_scaler.fit(train_data[features_columns])
train_data_scaler = min_max_scaler.transform(train_data[features_columns])
test_data_scaler = min_max_scaler.transform(test_data[features_columns])

train_data_scaler = pd.DataFrame(train_data_scaler)
train_data_scaler.columns = features_columns
test_data_scaler = pd.DataFrame(test_data_scaler)
test_data_scaler.columns = features_columns
train_data_scaler['target'] = train_data['target']

##PCA降维 保持90%的信息
from sklearn.decomposition import PCA   #主成分分析法
pca = PCA(n_components=0.9)
new_train_pca_90 = pca.fit_transform(train_data_scaler.iloc[:,0:-1])
new_test_pca_90 = pca.transform(test_data_scaler)
new_train_pca_90 = pd.DataFrame(new_train_pca_90)
new_test_pca_90 = pd.DataFrame(new_test_pca_90)
new_train_pca_90['target'] = train_data_scaler['target']

pca = PCA(n_components=0.95)
new_train_pca_16 = pca.fit_transform(train_data_scaler.iloc[:,0:-1])
new_test_pca_16 = pca.transform(test_data_scaler)
new_train_pca_16 = pd.DataFrame(new_train_pca_16)
new_test_pca_16 = pd.DataFrame(new_test_pca_16)
new_train_pca_16['target'] = train_data_scaler['target']


# ## 切分数据集

# In[9]:


#切分数据集
from sklearn.model_selection import train_test_split  # 切分数据
new_train_pca_16 = new_train_pca_16.fillna(0)  #采用 pca 保留16维特征的数据
train = new_train_pca_16[new_test_pca_16.columns]
target = new_train_pca_16['target']

# 切分数据 训练数据80% 验证数据20%
train_data, test_data, train_target, test_target = train_test_split(
    train, target, test_size=0.2, random_state=0)


# ## 线性回归调用方法

# In[8]:


from sklearn.metrics import mean_squared_error  #评价指标

#从sklearn算法库中导入线性回归模型算法
from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(train_data, train_target)
test_pred = clf.predict(test_data)
score = mean_squared_error(test_target, clf.predict(test_data))
print("LinearRegression:   ", score)


# ## K近邻回归调用方法

# $$设有2个点P和Q,其中P={p_1,p_2,p_3,...p_n},Q={q_1,q_2,q_3,...q_n},n=1,2,3...\\
# 那么P与Q之间的距离表示为d，则\\
# d=\sqrt{{(p_1-q_1)^2+(p_2-q_2)^2+...+(P_n-q_n)^2}}$$

# In[10]:


#从sklearn算法库中导入K近邻回归模型算法
from sklearn.neighbors import KNeighborsRegressor

clf = KNeighborsRegressor(n_neighbors=3)  # 最近三个
clf.fit(train_data, train_target)
test_pred = clf.predict(test_data)
score = mean_squared_error(test_target, clf.predict(test_data))
print("KNeighborsRegressor:   ", score)


# ## 决策树回归的损失函数

# $$
# L(D)=\sum_{i=1}^k(y_i-\overline{y_1})^2+\sum_{i=k+1}^8(y_i-\overline{y_2})^2\\
# 其中：\overline{y_1}=\frac{1}{k}\sum_{i=1}^ky_i,\quad \overline{y_2}=\frac{1}{8-k}\sum_{i=k+1}^8y_i
# $$

# $$
# \overline{y_1}=\frac{1}{1}\sum_{i=1}^1y_i=2.73,\overline{y_2}=\frac{1}{7}\sum_{i=2}^8y_i=11.05\\
# L(D)=\sum_{i=1}^1(y_i-\overline{y_1})^2+\sum_{i=2}^8(y_i-\overline{y_2})^2=138.67
# $$

# ## 决策树回归调用方法

# In[11]:


#从sklearn算法库中导入决策回归树模型算法
from sklearn.tree import DecisionTreeRegressor

clf = DecisionTreeRegressor()
clf.fit(train_data, train_target)
test_pred = clf.predict(test_data)
score = mean_squared_error(test_target, clf.predict(test_data))
print("DecisionTreeRegressor:   ", score)


# ## 随机森林回归

# In[12]:


#从sklearn算法库中导入随机森林回归树模型算法
from sklearn.ensemble import RandomForestRegressor

clf = RandomForestRegressor(n_estimators=200) # 200棵树模型
clf.fit(train_data, train_target)
test_pred = clf.predict(test_data)
score = mean_squared_error(test_target, clf.predict(test_data))
print("RandomForestRegressor:   ", score)


# # 赛题模型训练

# ## 导入相关的库

# In[15]:


from sklearn.linear_model import LinearRegression  #线性回归
from sklearn.neighbors import KNeighborsRegressor  #K近邻回归
from sklearn.tree import DecisionTreeRegressor     #决策树回归
from sklearn.ensemble import RandomForestRegressor #随机森林回归
from sklearn.svm import SVR  #支持向量回归
import lightgbm as lgb #lightGbm模型

from sklearn.model_selection import train_test_split # 切分数据
from sklearn.metrics import mean_squared_error #评价指标

from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


# ## 切分数据

# In[16]:


#采用 pca 保留16维特征的数据
new_train_pca_16 = new_train_pca_16.fillna(0)
train = new_train_pca_16[new_test_pca_16.columns]
target = new_train_pca_16['target']

# 切分数据 训练数据80% 验证数据20%
train_data, test_data, train_target, test_target = train_test_split(
    train, target, test_size=0.2, random_state=0)


# ## 多元线性回归

# In[30]:


clf = LinearRegression()
clf.fit(train_data, train_target)
score = mean_squared_error(test_target, clf.predict(test_data))
print("LinearRegression:   ", score)


# ##  K近邻回归

# In[32]:


clf = KNeighborsRegressor(n_neighbors=8) # 最近三个
clf.fit(train_data, train_target)
score = mean_squared_error(test_target, clf.predict(test_data))
print("KNeighborsRegressor:   ", score)


# ## 随机森林回归

# In[33]:


clf = RandomForestRegressor(n_estimators=200) # 200棵树模型
clf.fit(train_data, train_target)
score = mean_squared_error(test_target, clf.predict(test_data))
print("RandomForestRegressor:   ", score)


# ## LGB模型回归

# In[35]:


# lgb回归模型
clf = lgb.LGBMRegressor(
    learning_rate=0.01,
    max_depth=-1,
    n_estimators=5000,
    boosting_type='gbdt',
    random_state=2019,
    objective='regression',
)

# 训练模型
clf.fit(X=train_data, y=train_target, eval_metric='MSE', verbose=50)

score = mean_squared_error(test_target, clf.predict(test_data))
print("lightGbm:   ", score)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




