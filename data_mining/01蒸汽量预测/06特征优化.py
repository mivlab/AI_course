#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_cell_magic('html', '', "<style>\n.dataframe td,.dataframe thead th { \n    note:'pandas表格属性';\n    white-space: auto;\n    text-align:left;\n    border:1px solid;\n    font-size:12px\n}\n.input_prompt{\n    note:'隐藏cell左边的提示如 In[12]以便于截图';\n#     display:none;\n}\ndiv.output_text {\n    note:'输出内容的高度';\n    max-height: 500px;\n}\ndiv.output_area img{\n    note:'输出图片的宽度';\n    max-width:100%\n}\ndiv.output_scroll{\n    note:'禁用输出的阴影';\n    box-shadow: none;\n}\n</style>\n<h5>!!以上是作者为了排版而修改的排版效果，请注意是否需要使用!!</h5>")


# In[2]:


# 修改pandas默认的现实设置
import numpy as np
import pandas as pd
pd.set_option('display.max_columns',10)  
pd.set_option('display.max_rows',20)  
#禁用科学计数法
np.set_printoptions(suppress=True,   precision=10,  threshold=2000,  linewidth=150)  
pd.set_option('display.float_format',lambda x : '%.2f' % x)


# ## 特征优化
# 
# ### 导入数据

# In[1]:


import pandas as pd
train_data_file = "./zhengqi_train.txt"
test_data_file = "./zhengqi_test.txt"
train_data = pd.read_csv(train_data_file, sep='\t', encoding='utf-8')
test_data = pd.read_csv(test_data_file, sep='\t', encoding='utf-8')


# ### 定义特征构造方法，构造特征

# In[2]:


epsilon = 1e-5
#组交叉特征，可以自行定义，如增加： x*x/y, log(x)/y 等等
func_dict = {
    'add': lambda x, y: x + y,
    'mins': lambda x, y: x - y,
    'div': lambda x, y: x / (y + epsilon),
    'multi': lambda x, y: x * y
}


# ### 定义特征构造的函数

# In[5]:


def auto_features_make(train_data, test_data, func_dict, col_list):
    train_data, test_data = train_data.copy(), test_data.copy()
    for col_i in col_list:
        for col_j in col_list:
            for func_name, func in func_dict.items():
                for data in [train_data, test_data]:
                    func_features = func(data[col_i], data[col_j])
                    col_func_features = '-'.join([col_i, func_name, col_j])
                    data[col_func_features] = func_features
    return train_data, test_data


# ### 对训练集和测试集数据进行特征构造

# In[6]:


train_data2, test_data2 = auto_features_make(
    train_data,test_data,func_dict,col_list=test_data.columns)


# In[7]:


from sklearn.decomposition import PCA   #主成分分析法

#PCA方法降维
pca = PCA(n_components=500)
train_data2_pca = pca.fit_transform(train_data2.iloc[:,0:-1])
test_data2_pca = pca.transform(test_data2)
train_data2_pca = pd.DataFrame(train_data2_pca)
test_data2_pca = pd.DataFrame(test_data2_pca)
train_data2_pca['target'] = train_data2['target']


# In[8]:


X_train2 = train_data2[test_data2.columns].values
y_train = train_data2['target']


# ### 使用lightgbm模型对新构造的特征进行模型训练和评估

# In[10]:


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import numpy as np

# 5折交叉验证
Folds = 5
kf = KFold(n_splits=Folds, shuffle=True, random_state=2019)
# 记录训练和预测MSE“
MSE_DICT = {'train_mse': [], 'test_mse': []}

# 线下训练预测
for i, (train_index, test_index) in enumerate(kf.split(X_train2)):
    # lgb树模型
    lgb_reg = lgb.LGBMRegressor(
        learning_rate=0.01,
        max_depth=-1,
        n_estimators=5000,
        boosting_type='gbdt',
        random_state=2019,
        objective='regression',
    )

    # 切分训练集和预测集
    X_train_KFold = X_train2[train_index]
    X_test_KFold =  X_train2[test_index]
    y_train_KFold = y_train[train_index]
    y_test_KFold = y_train[test_index]

    # 训练模型
    lgb_reg.fit(X=X_train_KFold,
                y=y_train_KFold,
                eval_set=[(X_train_KFold, y_train_KFold),
                          (X_test_KFold, y_test_KFold)],
                eval_names=['Train', 'Test'],
                early_stopping_rounds=100,
                eval_metric='MSE',
                verbose=50)

    # 训练集预测 测试集预测
    y_train_KFold_predict = lgb_reg.predict(
        X_train_KFold, num_iteration=lgb_reg.best_iteration_)
    y_test_KFold_predict = lgb_reg.predict(
        X_test_KFold, num_iteration=lgb_reg.best_iteration_)

    print('第{}折 训练和预测 训练MSE 预测MSE'.format(i))
    train_mse = mean_squared_error(y_train_KFold_predict,
                                   y_train_KFold)
    print('------\n', '训练MSE\n', train_mse, '\n------')
    test_mse = mean_squared_error(y_test_KFold_predict,
                                  y_test_KFold)
    print('------\n', '预测MSE\n', test_mse, '\n------\n')

    MSE_DICT['train_mse'].append(train_mse)
    MSE_DICT['test_mse'].append(test_mse)
print('------\n', '训练MSE\n', MSE_DICT['train_mse'], '\n',
      np.mean(MSE_DICT['train_mse']), '\n------')
print('------\n', '预测MSE\n', MSE_DICT['test_mse'], '\n',
      np.mean(MSE_DICT['test_mse']), '\n------')


# In[ ]:




