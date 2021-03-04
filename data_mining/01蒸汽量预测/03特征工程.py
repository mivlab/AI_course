#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_cell_magic('html', '', "<style>\n.dataframe td,.dataframe thead th { \n    note:'pandas表格属性';\n    white-space: auto;\n    text-align:left;\n    border:1px solid;\n    font-size:12px\n}\n.input_prompt{\n    note:'隐藏cell左边的提示如 In[12]以便于截图';\n#     display:none;\n}\ndiv.output_text {\n    note:'输出内容的高度';\n    max-height: 300px;\n}\ndiv.output_area img{\n    note:'输出图片的宽度';\n    max-width:100%\n}\ndiv.output_scroll{\n    note:'禁用输出的阴影';\n    box-shadow: none;\n}\n</style>\n<h5>!!以上是作者为了排版而修改的排版效果，请注意是否需要使用!!</h5>")


# In[2]:


# 修改pandas默认的现实设置
import numpy as np
import pandas as pd
pd.set_option('display.max_columns',10)  
pd.set_option('display.max_rows',20)  
#禁用科学计数法
np.set_printoptions(suppress=True,   precision=10,  threshold=2000,  linewidth=150)  
pd.set_option('display.float_format',lambda x : '%.4f' % x)


# ## 数据标准化

# In[3]:


from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
iris = load_iris()
#标准化，返回值为标准化后的数据
StandardScaler().fit_transform(iris.data)


# ## 区间缩放法

# In[4]:


from sklearn.preprocessing import MinMaxScaler
#区间缩放，返回值为缩放到[0, 1]区间的数据
MinMaxScaler().fit_transform(iris.data)


# ## 归一化

# In[5]:


from sklearn.preprocessing import Normalizer
#归一化，返回值为归一化后的数据
Normalizer().fit_transform(iris.data)


# ## 二值化

# In[6]:


from sklearn.preprocessing import Binarizer
#二值化，阈值设置为3，返回值为二值化后的数据
Binarizer(threshold=3).fit_transform(iris.data)


# ## 哑编码

# In[7]:


from sklearn.preprocessing import OneHotEncoder
#哑编码，对IRIS数据集的目标值，返回值为哑编码后的数据
OneHotEncoder(categories='auto').fit_transform(iris.target.reshape((-1,1)))


# ## 缺失值处理

# In[8]:


from numpy import vstack, array, nan
from sklearn.impute import SimpleImputer 

#缺失值计算，返回值为计算缺失值后的数据 
#参数missing_value为缺失值的表示形式，默认为NaN 
#参数strategy为缺失值填充方式，默认为mean（均值） 
SimpleImputer().fit_transform(vstack((array([nan, nan, nan, nan]), 
                                      iris.data)))


# ## 多项式变换公式
# 
# 
# \begin{align}
# &(x'_1,x'_2,x'_3,x'_4,x'_5,x'_6,x'_7,x'_8,x'_9,x'_{10},x'_{11},x'_{12},x'_{13},x'_{14},x'_{15})\\
# & = (1,x_1,x_2,x_3,x_4,x^2_1,x_1*x_2,x_1*x_3,x_1*x_4,x^2_2,x_2*x_3,x_2*x_4,x_3^2,x_3*x_4,x_4^2)
# \end{align}
# 

# ## 多项式变换

# In[9]:


from sklearn.preprocessing import PolynomialFeatures
#多项式转换
#参数degree为度，默认值为2 
PolynomialFeatures().fit_transform(iris.data)


# ## 对数函数的数据变换

# In[10]:


from numpy import log1p
from sklearn.preprocessing import FunctionTransformer
#自定义转换函数为对数函数的数据变换
#第一个参数是单变元函数
FunctionTransformer(log1p, validate=False).fit_transform(iris.data)


# ## 方差选择法

# In[11]:


from sklearn.feature_selection import VarianceThreshold
from sklearn.datasets import load_iris
iris = load_iris()
#方差选择法，返回值为特征选择后的数据
#参数threshold为方差的阈值
VarianceThreshold(threshold=3).fit_transform(iris.data)


# ## 相关系数法

# In[12]:


import numpy as np
from sklearn.datasets import load_iris
iris = load_iris()
from array import array
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr
#选择K个最好的特征，返回选择特征后的数据
#第一个参数为计算评估特征是否好的函数，该函数输入特征矩阵和目标向量，输出二元组（评分，P值）的数组，数组第i项为第i个特征的评分和P值。在此定义为计算相关系数
#参数k为选择的特征个数
SelectKBest(
    lambda X, Y: np.array(list(map(lambda x: pearsonr(x, Y), X.T))).T[0],
    k=2).fit_transform(iris.data, iris.target)


# ## 卡方检验公式
# $$x^2=\sum{\frac{(A-E)^2}{E}}$$

# ## 卡方检验

# In[13]:


from sklearn.datasets import load_iris
iris = load_iris()
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#选择K个最好的特征，返回选择特征后的数据
SelectKBest(chi2, k=2).fit_transform(iris.data, iris.target)


# ## 互信息公式
# $$I(X;Y)=\sum_{x\in y}\sum_{y\in x}p(x,y)log\frac{p(x,y)}{p(x)p(y)}$$

# ## 互信息法

# In[14]:


import numpy as np
from sklearn.feature_selection import SelectKBest 
from minepy import MINE 
 
#由于MINE的设计不是函数式的，定义mic方法将其为函数式的，返回一个二元组，二元组的第2项设置成固定的P值0.5 
def mic(x, y): 
    m = MINE() 
    m.compute_score(x, y) 
    return (m.mic(), 0.5) 

#选择K个最好的特征，返回特征选择后的数据 
SelectKBest(
    lambda X, Y: np.array(list(map(lambda x: mic(x, Y), X.T))).T[0],
    k=2).fit_transform(iris.data, iris.target)


# ## 递归特征消除法RFE

# In[15]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

#递归特征消除法，返回特征选择后的数据
#参数estimator为基模型
#参数n_features_to_select为选择的特征个数
RFE(estimator=LogisticRegression(multi_class='auto',
                                 solver='lbfgs',
                                 max_iter=500),
    n_features_to_select=2).fit_transform(iris.data, iris.target)


# ## 基于惩罚项的特征选择法

# In[16]:


from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

#带L1惩罚项的逻辑回归作为基模型的特征选择
SelectFromModel(
    LogisticRegression(penalty='l2', C=0.1, solver='lbfgs',
                       multi_class='auto')).fit_transform(
                           iris.data, iris.target)


# ## 基于树模型的特征选择法

# In[17]:


from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier
 
# GBDT作为基模型的特征选择
SelectFromModel(GradientBoostingClassifier()).fit_transform(iris.data,
                                                            iris.target)


# ## 主成分分析法

# In[18]:


from sklearn.decomposition import PCA
 
#主成分分析法，返回降维后的数据
#参数n_components为主成分数目
PCA(n_components=2).fit_transform(iris.data)


# ## 线性判别分析法

# In[19]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
 
#线性判别分析法，返回降维后的数据
#参数n_components为降维后的维数
LDA(n_components=2).fit_transform(iris.data, iris.target)


# ## 导入工具和读取数据

# In[20]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

import warnings
warnings.filterwarnings("ignore")
 
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[21]:


train_data_file = "./zhengqi_train.txt"
test_data_file =  "./zhengqi_test.txt"

train_data = pd.read_csv(train_data_file, sep='\t', encoding='utf-8')
test_data = pd.read_csv(test_data_file, sep='\t', encoding='utf-8')


# ## 绘制各个特征的箱线图

# In[22]:


plt.figure(figsize=(18, 10))
plt.boxplot(x=train_data.values,labels=train_data.columns)
plt.hlines([-7.5, 7.5], 0, 40, colors='r')
plt.show()


# In[23]:


pd.set_option('display.max_columns',6)


# ## 删除训练集异常值

# In[24]:


train_data = train_data[train_data['V9']>-7.5]
test_data = test_data[test_data['V9']>-7.5]
#display(train_data.describe())
#display(test_data.describe())


# ## 归一化数据集

# In[25]:


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

#display(train_data_scaler.describe())
#display(test_data_scaler.describe())


# ## 查看数据分布情况

# In[26]:


drop_col = 6
drop_row = 1
plt.figure(figsize=(5 * drop_col, 5 * drop_row))

for i, col in enumerate(["V5", "V9", "V11", "V17", "V22", "V28"]):
    ax = plt.subplot(drop_row, drop_col, i + 1)
    ax = sns.kdeplot(train_data_scaler[col], color="Red", shade=True)
    ax = sns.kdeplot(test_data_scaler[col], color="Blue", shade=True)
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    ax = ax.legend(["train", "test"])
plt.show()


# ## 特征相关性

# In[27]:


plt.figure(figsize=(20, 16))  
column = train_data_scaler.columns.tolist()  
mcorr = train_data_scaler[column].corr(method="spearman")  
mask = np.zeros_like(mcorr, dtype=np.bool)  
mask[np.triu_indices_from(mask)] = True  
cmap = sns.diverging_palette(220, 10, as_cmap=True)  
g = sns.heatmap(mcorr, mask=mask, cmap=cmap, square=True, 
                annot=True, fmt='0.2f')  
plt.show()


# ## 特征相关性初筛

# In[28]:


mcorr=mcorr.abs()
numerical_corr=mcorr[mcorr['target']>0.1]['target']
print(numerical_corr.sort_values(ascending=False))


# ## 多重共线性分析

# In[29]:


from statsmodels.stats.outliers_influence import variance_inflation_factor #多重共线性方差膨胀因子

#多重共线性
new_numerical=['V0', 'V2', 'V3', 'V4', 'V5', 'V6', 'V10','V11',
               'V13', 'V15', 'V16', 'V18', 'V19', 'V20', 
               'V22','V24','V30', 'V31', 'V37']
X=np.matrix(train_data_scaler[new_numerical])
VIF_list=[variance_inflation_factor(X, i) for i in range(X.shape[1])]
VIF_list


# In[30]:


pd.set_option('display.max_columns',6)


# ## PCA方法进行降维

# In[31]:


from sklearn.decomposition import PCA   #主成分分析法

#保持90%的信息
pca = PCA(n_components=0.9)
new_train_pca_90 = pca.fit_transform(train_data_scaler.iloc[:,0:-1])
new_test_pca_90 = pca.transform(test_data_scaler)
new_train_pca_90 = pd.DataFrame(new_train_pca_90)
new_test_pca_90 = pd.DataFrame(new_test_pca_90)
new_train_pca_90['target'] = train_data_scaler['target']
new_train_pca_90.describe()


# In[32]:


train_data_scaler.describe()


# ## PCA保留16个主成分

# In[33]:


pca = PCA(n_components=0.95)
new_train_pca_16 = pca.fit_transform(train_data_scaler.iloc[:,0:-1])
new_test_pca_16 = pca.transform(test_data_scaler)
new_train_pca_16 = pd.DataFrame(new_train_pca_16)
new_test_pca_16 = pd.DataFrame(new_test_pca_16)
new_train_pca_16['target'] = train_data_scaler['target']
new_train_pca_16.describe()


# In[34]:


pca = PCA(n_components=16)
new_train_pca_16 = pca.fit_transform(train_data_scaler.iloc[:,0:-1])
new_test_pca_16 = pca.transform(test_data_scaler)
new_train_pca_16 = pd.DataFrame(new_train_pca_16)
new_test_pca_16 = pd.DataFrame(new_test_pca_16)
new_train_pca_16['target'] = train_data_scaler['target']
new_train_pca_16.describe()


# In[ ]:




