import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
from scipy.stats import norm
from scipy import stats
pd.set_option('display.width',1000)
train=pd.read_csv('Data2/train.csv')
np.log1p(train.SalePrice)
test=pd.read_csv('Data2/test.csv')
data_train=pd.concat([train, test],keys=['train','test'])
output=data_train['SalePrice']
data_train=data_train.drop(['SalePrice'],axis=1)



#data_train.info()
#print(data_train.describe())
#print(data_train.columns)
#sns.distplot(data_train['SalePrice'])
#plt.show()
#print('skewness: {0}, kurtosis: {1}'.format(data_train['SalePrice'].skew(), data_train['SalePrice'].kurt()))
var0,var1,var2='GrLivArea','TotalBsmtSF','OverallQual'
# fig,axes=plt.subplots(nrows=1,ncols=3,figsize=(10,5))
# data_train.plot.scatter(x=var0,y=output,ylim=(0,800000),ax=axes[0])
# data_train.plot.scatter(x=var1,y=output,ylim=(0,800000),ax=axes[1])
# data_train.plot.scatter(x=var2,y=output,ylim=(0,800000),ax=axes[2])
# plt.show()


#overallQual
# fig, ax = plt.subplots(figsize=(8,6))
# sns.boxplot(x=var2,y=output,data=data_train)
# ax.set_ylim(0,800000)
# plt.show()

##YearBuild
# var3 = 'YearBuilt'
# fig, ax = plt.subplots(figsize=(16,8))
# sns.boxplot(x=var3,y=output,data=data_train)
# ax.set_ylim(0,800000)
# plt.xticks(rotation=90)
# plt.show()

#pairplot
# var_set = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
# sns.set(font_scale=1)  # 设置横纵坐标轴的字体大小
# sns.pairplot(data_train[var_set],markers='+')  # 7*7图矩阵
# # 可在kind和diag_kind参数下设置不同的显示类型，此处分别为散点图和直方图，还可以设置每个图内的不同类型的显示
# plt.show()

#取数值型变量
# df_tr=pd.read_csv('Data2/train.csv').drop('Id',axis=1)
# df_X=df_tr.drop('SalePrice',axis=1)
# df_Y=df_tr['SalePrice']
# quantity=[attr for attr in df_X.columns if df_X.dtypes[attr]!='object']
# quality = [attr for attr in df_X.columns if df_X.dtypes[attr] == 'object']
# #print(df_X )
# #print(quantity)
# #print(quality )
# melt_X = pd.melt(df_X, value_vars=quantity)
# #print(melt_X.values)
# #print(melt_X.head())
# g = sns.FacetGrid(melt_X, col="variable",  col_wrap=5, sharex=False, sharey=False)
# g = g.map(sns.distplot, "value")  # 以melt_X['value']作为数据
# plt.show()

#各个特征关系分析
#corrmat = data_train.corr()
# f, ax = plt.subplots(figsize=(12, 9))
# sns.heatmap(corrmat, vmax=.8, square=True, ax=ax)  # square参数保证corrmat为非方阵时，图形整体输出仍为正方形
# ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
# #ax.set_yticklabels(ax.get_yticklabels(),rotation=90)
# plt.show()
#选择与output相关系数最高的10个特征
# k = 10
# top10_attr = corrmat.nlargest(k, output).index
# top10_mat = corrmat.loc[top10_attr, top10_attr]
# fig,ax = plt.subplots(figsize=(8,6))
# sns.set(font_scale=1.25)
# sns.heatmap(top10_mat, annot=True, annot_kws={'size':12}, square=True)
# # # 设置annot使其在小格内显示数字，annot_kws调整数字格式
# ax.set_xticklabels(corrmat.index, rotation=90)
# ax.set_yticklabels(corrmat.index,fontsize=6)
# plt.show()
#观察缺失值情况
na_count = data_train.isnull().sum().sort_values(ascending=False)
na_rate = na_count / len(data_train)
na_data = pd.concat([na_count,na_rate],axis=1,keys=['count','ratio'])
# print(na_data.head(20))
# print(na_count.values)
# print(na_count.index)
#处理缺失值
data_train=data_train.drop(na_count[na_count.values>2].index,axis=1)
data_train=data_train.drop(data_train.loc[data_train['Electrical'].isnull()].index)

fill_mode=["KitchenQual","Functional","SaleType","Electrical","Exterior1st","Exterior2nd"]
fill_0=['BsmtFullBath','BsmtHalfBath',"GarageCars","GarageArea",'TotalBsmtSF','BsmtUnfSF','BsmtFinSF1','BsmtFinSF2']
for col in fill_mode:
    data_train[col].fillna(data_train[col].mode()[0],inplace=True)
for col in fill_0:
    data_train[col].fillna(0,inplace=True)
data_train.drop(['Utilities'],axis=1,inplace=True)
#na_count = data_train.isnull().sum().sort_values(ascending=False)

#print(data_train.shape)

#数值型变量与类型值变量
#output='SalePrice'
#data_train.drop(['Id'],axis=1)
valueInnum=[attr for attr in data_train.columns if data_train[attr].dtype!='object']
valueInclass=[attr for attr in data_train.columns if data_train[attr].dtype=='object']
#print(valueInnum)

#对离散值一元方差分析
# def anova(frame, qualitative):
#     anv = pd.DataFrame()
#     anv['feature'] = qualitative
#     pvals = []
#     for c in qualitative:
#         samples = []
#         for cls in frame[c].unique():
#             s = frame[frame[c] == cls]['SalePrice'].values
#             samples.append(s)  # 某特征下不同取值对应的房价组合形成二维列表
#         pval = stats.f_oneway(*samples)[1]  # 一元方差分析得到 F，P，要的是 P，P越小，对方差的影响越大。
#         pvals.append(pval)
#     anv['pval'] = pvals
#     return anv.sort_values('pval')
#
# a = anova(data_train,valueInclass)
# a['disparity'] = np.log(1./a['pval'].values)  # 悬殊度
# fig, ax = plt.subplots(figsize=(16,8))
# sns.barplot(data=a, x='feature', y='disparity')
# x=plt.xticks(rotation=90)
# plt.show()

#将离散变量转化为数值型变量
def encode(frame, feature):
    '''
    对所有类型变量，依照各个类型变量的不同取值对应的样本集内房价的均值，按照房价均值高低
    对此变量的当前取值确定其相对数值1,2,3,4等等，相当于对类型变量赋值使其成为连续变量。
    此方法采用了与One-Hot编码不同的方法来处理离散数据，值得学习
    注意：此函数会直接在原frame的DataFrame内创建新的一列来存放feature编码后的值。
    '''
    ordering = pd.DataFrame()
    ordering['val'] = frame[feature].unique()
    ordering.index = ordering.val
   # ordering['price_mean'] = frame[[feature, 'SalePrice']].groupby(feature).mean()['SalePrice']
    # 上述 groupby()操作可以将某一feature下同一取值的数据整个到一起，结合mean()可以直接得到该特征不同取值的房价均值
    #ordering = ordering.sort_values('price_mean')
    ordering['order'] = range(1, ordering.shape[0]+1)
    ordering = ordering['order'].to_dict()
    for attr_v, score in ordering.items():
        # e.g. qualitative[2]: {'Grvl': 1, 'MISSING': 3, 'Pave': 2}
        frame.loc[frame[feature] == attr_v, feature+'_E'] = score

valueInclass_encoded = []
# 由于qualitative集合中包含了非数值型变量和伪数值型变量（多为评分、等级等，其取值为1,2,3,4等等）两类
# 因此只需要对非数值型变量进行encode()处理。
# 如果采用One-Hot编码，则整个qualitative的特征都要进行pd,get_dummies()处理
for q in valueInclass:
    encode(data_train, q)
    valueInclass_encoded.append(q+'_E')
data_train.drop(valueInclass, axis=1, inplace=True)  # 离散变量已经有了编码后的新变量，因此删去原变量

# df_tr.shape = (1460, 80)
#print(valueInclass_encoded, '\n{} qualitative attributes have been encoded.'.format(len(valueInclass_encoded)))

#求spearman相关系数

# def spearman(frame, features):
#     '''
#     采用“斯皮尔曼等级相关”来计算变量与房价的相关性(可查阅百科)
#     此相关系数简单来说，可以对上述encoder()处理后的等级变量及其它与房价的相关性进行更好的评价（特别是对于非线性关系）
#     '''
#     spr = pd.DataFrame()
#     spr['feature'] = features
#     spr['corr'] = [frame[f].corr(frame['SalePrice'], 'spearman') for f in features]
#     spr = spr.sort_values('corr')
#     plt.figure(figsize=(6, 0.25*len(features)))
#     sns.barplot(data=spr, y='feature', x='corr', orient='h')
#     plt.show()
# features = valueInnum + valueInclass_encoded
# spearman(data_train, features)

#求变量间相关系数

#fig,axes=plt.subplots(nrows=1,ncols=3)
# plt.figure(1,figsize=(12,9))  # 连续型变量相关图
# corr = data_train[valueInnum+['SalePrice']].corr()
# print(corr)
# sns.heatmap(corr,yticklabels=False, cmap="YlGnBu",ax=axes[0])
# x=plt.xticks(rotation=90)
# plt.show()
#
# plt.figure(2,figsize=(12,9))  # 等级型变量相关图（离散型和伪数值型变量均已被概括为等级型变量）
# corr = data_train[valueInclass_encoded+['SalePrice']].corr('spearman')
# sns.heatmap(corr,yticklabels=False, cmap="YlGnBu",ax=axes[1])
# x=plt.xticks(rotation=90)
# plt.show()
#
# plt.figure(3,figsize=(12,9)) # 连续型变量-等级型变量相关图
# corr = pd.DataFrame(np.zeros([len(valueInnum)+1, len(valueInclass_encoded)+1]),
#                     index=valueInnum+['SalePrice'], columns=valueInclass_encoded+['SalePrice'])
# for q1 in valueInnum+['SalePrice']:
#     for q2 in valueInclass_encoded+['SalePrice']:
#         corr.loc[q1, q2] = data_train[q1].corr(data_train[q2], 'spearman')
# sns.heatmap(corr,yticklabels=False, cmap="YlGnBu",ax=axes[2])
# x=plt.xticks(rotation=90)
# plt.show()

# 给房价分段，并由此查看各段房价内那些特征的取值会出现悬殊
# poor = data_train[data_train['SalePrice'] < 200000][valueInnum].mean()
# pricey = data_train[data_train['SalePrice'] >= 200000][valueInnum].mean()
# diff = pd.DataFrame()
# diff['attr'] = valueInnum
# diff['difference'] = ((pricey-poor)/poor).values
# plt.figure(figsize=(10,4))
# sns.barplot(data=diff, x='attr', y='difference')
# plt.xticks(rotation=90)
# plt.show()
#删除离群值
data_train = data_train.drop(data_train[data_train['Id'] == 1299].index)
data_train = data_train.drop(data_train[data_train['Id'] == 524].index)




#print(data_train.columns)
#处理连续变量为离散值
data_train['HasBasement'] = data_train['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
data_train['HasGarage'] = data_train['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
data_train['Has2ndFloor'] = data_train['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
#data_train['HasMasVnr'] = data_train['MasVnrArea'].apply(lambda x: 1 if x > 0 else 0)
data_train['HasWoodDeck'] = data_train['WoodDeckSF'].apply(lambda x: 1 if x > 0 else 0)
data_train['HasPorch'] = data_train['OpenPorchSF'].apply(lambda x: 1 if x > 0 else 0)
data_train['HasPool'] = data_train['PoolArea'].apply(lambda x: 1  if x > 0 else 0)
data_train['IsNew'] = data_train['YearBuilt'].apply(lambda x: 1 if x > 2000 else 0)
boolean = ['HasBasement', 'HasGarage', 'Has2ndFloor', 'HasMasVnr',
           'HasWoodDeck', 'HasPorch', 'HasPool', 'IsNew']
old=['TotalBsmtSF','GarageArea','2ndFlrSF','WoodDeckSF','OpenPorchSF','PoolArea','YearBuilt']
data_train.drop(old,axis=1, inplace=True)
data_train.drop('Id',axis=1, inplace=True)

# #取对数缓解数据正偏性
# def log_transform(feature):
#     # np.log1p(x) = log(1+x)，这样就可以对0值求对数（针对 `TotalBsmtSF` 这样含有0的特征）
#     data_train[feature] = np.log1p(data_train[feature].values)
from scipy.special import boxcox1p
from scipy.stats import skew

valueInnum=[attr for attr in data_train.columns if data_train[attr].dtype!='object']
valueInclass=[attr for attr in data_train.columns if data_train[attr].dtype=='object']
skew_features=data_train[valueInnum].apply(lambda x : skew(x.dropna()))
skw_feature=skew_features[skew_features>0.75].index
data_train[skw_feature] = boxcox1p(data_train[skw_feature],0.15)
output=boxcox1p(output,0.15)

#Featrue=[featrue for featrue in data_train.columns]


# def quadratic(feature):
#     data_train[feature] = data_train[feature[:-1]]**2
#
# qdr = ['OverallQual2', 'YearBuilt2', 'YearRemodAdd2', 'TotalBsmtSF2',
#         '2ndFlrSF2', 'Neighborhood_E2', 'RoofMatl_E2', 'GrLivArea2']
# for feature in qdr:
#     quadratic(feature)
# print(data_train.shape)
# data_train = pd.get_dummies(data_train)
# print(data_train.shape)
#category_feats = features.dtypes[features.dtypes == "object"].index
# print(data_train.columns)
#
#对离散值进行oneHot编码
for col in valueInclass_encoded:
     for_dummy = data_train.pop(col)
     extra_data = pd.get_dummies(for_dummy,prefix=col)
     #print(col,":",extra_data.shape)
     data_train = pd.concat([data_train, extra_data],axis=1)


#
#归一化
from sklearn.preprocessing import RobustScaler
N = RobustScaler()
scale_train = N.fit_transform(data_train)
# from sklearn.preprocessing import StandardScaler
# output=StandardScaler.fit_transform(output)


#from sklearn.utils import shuffle
train_set=scale_train[:train.shape[0]]
test_set=scale_train[train.shape[0]:]
output=output[:train.shape[0]]
#scale_output = N.fit_transform(output[:train.shape[0]])
#train_set,scale_output= shuffle(train_set,scale_output,random_state=5)
#print(output)
# sns.distplot(output)
# plt.show()
# print(train_set)
# print(test_set)
# print(output)

Y=output.values
from sklearn import linear_model
lasso=linear_model.LassoLarsCV(max_iter=10000)
lasso.fit(train_set,Y)
Ypred=lasso.predict(train_set)
#error(Y,Ypred)

def error(actual, predicted):
    actual = np.log(actual)
    predicted = np.log(predicted)
    return np.sqrt(np.sum(np.square(actual-predicted))/len(actual))

print(error(Y,Ypred))
n_test_samples = len(train_set)
X = range(n_test_samples)
plt.plot(X, Ypred, 'r--*', label = 'Predict Price')
plt.plot(X, output, 'g:', label='True Price')
legend = plt.legend()
plt.title("Ridge Regression (Boston)")
plt.ylabel("Price (1000 U.S.D)")
plt.savefig("Ridge Regression (Boston).png", format='png')
plt.show()
