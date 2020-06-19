# coding=utf-8
import numpy as np
import pandas as pd
from sklearn.preprocessing import Binarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import  linear_model
from sklearn import svm
from sklearn.model_selection import cross_val_score
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns

from pylab import mpl                            #显示中文字体
mpl.rcParams['font.sans-serif'] = ['FangSong']   #指定默认字体：仿宋
mpl.rcParams['axes.unicode_minus'] = False       #解决保存图像是负号'-'显示为方块的问题
from matplotlib.font_manager import FontProperties
myfont=FontProperties(fname=r'C:\Windows\Fonts\simhei.ttf',size=14)
sns.set(font=myfont.get_name())

##导入文件，观察总体情况
data_train= pd.read_csv('DateP/IVF-XN.csv',encoding='gbk')
pd.set_option('display.width', 1000)
#data_train .info()
#print(data_train .describe())
data_train.columns
##观察缺失值情况
output=data_train['临床妊娠']
#a=output.isnull().sum()
data_train=data_train.drop(data_train.loc[output.isnull()].index)#去掉没有结果的样本
na_count = data_train.isnull().sum().sort_values(ascending=False)
na_rate = na_count / len(data_train)
na_data = pd.concat([na_count,na_rate],axis=1,keys=['count','ratio'])
#print(na_data.head(70))
# #print(na_count.values)
# #print(na_count.index)
data_train=data_train.drop(na_count[na_count.values>=174].index,axis=1)#将缺失值比率大于10%的特征直接删除
data_train=data_train.drop(['就诊地点','患者编号'],axis=1)
data_train=data_train.drop(['et14天p','et14天e2'],axis=1)
fill_mode=['首诊医生','et14天hcg','hcg次日hcg','用药记录者','实验室et医生','女方诊断','et胚胎评分','外管出血','内芯出血']
fill_0=['人工流产','婚育史产','自然流产']
for col in fill_mode:
    data_train[col].fillna(data_train[col].mode()[0],inplace=True)
for col in fill_0:
    data_train[col].fillna(0,inplace=True)
# data_train=data_train.drop(['初诊日期','取卵日期'],axis=1)
# print(output)
# na_count2= data_train.isnull().sum().sort_values(ascending=False)
# na_rate2 = na_count2 / len(data_train)
# na_data2 = pd.concat([na_count2,na_rate2],axis=1,keys=['count','ratio'])
# print(na_data2.head(20))
#将下面几个特征从字符型改为数值型
for i,item in enumerate(data_train['et14天hcg']):
    if item=='<2'or item=='>2.00'or item=='<0.50'or item=='<2.00'or item=='<2.0':
        data_train['et14天hcg'].iloc[i]='2'
    elif item=='>1000':
        data_train['et14天hcg'].iloc[i]='1000'
data_train['et14天hcg']=list(map(float,data_train['et14天hcg']))

for i,item in enumerate(data_train['hcg次日hcg']):
    if item=='<2.00':
        data_train['hcg次日hcg'].iloc[i]='2'
    elif item=='<0.50':
        data_train['hcg次日hcg'].iloc[i]='0.5'
data_train['hcg次日hcg']=list(map(float,data_train['hcg次日hcg']))

#删除部分特征
data_train=data_train.drop(['初诊日期','体检日期',  '降调日期', 'gn日期','取卵日期','et日期','冷冻胚胎费用到期日期','et胚胎评分','生化妊娠',
                            '女1体重','女2体重','男1体重','男2体重','分娩日期','男1身长','男2身长','女1身长','实验室et数','降调14天fsh','体重指数',
                            'gn启动日afc平均值','gn启动量','降调天数','mii','临床取卵数','实验室拾卵数','卵裂数','获胚数','冷冻胚胎数','冷冻管数','icsi卵裂数',
                            '女2身长','孕三个月胚儿数','胎数','出生男','出生女','达菲林首日量','赠与卵数','科研胚胎数','冷冻卵子数'],axis=1)


#创建新的特征代替女方诊断
data_train['女方诊断_原发不孕']=data_train['女方诊断'].apply(lambda x: 1 if '原发不孕'in x else 0)
data_train['女方诊断_继发不孕']=data_train['女方诊断'].apply(lambda x: 1 if '继发不孕'in x else 0)
data_train['女方诊断_男方畸精症']=data_train['女方诊断'].apply(lambda x: 1 if( '男方畸精症' or '畸形精子症'or '男方弱畸精症' )in x else 0)
data_train['女方诊断_ART助孕失败']=data_train['女方诊断'].apply(lambda x: 1 if 'ART助孕失败'in x else 0)
data_train['女方诊断_盆腔子宫内膜异位症']=data_train['女方诊断'].apply(lambda x: 1 if '盆腔子宫内膜异位症'in x else 0)
data_train['女方诊断_男方重度少弱畸精症']=data_train['女方诊断'].apply(lambda x: 1 if( '男方重度少弱畸精症'or '男方极重度少精症')in x else 0)
data_train['女方诊断_PID后遗症']=data_train['女方诊断'].apply(lambda x: 1 if 'PID后遗症'in x else 0)
data_train['女方诊断_盆腔炎性疾病后遗症']=data_train['女方诊断'].apply(lambda x: 1 if '盆腔炎性疾病后遗症'in x else 0)
data_train['女方诊断_男方无精症']=data_train['女方诊断'].apply(lambda x: 1 if ('男方梗阻性无精症' or '男方非梗阻性无精症')in x else 0)
data_train['女方诊断_排卵障碍']=data_train['女方诊断'].apply(lambda x: 1 if '排卵障碍'in x else 0)
data_train['女方诊断_多囊卵巢综合症']=data_train['女方诊断'].apply(lambda x: 1 if('多囊卵巢综合症' or'PCOS' or '双侧卵巢多囊样改变')in x else 0)
data_train['女方诊断_卵巢低反应']=data_train['女方诊断'].apply(lambda x: 1 if '卵巢低反应'in x else 0)
data_train['女方诊断_高泌乳素血症']=data_train['女方诊断'].apply(lambda x: 1 if '高泌乳素血症'in x else 0)
data_train['女方诊断_复发性自然流产']=data_train['女方诊断'].apply(lambda x: 1 if '复发性自然流产'in x else 0)
data_train['女方诊断_卵巢功能减退']=data_train['女方诊断'].apply(lambda x: 1 if '卵巢功能减退'in x else 0)
data_train['女方诊断_精液不液化']=data_train['女方诊断'].apply(lambda x: 1 if '精液不液化'in x else 0)
data_train['女方诊断_输卵管缺如']=data_train['女方诊断'].apply(lambda x: 1 if '缺如'in x else 0)
data_train['女方诊断_子宫腺肌症']=data_train['女方诊断'].apply(lambda x: 1 if '子宫腺肌症'in x else 0)
data_train['女方诊断_IVF助孕未孕']=data_train['女方诊断'].apply(lambda x: 1 if 'IVF'in x else 0)
data_train['女方诊断_胰岛素抵抗']=data_train['女方诊断'].apply(lambda x: 1 if '胰岛素抵抗'in x else 0)
data_train['女方诊断_输卵管炎']=data_train['女方诊断'].apply(lambda x: 1 if '输卵管炎'in x else 0)
data_train['女方诊断_子宫肌瘤剥除术后']=data_train['女方诊断'].apply(lambda x: 1 if '子宫肌瘤剥除术后'in x else 0)
data_train['女方诊断_纵隔子宫']=data_train['女方诊断'].apply(lambda x: 1 if '纵隔子宫'in x else 0)
data_train['女方诊断_克氏综合症']=data_train['女方诊断'].apply(lambda x: 1 if '克氏综合症'in x else 0)
data_train['女方诊断_双侧输卵管结扎术后']=data_train['女方诊断'].apply(lambda x: 1 if '双侧输卵管结扎术后'in x else 0)
data_train['女方诊断_AIH失败']=data_train['女方诊断'].apply(lambda x: 1 if 'AIH失败'in x else 0)
data_train['女方诊断_反复宫外孕']=data_train['女方诊断'].apply(lambda x: 1 if '反复宫外孕'in x else 0)
data_train.drop('女方诊断',axis=1, inplace=True)

data_train['临床妊娠']=data_train['临床妊娠'].apply(lambda x: 1 if x=='阳性' else 0)
#data_train['生化妊娠']=data_train['生化妊娠'].apply(lambda x: 1 if x=='阳性' else 0)
#将特征分为数值型和非数值型
feature_num=[attr for attr in data_train.columns if data_train.dtypes[attr]!='object']
feature_nonum= [attr for attr in data_train.columns if data_train.dtypes[attr] == 'object']

#对离散值进行oneHot编码
for col in feature_nonum:
     for_dummy = data_train.pop(col)
     extra_data = pd.get_dummies(for_dummy,prefix=col)
     #print(col,":",extra_data.shape)
     data_train = pd.concat([data_train, extra_data],axis=1)



# def spearman(frame, features):
#     '''
#     采用“斯皮尔曼等级相关”来计算变量与房价的相关性(可查阅百科)
#     此相关系数简单来说，可以对上述encoder()处理后的等级变量及其它与房价的相关性进行更好的评价（特别是对于非线性关系）
#     '''
#     spr = pd.DataFrame()
#     spr['feature'] = features
#     spr['corr'] = [frame[f].corr(frame['临床妊娠'], 'spearman') for f in features]
#     spr = spr.sort_values('corr')
#     spr.to_csv("spr_featrueandoutput.csv", index=False)
#     # plt.figure(figsize=(6, 0.5*len(features)))
#     # sns.barplot(data=spr, y='feature', x='corr', orient='h')
#     # plt.show()
features = data_train.columns
# spearman(data_train, features)
import copy
#数值型变量间的相互关系
# corr_feature_num = data_train[feature_num].corr()
# #corr_feature_num.to_csv("corr_feature_num.csv", index=False)
# corr_feature_num_nlargest=corr_feature_num.nsmallest(10,'临床妊娠').index.values
# corr_feature_num_nlargestcopy=corr_feature_num_nlargest.tolist()
# corr_feature_num_nlargestcopy.append('临床妊娠')
# corr_feature_num_nlargestmat=corr_feature_num.loc[corr_feature_num_nlargestcopy,corr_feature_num_nlargestcopy]
# fig,ax = plt.subplots(figsize=(8,6))
# sns.set(font_scale=1.25)
# sns.heatmap(corr_feature_num_nlargestmat, annot=True, annot_kws={'size':10}, square=True)
# # # 设置annot使其在小格内显示数字，annot_kws调整数字格式
# ax.set_xticklabels(corr_feature_num_nlargestmat.index, rotation=90)
# ax.set_yticklabels(corr_feature_num_nlargestmat.index, rotation=0)
# plt.show()
#非数值型变量的相互关系
# feature_encode=list(filter(lambda x: x not in feature_num  ,features))
# corr_feature_nonum = data_train[feature_encode+['临床妊娠']].corr('spearman')
# #corr_feature_nonum.to_csv("corr_feature_nonum.csv", index=False)
# corr_feature_nonum_nlargest=corr_feature_nonum.nsmallest(10,'临床妊娠').index
# corr_feature_nonum_nlargestmat=corr_feature_nonum.loc[corr_feature_nonum_nlargest,corr_feature_nonum_nlargest]
# fig,ax = plt.subplots(figsize=(8,6))
# sns.set(font_scale=1.25)
# sns.heatmap(corr_feature_nonum_nlargestmat, annot=True, annot_kws={'size':10}, square=True)
# ax.set_xticklabels(corr_feature_nonum_nlargestmat.index, rotation=90)
# ax.set_yticklabels(corr_feature_nonum_nlargestmat.index, rotation=0)
# plt.show()

#将某些特征分布偏度比较大通过对数变化变成正态分布
from scipy.special import boxcox1p
from scipy.stats import skew

skew_features=data_train[feature_num].apply(lambda x : skew(x.dropna()))
skw_feature=skew_features[skew_features>0.75].index
data_train[skw_feature] = boxcox1p(data_train[skw_feature],0.15)
from sklearn.preprocessing import StandardScaler
 #标准化，返回值为标准化后的数据
StandardScaler().fit_transform(data_train)
output=data_train['临床妊娠']
data_train.drop('临床妊娠',axis=1,inplace=True)

#logistic
lr=linear_model.LogisticRegression(C=0.2)
predicted=np.mean(cross_val_score(lr,data_train,output,cv=10))
print(predicted)

#支持向量机
# clf=svm.SVC(C=4)
# predicted=np.mean(cross_val_score(clf,data_train,output,cv=10))
# # print(predicted)
# from sklearn import tree
# clf = tree.DecisionTreeClassifier()
#决策树
# from sklearn import tree
# clf = tree.DecisionTreeClassifier()
# predicted=np.mean(cross_val_score(clf,data_train,output,cv=10))
#随机森林
# from sklearn.ensemble import RandomForestClassifier
#  rlf = RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
# predicted=np.mean(cross_val_score(rlf,data_train,output,cv=10))

#adaboost
# from sklearn.ensemble import AdaBoostClassifier
# clf = AdaBoostClassifier(n_estimators=100)
# predicted=np.mean(cross_val_score(clf,data_train,output,cv=10))
#GBDT
# from sklearn.ensemble import GradientBoostingClassifier
#  clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
# ...     max_depth=1, random_state=0)
# predicted=np.mean(cross_val_score(clf,data_train,output,cv=10))