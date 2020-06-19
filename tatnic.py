import pandas as pd #数据分析
import numpy as np #科学计算
from pandas import Series,DataFrame
pd.set_option('display.width', 1000)
data_train = pd.read_csv("Data/Train.csv")
data_test = pd.read_csv("Data/test.csv")
data=pd.concat([data_test,data_train],axis=0).reset_index()
#data_train .info()
#print(data_train .describe())
from pylab import mpl                            #显示中文字体
mpl.rcParams['font.sans-serif'] = ['FangSong']   #指定默认字体：仿宋
mpl.rcParams['axes.unicode_minus'] = False       #解决保存图像是负号'-'显示为方块的问题
#
import matplotlib.pyplot as plt
fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

# plt.subplot2grid((2,3),(0,0))             # 在一张大图里分列几个小图
# data_train.Survived.value_counts().plot(kind='bar')# 柱状图
# plt.title(u"获救情况 (1为获救)") # 标题
# plt.ylabel(u"人数")
#
# plt.subplot2grid((2,3),(0,1))
# data_train.Pclass.value_counts().plot(kind="bar")
# plt.ylabel(u"人数")
# plt.title(u"乘客等级分布")
#
# plt.subplot2grid((2,3),(0,2))
# plt.scatter(data_train.Survived, data_train.Age)
# plt.ylabel(u"年龄")                         # 设定纵坐标名称
# plt.grid(b=True, which='major', axis='y')
# plt.title(u"按年龄看获救分布 (1为获救)")
#
#
# plt.subplot2grid((2,3),(1,0), colspan=2)
# data_train.Age[data_train.Pclass == 1].plot(kind='kde')
# data_train.Age[data_train.Pclass == 2].plot(kind='kde')
# data_train.Age[data_train.Pclass == 3].plot(kind='kde')
# plt.xlabel(u"年龄")# plots an axis lable
# plt.ylabel(u"密度")
# plt.title(u"各等级的乘客年龄分布")
# plt.legend((u'头等舱', u'2等舱',u'3等舱'),loc='best') # sets our legend for our graph.
#
#
# plt.subplot2grid((2,3),(1,2))
# data_train.Embarked.value_counts().plot(kind='bar')
# plt.title(u"各登船口岸上船人数")
# plt.ylabel(u"人数")
# plt.show()
#
# fig = plt.figure()
# fig.set(alpha=0.2)   #设置图表颜色透明度
# #看看各等级乘客的获救情况
# Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
# Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
# df = pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
# df.plot(kind='bar', stacked=True)       #条形图堆叠
# plt.title(u"各乘客等级的获救情况")
# plt.xlabel(u"乘客等级")
# plt.ylabel(u"人数")
#
# plt.show()

#看看各性别的获救情况
# fig = plt.figure()
# fig.set(alpha=0.2)
#
# Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
# Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
# df = pd.DataFrame({u'男性':Survived_m, u'女性':Survived_f})
# df.plot(kind='bar', stacked=True)
# plt.title(u"按性别看获救情况（1为获救）")
# plt.xlabel(u"性别")
# plt.ylabel(u"人数")
#
# plt.show()
#
# fig = plt.figure()
# fig.set(alpha=0.65)    #设置图表颜色透明度
# plt.title(u"各舱等级和性别的获救情况")
#
#
# axl = fig.add_subplot(141)   #参数141的意思: 将画布分割成1行4列，图像画在从左到右的第1块
# ds1 = data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts()  #1等舱/2等舱为高级舱
# ds1.plot(kind='bar', label="female highclass", color='#FA2479')
# axl.set_xticklabels([u"获救", u"未获救"], rotation=0)    #设置x轴标签，且标签不旋转
# axl.set_ylim(0, 350)     #设置y轴坐标刻度
# axl.legend([u"女性/高级舱"], loc='best')    #设置图例
#
# ax2 = fig.add_subplot(142)
# ds2 = data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts()  #3等舱为低级舱
# ds2.plot(kind='bar', label='female low class', color='pink')
# ax2.set_xticklabels([u"未获救", u"获救"], rotation=0)
# ax2.set_ylim(0, 350)     #设置y轴坐标刻度，保证子图纵坐标一致
# plt.legend([u"女性/低级舱"], loc='best')
#
# ax3 = fig.add_subplot(143)
# ds3 = data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts()
# ds3.plot(kind='bar', label='male highclass', color='lightblue')
# ax3.set_xticklabels([u"未获救", u"获救"], rotation=0)
# ax3.set_ylim(0, 350)     #设置y轴坐标刻度，保证子图纵坐标一致
# plt.legend([u"男性/高级舱"], loc='best')
#
# ax4 = fig.add_subplot(144)
# ds4 = data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts()
# ds4.plot(kind='bar', label='male low class', color='steelblue')
# ax4.set_xticklabels([u"未获救", u"获救"], rotation=0)
# ax4.set_ylim(0, 350)     #设置y轴坐标刻度，保证子图纵坐标一致
# plt.legend([u"男性/低级舱"], loc='best')
#
# plt.show()

#各登船港口的获救情况
# fig = plt.figure()
# fig.set(alpha=0.2)
#
# Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
# Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
# df = pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
# df.plot(kind='bar', stacked=True)
# plt.title(u"各登录港口乘客的获救情况")
# plt.xlabel(u"登录港口")
# plt.ylabel(u"人数")
#
# plt.show()

#简单数据预处理
from sklearn.ensemble import RandomForestRegressor


#使用RandomForestClassfier(分类器)填补缺失的Age属性
def set_missing_ages(df):

    #把已有的数值特征取出来放到RandomForestRegressor中
    age_df = df[['Age','Fare','Parch','SibSp','Pclass']]

    #乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    #y即目标年龄
    y = known_age[:,0]

    #x即特征属性
    X = known_age[:,1:]

    #fit到RandomForestRegressor中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X,y)

    #用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:,1::])

    #用得到的预测结果填补原缺失数据
    df.loc[(df.Age.isnull()), 'Age'] = predictedAges

    return df, rfr


#将Cabin按有无数据，把属性处理成Yes和No两种类型
def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = "Yes"
    df.loc[(df.Cabin.isnull()), 'Cabin'] = "No"

    return df


data['Fare'].fillna(data['Fare'].median(), inplace = True)
data, rfr = set_missing_ages(data)
data = set_Cabin_type(data)

#检查Age/Cabin的缺失值是否填补
#print(data_train)

#对类目型的特征进行因子化：Cabin/Embarked/Sex/Pclass
dummies_Cabin = pd.get_dummies(data['Cabin'], prefix='Cabin')
dummies_Embarked = pd.get_dummies(data['Embarked'], prefix='Embarked')
dummies_Sex = pd.get_dummies(data['Sex'], prefix='Sex')
dummies_Pclass = pd.get_dummies(data['Pclass'], prefix='Pclass')

#将因子化后的数据拼接在原来的 "data_train" 上
df = pd.concat([data, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)

#并将下面几个原始字段从 "data_train" 中拿掉
df.drop(['Pclass','Sex','Cabin','Embarked'], axis=1, inplace=True)



#将Age/Fare标准化到[-1,1]范围内
# import sklearn.preprocessing as preprocessing
# scaler = preprocessing.StandardScaler()

#data.values.reshape(-1, 1)是根据报错信息修改的，主要是因为工具包版本更新造成的
# age_scale_param = scaler.fit(df['Age'].values.reshape(-1, 1))
# df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1, 1), age_scale_param)
#
# fare_scale_param = scaler.fit(df['Fare'].values.reshape(-1, 1))
# df['Fare_scaled'] =scaler.fit_transform(df['Fare'].values.reshape(-1, 1), fare_scale_param)

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

df['IsAlone'] = 1  # initialize to yes/1 is alone
df['IsAlone'].loc[df['FamilySize'] > 1] = 0

df['Title'] = df['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

df['FareBin'] = pd.qcut(df['Fare'], 4)
df['AgeBin'] = pd.cut(df['Age'].astype(int), 5)
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
label = LabelEncoder()

df['Title_Code'] = label.fit_transform(df['Title'])
df['AgeBin_Code'] = label.fit_transform(df['AgeBin'])
df['FareBin_Code'] = label.fit_transform(df['FareBin'])

df.drop(['Fare','Age','Title'], axis=1, inplace=True)



from sklearn import linear_model

#用正则regex取出我们要的属性值
# ".*"是正则表达式中的贪婪模式，匹配任意字符0或者多次(大于等于0次)。点是任意字符，*是取0至无限长度。
# train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')   #共取了15个feature字段
# train_np = train_df.as_matrix()      #转成numpy格式
#
# #y即Survived结果
# y = train_np[:, 0]        #切片含义: 所有行/第1列，即Survived列
#
# #X即特征属性值
# X = train_np[:, 1:]       #切片含义: 所有行/第2列到最后一列，即Age_scaled列到Pclass_3列，共14个字段。

#fit到RandomForestRegressor中
#penalty='l1'第一个是英文字母L的小写，不是数字1；
#tol=1e-6代表科学技术法，即1乘以10的-6次方，注意这里的1不能省略，因为可能造成歧义；也可以用tol=0.000001表达。
# clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)     #创建分类器对象
# clf.fit(X, y)     #用训练数据拟合分类器模型
#
# print(clf)

#test.csv不能直接放到model，和train数据一样要先做预处理


# data_test.loc[(data_test.Fare.isnull()), 'Fare'] = 0
#
# #接着我们对test做和train中一致的特征变换
# #首先用同样的RandomForestRegressor模型填上丢失的年龄
# tmp_df = data_test[['Age','Fare','Parch','SibSp','Pclass']]
# null_age = tmp_df[data_test.Age.isnull()].as_matrix()
#
# #根据特征属性X预测缺失的年龄并补上
# X = null_age[:, 1:]
# predictedAges = rfr.predict(X)
# data_test.loc[(data_test.Age.isnull()), 'Age'] = predictedAges
#
# data_test = set_Cabin_type(data_test)
# dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix='Cabin')
# dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix='Embarked')
# dummies_Sex = pd.get_dummies(data_test['Sex'], prefix='Sex')
# dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix='Pclass')

# df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
# df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
# df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'].values.reshape(-1, 1), age_scale_param)
# df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'].values.reshape(-1, 1), age_scale_param)

#打印预处理后的数据，检查处理效果
# print(df_test)
#
# #数据处理完成，下面做预测取结果
# test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# predictions = clf.predict(test)     #用训练好的分类器去预测test数据的标签
# result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
#
# #保存预测结果
# result.to_csv("logistic_regression_predictions_tatnic.csv", index=False)
from xgboost import XGBClassifier
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
#Common Model Helpers
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix

MLA = [
    # Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),

    # Gaussian Processes
    gaussian_process.GaussianProcessClassifier(),

    # GLM
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    linear_model.Perceptron(),

    # Navies Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),

    # Nearest Neighbor
    neighbors.KNeighborsClassifier(),

    # SVM
    svm.SVC(probability=True),
    svm.NuSVC(probability=True),
    svm.LinearSVC(),

    # Trees
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),

    # Discriminant Analysis
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),

    # xgboost: http://xgboost.readthedocs.io/en/latest/model.html
    XGBClassifier()
]
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics
data_test=df[0:417]
data1=df[418:]
