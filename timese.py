from sklearn import svm
from sklearn.model_selection import cross_val_score
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd

from pylab import mpl                            #显示中文字体
mpl.rcParams['font.sans-serif'] = ['FangSong']   #指定默认字体：仿宋
mpl.rcParams['axes.unicode_minus'] = False       #解决保存图像是负号'-'显示为方块的问题
from matplotlib.font_manager import FontProperties
myfont=FontProperties(fname=r'C:\Windows\Fonts\simhei.ttf',size=14)
sns.set(font=myfont.get_name())

##导入文件，观察总体情况

data_train= pd.read_csv('data_time/international-airline-passengers.csv',encoding='utf-8')
pd.set_option('display.width', 1000)
data_train.drop(144,inplace=True)
data_train.rename( columns={'International airline passengers: monthly totals in thousands. Jan 49 ? Dec 60':'value'},inplace=True)
data_train['Month'].apply(lambda x :pd.datetime.strptime(x,'%Y-%m'))
data_train.set_index('Month',inplace=True)
ts=data_train['value']
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
plt.plot(ts)
ax.set_xticks(['1949-01','1951-01','1953-01','1955-01','1957-01','1959-01'])
from statsmodels.tsa.stattools import adfuller
dftest=adfuller(ts,autolag='AIC')
dfoutput=pd.Series(dftest[0:4],index=['Test statistic','p-value','Lags Used','Number of observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key]=value
print (dfoutput)