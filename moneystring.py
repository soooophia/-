# coding=utf-8
import numpy as np
import pandas as pd
data_train= pd.read_csv('Data_zhao/train.txt',encoding='utf-8',names=['ID','bal','day','rest_money'])
data_test= pd.read_csv('Data_zhao/test_input.txt',encoding='utf-8',names=['ID','bal','day','rest_money'])
money_train=pd.DataFrame(columns=range(0,90),index=data_train['ID'].unique())
money_test=pd.DataFrame(columns=range(0,90),index=data_train['ID'].unique())
table_train =pd.DataFrame.pivot_table(data_train, values='rest_money', index=['day'],columns=['ID'])
table_test =pd.DataFrame.pivot_table(data_test, values='rest_money', index=['day'],columns=['ID'])