from __future__ import print_function

import baostock as bs

import pandas as pd

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import matplotlib.font_manager as matfont
matplotlib.rcParams['axes.unicode_minus']=False  #显示负号'-'
plt.rcParams['font.family'] = ['sans-serif']
# 显示中文。  如果是在 PyCharm 里，只要下面一行，上面的一行可以删除
plt.rcParams['font.sans-serif'] = ['SimHei']


import datetime

from scipy import  stats

import statsmodels.api as sm

from statsmodels.graphics.api import qqplot
 
from statsmodels.graphics.tsaplots import acf,pacf,plot_acf,plot_pacf


def get_shibor_data(startDate,endDate):

    """获取历史shibor数据"""

    # 登陆系统

    lg = bs.login()

    # 显示登陆返回信息

    print('login respond error_code:'+lg.error_code)

    print('login respond  error_msg:'+lg.error_msg)
   
    # 获取银行间同业拆放利率

    rs = bs.query_shibor_data(start_date=startDate, end_date=endDate)

    print('query_shibor_data respond error_code:'+rs.error_code)

    print('query_shibor_data respond  error_msg:'+rs.error_msg)
   
    # 打印结果集

    data_list = []

    while (rs.error_code == '0') & rs.next():

        # 获取一条记录，将记录合并在一起

        data_list.append(rs.get_row_data())

    result = pd.DataFrame(data_list, columns=rs.fields)
    print('shiborON_list:', data_list)
  
    # 登出系统

    bs.logout()

    return result

def plot_shibor_pic():

    startdate = "2019-01-18"    

    endate = "2019-11-06"    #这里调整shibor取数的时期

    global shiborData

    shiborData = get_shibor_data(startdate, endate)

    datelist = shiborData.loc[:,'date']

    shiborON = shiborData.loc[:,'shiborON']

def shibor_ARIMA():
    data = shiborData

    data.to_csv("shibor_data.csv")

    df = pd.read_csv('shibor_data.csv')

    global shibor_term_set
    shibor_term = ['shiborON', 'shibor1W', 'shibor2W', 'shibor1M', 'shibor3M', 'shibor6M', 'shibor9M', 'shibor1Y']
    shibor_term_set = shibor_term[0]   #从0开始，不同的序号表示要处理的不同期限产品

    dta = pd.Series(df[shibor_term_set].values, index=df['date'].values)
     
    plt.show()

    # 2时间序列的差分d
    fig = plt.figure(figsize=(12,8))
    ax1= fig.add_subplot(111)
    diff1 = dta.diff(1)
    sum_nan_diff1 =  diff1.isnull().sum().sum()
    print('在我们diff1中NaN的数量:', sum_nan_diff1)
    where_are_nan = np.isnan(diff1)  #定位nan；用0替换nan
    diff1[where_are_nan] = 0
    diff1.plot(ax=ax1)
    #以下是通过了 ADF 检验，查看序列是否平稳。通过观察t统计量是否小于置信度的临界值
    t=sm.tsa.stattools.adfuller(dta)
    # t=sm.tsa.stattools.adfuller(diff1)
    output=pd.DataFrame(index=['Test Statistic Value', "p-value", "Lags Used", "Number of Observations Used","Critical Value(1%)","Critical Value(5%)","Critical Value(10%)"],columns=['value'])
    output['value']['Test Statistic Value'] = t[0]
    output['value']['p-value'] = t[1]
    output['value']['Lags Used'] = t[2]
    output['value']['Number of Observations Used'] = t[3]
    output['value']['Critical Value(1%)'] = t[4]['1%']
    output['value']['Critical Value(5%)'] = t[4]['5%']
    output['value']['Critical Value(10%)'] = t[4]['10%']
    print(output)
    #确定自相关系数和平均移动系数（p,q）
    #根据时间序列的识别规则，采用 ACF 图、PAC 图，AIC 准则（赤道信息量准则）和 BIC 准则（贝叶斯准则）
    #相结合的方式来确定 ARMA 模型的阶数, 应当选取 AIC 和 BIC 值达到最小的那一组为理想阶数。
    plot_acf(diff1)
    plot_pacf(diff1)
    plt.show()

    r,rac,Q = sm.tsa.acf(diff1, qstat=True)
    prac = pacf(diff1,method='ywmle')
    table_data = np.c_[range(1,len(r)), r[1:],rac,prac[1:len(rac)+1],Q]
    table = pd.DataFrame(table_data, columns=['lag', "AC","Q", "PAC", "Prob(>Q)"])

    print('table',table)
    
    (p, q) =(sm.tsa.arma_order_select_ic(diff1,max_ar=3,max_ma=3,ic='aic')['aic_min_order'])
    #这里需要设定自动取阶的 p和q 的最大值，即函数里面的max_ar,和max_ma。
    #ic 参数表示选用的选取标准，这里设置的为aic,当然也可以用bic。
    #然后函数会算出每个 p和q 组合(这里是(0,0)~(3,3)的AIC的值，取其中最小的.

    print('p = %d , q = %d' %(p,q))

    #预测
    arima_model = sm.tsa.ARIMA(dta,(8,1,0)).fit()
    # predict_data = arma_model.predict(start=str(1979), end=str(2010+3), dynamic = False)
    predict_data = arima_model.predict(start="2019-10-25", end="2019-11-06",dynamic=True,typ='levels')
    shibor_forcast = pd.concat([dta['2019-09-21':"2019-11-06"],predict_data],axis=1,keys=['original', 'predicted'])  
    #将原始数据和预测数据相结合，使用keys来分层
    plt.figure()

    plt.plot(shibor_forcast)

    plt.title(shibor_term_set+' 真实值vs预测值')

    plt.xticks(rotation=50)
    plt.show()
if __name__ == '__main__':

    plot_shibor_pic()
    shibor_ARIMA()
