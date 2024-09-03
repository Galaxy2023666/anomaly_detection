# coding: UTF-8

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose, STL
import math
import os


import matplotlib.pylab as plt

import datetime as dt
import chinese_calendar as calendar

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from fbprophet import Prophet

import logging

from tools.r_pca import _check_representation, _scale, _reshape_pad, _autodiff, _mad_outlier
from tools.r_pca import RPCA as rpca

from tools.statistics_calculation import *

#处理数据
def process_data(data_file, dim_cols, value_col='value', order_col='年月'): #, index_col='指标名称'
    data = pd.read_csv(data_file)

    data_long = pd.melt(data, id_vars=dim_cols, var_name=order_col, value_name=value_col)
    data_long.loc[:, 'Month'] = pd.to_datetime(data_long.loc[:, order_col], format='%Y-%m')
    data_long.drop(order_col, axis=1, inplace=True)
    data_long.loc[:, value_col] = data_long.loc[:, value_col].apply(
        lambda x: 0.0 if (x == '-' or x == '/' or x == '' or pd.isnull(x)) else x
    )

    data_long.loc[:, 'value'] = data_long.loc[:, 'value'].apply(
        lambda x: 0.0 if pd.isnull(x) else x
    )

    data_long.loc[:, 'value'] = pd.to_numeric(data_long['value'], errors='coerce').fillna(0)

    return data_long

# 分解时间序列各个因素
class ProphetDecomposeResult(object):
    def __init__(self, observed, trend, yearly, weekly, holidays, resid):
        self._yearly = yearly
        self._weekly = weekly
        self._trend = trend
        self._resid = resid
        self._observed = observed
        self._holidays = holidays

    @property
    def observed(self):
        """Observed data"""
        return self._observed

    @property
    def holidays(self):
        """holidays component"""
        return self._holidays

    @property
    def trend(self):
        """The estimated trend component"""
        return self._trend

    @property
    def resid(self):
        """The estimated residuals"""
        return self._resid

    @property
    def yearly(self):
        """The yearly season"""
        return self._yearly

    @property
    def weekly(self):
        """The weekly season"""
        return self._weekly

    @property
    def nobs(self):
        """Number of observations"""
        return self._observed.shape

# def detect(series_, anomaly, if_plot=False):
#     ts = convert_2_ts(series_)
#
#     result = anomaly.detect(ts, if_plot=if_plot)
#
#     return result

def convert_2_ts(series_):
    df = series_.to_frame().reset_index()
    df.columns = ['ds', 'value']
    df.index = df.loc[:, 'ds']

    df.drop('ds', axis=1, inplace=True)

    ds = df.loc[:, 'value']

    return ds

def identify_single_unique(df):
        """Finds features with only a single unique value. NaNs do not count as a unique value. """

        # Calculate the unique counts in each column
        unique_counts = df.nunique()

        # Find the columns with only one unique count
        record_single_unique = pd.DataFrame(unique_counts[unique_counts == 1]).reset_index().rename(
            columns={
                'index': 'feature',
                0: 'nunique'
            }
        )

        to_drop = list(record_single_unique['feature'])
        print("所有值一样的列为：{}". format(','.join(to_drop)))

        df.drop(to_drop, 1, inplace=True)

        logging.info('%d features with a single unique value.\n' % len(to_drop))

        return

# 异常检测算法基类
class AnomalyDetection(object):
    def __init__(self):
        object.__init__(self)

    def detect(self, **kwargs):
        """
        :param data:类型为series的序列数据
        :return:返回可能存在的异常值
        """
        pass

    def detect_from_file(self, data_file):
        pass

    def test_stationarity(self, data, if_plot=True):
        """
        ADF检验的原假设是不平稳，这里P值小于0.05， 则拒绝原假设，认为序列平稳。
        :param data: 时序数据
        :return:
        """
        # 汉字字体，优先使用楷体，找不到则使用黑体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

        # 正常显示负号
        plt.rcParams['axes.unicode_minus'] = False

        rolmean = pd.Series(data).rolling(window=12).mean()
        rolstd = pd.Series(data).rolling(window=12).std()

        # Plot rolling statistics:
        if if_plot:
            plt.plot(data, color='blue', label='原始数据')
            plt.plot(rolmean, color='red', label='移动平均数')
            plt.plot(rolstd, color='black', label='移动方差')
            plt.legend(loc='best')
            plt.title('移动平均与移动方差')
            plt.show()

            plot_acf(data)
            plt.show()
            plot_pacf(data)
            plt.show()

        # Perform Dickey-Fuller test 单位根检验 如果存在根在1以内，则非平稳:
        print('Results of Dickey-Fuller Test:')
        # 返回值依次为：adf, pvalue p值， usedlag, nobs, critical values临界值 , icbest, regresults, resstore
        df_test = adfuller(data, autolag='AIC')
        df_output = pd.Series(df_test[0:4],
                              index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
        for key, value in df_test[4].items():
            df_output['Critical Value (%s)' % key] = value

        print("The p-value of stationarity test is {}".format(df_output['p-value']))

        return df_output['p-value'] #若小于0.05，则可以认为拒绝原假设，数据不存在单位根，序列平稳；若大于或等于0.05，则不能显著拒绝原假设，需要进行下一步判断

    def test_noisy(self, data):
        """
        如果p值大于0.05，则无法拒绝原假设，即认为该数据是纯随机数据
        :param data:
        :return:
        """
        df_test = acorr_ljungbox(data, lags=1)
        df_test.loc[:, 'lb_pvalue'].values.tolist()[0]

        print("The p-value of noisy test is {}".format(df_test.loc[:, 'lb_pvalue'].values.tolist()[0]))

        return df_test.loc[:, 'lb_pvalue'].values.tolist()[0]

    def plot_fitted_result(self, origianl_value, fitted_value):
        """
        :param origianl_value: 原始数据
        :param fitted_value: 拟合数据
        :return:
        """
        plt.plot(origianl_value, label='原始数据')
        plt.plot(fitted_value, color='red', label='拟合数据')
        plt.title('RSS: %.4f' % sum((origianl_value - origianl_value) ** 2))
        plt.show()

# 基于统计的异常检测
class AnomalyDetectionOnStatistics(AnomalyDetection):
    def __init__(self, alpha, max_outliers):
        """
        :param alpha:显著性水平
        :param max_outliers:需要检测的异常值数量上限
        """
        AnomalyDetection.__init__(self)
        self.alpha = alpha
        self.max_outliers = max_outliers

    def detect_from_file(self, data_file):
        pass

    def detect(self, **kwargs):
        """
        :param data:
        :return:
        Extreme Studentized Deviate test
        https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h3.htm
        https://www.cnblogs.com/en-heng/p/9202654.html
        example:
        import matplotlib.pyplot as plt

        y = np.random.random(100)
        x = np.arange(len(y))

        y[14] = 9
        y[83] = 10
        y[44] = 14

        plt.scatter(x,y)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

        test = AnomalyDetectionOnStatistics(0.05, 7)

        test.detect(y)
        """
        max_i = -1

        statss = []
        critical_vals = []
        max_index_list = []
        #转换数据结构
        _data = convert_2_ts(kwargs['data'][:])
        #Extreme Studentized Deviate test 进行K次test
        for iterations in range(1, self.max_outliers + 1):
            #这里的test_stat，calculate_critical_value，check_values从哪里调用的？？？
            stat, max_index = test_stat(_data, iterations)
            critical = calculate_critical_value(len(_data), self.alpha, iterations)
            check_values(stat, critical, _data, max_index, iterations)
            _data = np.delete(_data, max_index)
            critical_vals.append(critical)
            statss.append(stat)
            max_index_list.append(max_index)
            if stat > critical:
                max_i = iterations
        print('H0:  there are no outliers in the data')
        print('Ha:  there are up to 10 outliers in the data')
        print('')
        print('Significance level:  α = {}'.format(self.alpha))
        print('Critical region:  Reject H0 if Ri > critical value')
        print('Ri: Test statistic')
        print('λi: Critical Value')
        print(' ')
        df = pd.DataFrame({'i': range(1, self.max_outliers + 1), 'R': statss, 'C': critical_vals, 'index': max_index_list})

        df.loc[:, 'is_anomaly'] = df.apply(lambda x: 'Yes' if x['R'] > x['C'] else 'No', axis=1)

        df.index = df.index + 1
        print('Number of outliers {}'.format(max_i))

        df_anomaly = df.query("is_anomaly == 'Yes'")

        anomaly_indices = df_anomaly.loc[:, 'index'].values.tolist()
        # anomalies = np.asarray([])
        anomalies = pd.Series()
        if not len(anomaly_indices) == 0:
            anomalies = kwargs['data'][anomaly_indices]

        return anomalies

# 基于时间序列的异常检测
class AnomalyDetectionOnTimeSeries(AnomalyDetection):
    def __init__(self):
        return

    def get_seasonal_decompose_resid(self, data, period=None, method=None, if_plot=True):
        """
        :param data:
        :param method: 分级方法
                      classic: 经典分解法
                               特点：
                                   该技术对异常值不可靠。
                                   它倾向于使时间序列数据中的突然上升和下降过度平滑。
                                   假设季节性因素每年只重复一次。
                                   对于前几次和最后几次观察，该方法都不会产生趋势周期估计。
                      stl: stl分解法
                               特点：
                                   趋势周期平滑度
                                   季节性变化率
                                   可以控制对用户异常值或异常值的鲁棒性。这样你就可以控制离群值对季节性和趋势性的影响。
                      prophet：Prophet模型
        :return:
        """
        # decomposition = None
        if method == 'stl':
            stl = STL(data, period=period, robust=True)
            decomposition = stl.fit()
            # fig = decomposition.plot()
        elif method == 'prophet':
            # holidays_df = pd.read_csv(holiday_file)
            # holidays_df.loc[:, 'ds'] = holidays_df.loc[:, 'ds'].apply(
            #     lambda x: [dt.datetime.strptime(i, "%Y-%m-%d") for i in x]
            # )
            # holidays_df.loc[:, 'ds'] = pd.to_datetime(holidays_df.loc[:, 'ds'], format='%Y-%m-%d')
            # m = Prophet(yearly_seasonality=True, weekly_seasonality=False, holidays=holidays_df)

            m = Prophet(yearly_seasonality=True, weekly_seasonality=False)

            # df.loc[:, 'ts'] = pd.DataFrame(data)
            df = pd.DataFrame(data)
            df = df.reset_index()
            df.columns = ['ds', 'y']
            m.add_country_holidays(country_name='CN')
            m.fit(df)

            horizon = 0
            future = m.make_future_dataframe(periods=horizon)
            forecast = m.predict(future)

            df_all = pd.merge(forecast, df)
            df_all.loc[:, 'resid'] = df_all.loc[:, 'y'] - df_all.loc[:, 'trend'] - df_all.loc[:, 'holidays'] - df_all.loc[:, 'yearly']
            # observed = convert_2_ts(df_all.loc[:, 'y'])
            observed = convert_2_ts(pd.Series(df_all.loc[:, 'y'].values, index=df_all.loc[:, 'ds'].values))
            resid = convert_2_ts(pd.Series(df_all.loc[:, 'resid'].values, index=df_all.loc[:, 'ds'].values))
            trend = convert_2_ts(pd.Series(df_all.loc[:, 'trend'].values, index=df_all.loc[:, 'ds'].values))
            yearly = convert_2_ts(pd.Series(df_all.loc[:, 'yearly'].values, index=df_all.loc[:, 'ds'].values))
            # weekly = convert_2_ts(df_all.loc[:, 'weekly'])
            holidays = convert_2_ts(pd.Series(df_all.loc[:, 'holidays'].values, index=df_all.loc[:, 'ds'].values))

            decomposition = ProphetDecomposeResult(observed, trend, yearly, weekly=None, holidays=holidays, resid=resid)
        else:
            decomposition = seasonal_decompose(data, model='additive', period=7)

        decomposition_resid = decomposition.resid
        data_compare = decomposition_resid[~np.isnan(decomposition_resid)]
        stationarity_result = self.test_stationarity(data_compare, if_plot=if_plot)

        if stationarity_result < 0.05:
        #if stationarity_result['p-value'] < 0.05:
            print("时序分解后的残差是平稳的")
        else:
            print("时序分解后的残差是不平稳的")

        noisy_result = self.test_noisy(data_compare)
        if noisy_result < 0.05:
            print("时序分解后的残差不是白噪声")
        else:
            print("时序分解后的残差是白噪声")

        # todo 如果需要预测未来才需要
        # p, q = self.get_best_p_d_q(data_compare)
        # arma_mod = ARIMA(data_compare, order=(p, 1, q)).fit()
        # self.plot_fitted_result(data_compare, arma_mod.resid)

        if if_plot:
            self.plot_trend_and_cycle(data, decomposition)

        return decomposition

#上下界
    def get_range_within_sigma(self, data, method='stl', sigma=2, if_plot=True):
        ts = convert_2_ts(data)

        decomposition = self.get_seasonal_decompose_resid(ts, method=method, if_plot=if_plot)

        resid = decomposition.resid
        resid = resid[~np.isnan(resid)]
        # resid_mean = np.mean(resid)
        resid_std = np.std(resid)

        resid_df = resid.to_frame()

        low = []
        high = []

        for re in resid:
            low.append(re-sigma*resid_std)
            high.append(re+sigma*resid_std)

        resid_df.insert(resid_df.shape[1], 'low', low)
        resid_df.insert(resid_df.shape[1], 'high', high)
        resid_df.columns = ['resid', 'low', 'high']
        print(resid_df)
        return resid_df


    def get_range_mad(self, data, method='stl', k=3, if_plot=True):
        ts = convert_2_ts(data)

        decomposition = self.get_seasonal_decompose_resid(ts, method=method, if_plot=if_plot)

        resid = decomposition.resid
        resid = resid[~np.isnan(resid)]
        # resid_mean = np.mean(resid)

        resid_df = resid.to_frame()
        medi_point = resid.median()  # 序列的中位数
        list_value = []
        for index, value in resid.items():
            list_value.append(abs(value - medi_point))
        median = np.median(list_value)
        mad = median * 1.4826 # 绝对偏差中位数
        low = medi_point - mad * k
        high = medi_point + mad * k
        resid_df.insert(resid_df.shape[1], 'low', low)
        resid_df.insert(resid_df.shape[1], 'high', high)
        resid_df.columns = ['resid', 'low', 'high']
        print(resid_df)
        return resid_df

    def get_range_boxplot(self, data, method='stl', if_plot=True):
        ts = convert_2_ts(data)

        decomposition = self.get_seasonal_decompose_resid(ts, method=method, if_plot=if_plot)

        resid = decomposition.resid
        resid = resid[~np.isnan(resid)]
        # resid_mean = np.mean(resid)

        resid_df = resid.to_frame()
        q3 = resid.quantile(0.75)
        q1 = resid.quantile(0.25)
        high = q3 + 1.5 * (q3 - q1)
        low = q1 - 1.5 * (q3 - q1)
        resid_df.insert(resid_df.shape[1], 'low', low)
        resid_df.insert(resid_df.shape[1], 'high', high)
        resid_df.columns = ['resid', 'low', 'high']
        return resid_df

    def detect_from_file(self, data_file, value_colname, index_colname='Month', parse_dates=['Month'],
                         data_format='%Y-%m'):
        """
        :param data_file: 数据文件
        :param value_colname: 观测值的列名
        :param index_colname: 索引值的列名
        :param parse_dates: 日期
        :param data_format: 日期格式
        """
        AnomalyDetection.__init__(self)
        dateparse = lambda dates: pd.datetime.strptime(dates, data_format)

        data = pd.read_csv(data_file, parse_dates=parse_dates, index_col=index_colname, date_parser=dateparse)
        ts = data[value_colname]

        return self.detect(ts)

    def plot_trend_and_cycle(self, data, decomposition):
        # 汉字字体，优先使用楷体，找不到则使用黑体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

        # 正常显示负号
        plt.rcParams['axes.unicode_minus'] = False

        plt.subplot(411)
        plt.plot(data, label='原始数据')
        plt.legend(loc='best')
        plt.subplot(412)
        plt.plot(decomposition.trend, label='趋势数据')
        plt.legend(loc='best')
        plt.subplot(413)
        plt.plot(decomposition.seasonal, label='周期数据')
        plt.legend(loc='best')
        plt.subplot(414)
        plt.plot(decomposition.resid, label='残差数据')
        plt.legend(loc='best')
        plt.tight_layout()

        plt.show()



class AnomalyDetectionOnArima(AnomalyDetectionOnTimeSeries):
    def __init__(self):
        AnomalyDetectionOnTimeSeries.__init__(self)
        self.model = None

    def get_best_p_d_q(self, data):
        p = 0
        q = 0
        output = self.test_stationarity(data)

#用BIC 定阶
        if output['p-value'] < 0.05:
            pmax = int(len(data) / 10)  # 一般阶数不超过 length /10
            qmax = int(len(data) / 10)
            bic_matrix = []
            for p_i in range(pmax + 1):
                temp = []
                for q_i in range(qmax + 1):
                    # try:
                    #     temp.append(ARIMA(data_diff, oder=(p_i, 1, q_i)).fit().bic)
                    # except:
                    #     temp.append(None)
                    temp.append(ARIMA(data, order=(p_i, 1, q_i)).fit().bic)
                    bic_matrix.append(temp)

            bic_matrix = pd.DataFrame(bic_matrix)  # 将其转换成Dataframe 数据结构
            p, q = bic_matrix.stack().idxmin()  # 先使用stack 展平， 然后使用 idxmin 找出最小值的位置
            print('BIC 最小的p值 和 q 值：%s,%s' % (p, q))  # BIC 最小的p值 和 q 值：0,1

        return p, q

    def detect(self, **kwargs):
        # data, sigma = 3, method = 'stl', if_plot = True
        data = convert_2_ts(kwargs['data'])
        method = kwargs['method']
        if_plot = kwargs['if_plot']
        sigma = kwargs['sigma']

        decomposition = self.get_seasonal_decompose_resid(data, method, if_plot)
        resid = decomposition.resid
        resid_mean = np.mean(resid)
        resid_std = np.std(resid)

        anomalies = []
        anomaly_indices = []
        for i in range(len(resid)):
            if abs((resid[i]-resid_mean)/resid_std) > sigma:
                anomaly_indices.append(i)

        if not len(anomaly_indices) == 0:
            anomalies = data[anomaly_indices]

        return anomalies

# 时间序列异常检测算法S-H-ESD
class AnomalyDetectionOnSeasonalESD(AnomalyDetectionOnTimeSeries):
    """
    example：
    from tools.plot import plot_anomalies

    esd = SeasonalESD(anomaly_ratio=0.1, hybrid=True, alpha=0.05)
    test_data = np.asarray(
        [-0.25, 0.98, 0.94, 1.15, 1.20, 1.26, 1.26, 1.34, 1.38, 1.43, 1.49, 1.49, 1.55, 1.56, 1.58, 1.65, 1.69, 1.70,
         1.76, 1.77, 1.81, 1.91, 1.94, 1.96, 1.99, 2.06, 2.09, 2.10, 2.14, 2.15, 2.23, 2.24, 2.26, 2.35, 2.37, 2.40,
         2.47, 2.54, 2.62, 2.64, 2.90, 2.92, 2.92, 2.93, 3.21, 3.26, 3.30, 3.59, 3.68, 4.30, 4.64, 5.34, 5.42, 6.01])
    anomalies, indices = esd.detect(test_data)
    plot_anomalies(test_data, indices, anomalies)

    """
    def __init__(self, anomaly_ratio, hybrid, alpha):
        """
        :param anomaly_ratio:
        :param hybrid: 确定是用s-esd还是s-head-esd
        :param alpha: 显著性水平
        """
        AnomalyDetectionOnTimeSeries.__init__(self)
        self.anomaly_ratio = anomaly_ratio
        self.hybrid = hybrid
        self.alpha = alpha

        assert self.anomaly_ratio <= 0.49, "anomaly ratio is too high"

        self.n = None
        self.k = None

        self.count = 0

    def _get_residuals(self, data, method):
        median = np.median(data)
        result = self._get_seasonal_decomposition(data, method)
        # residuals = np.asarray(data - result.seasonal - median)
        residuals = np.asarray(result.resid)
        return residuals

    # @staticmethod
    def _get_seasonal_decomposition(self, data, method='stl', period=7):
        # result = seasonal_decompose(data, model=model, period=period)
        result = self.get_seasonal_decompose_resid(data, method=method, period=period, if_plot=False)
        return result

    def detect(self, **kwargs):
        self.count += 1
        print("run number: {}".format(self.count))
        data = convert_2_ts(kwargs['data'])
        if_plot = kwargs['if_plot']
        method = kwargs['method']

        self.n = data.shape[0]
        self.k = math.floor(float(self.n) * self.anomaly_ratio)

        residuals = self._get_residuals(data, method)

        # 为了避免后续重复计算，这里需要保存残差
        self.test_stationarity(residuals, if_plot)

        critical_values = self._calc_critical_values()

        anomaly_indices = self._esd(residuals, critical_values)
        # anomalies = np.asarray([])
        anomalies = pd.Series(dtype='float')
        if not anomaly_indices.size == 0:
            anomalies = data[anomaly_indices]

        return anomalies

    def _esd(self, residuals, critical_values):
        indices, statistics = self._calc_statistics(residuals)

        test_length = len(statistics)
        max_idx = -1
        for i in range(test_length):
            if statistics[i] > critical_values[i]:
                max_idx = i

        anomaly_indices = np.asarray([])
        if max_idx > -1:
            anomaly_indices = np.asarray(indices[: max_idx + 1])

        return anomaly_indices

    def _calc_critical_values(self):
        # critical values are calculates as explained in:
        # https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h3.htm

        critical_values = []
        for i in range(1, self.k+1):
            f_degree = self.n - i - 1
            p = 1 - self.alpha / (2 * (self.n - i + 1))
            t_stat = stats.t.ppf(p, df=f_degree)

            numerator = (self.n - i) * t_stat
            denominator = np.sqrt((self.n - i - 1 + t_stat ** 2) * (self.n - i + 1))
            critical_value = numerator / denominator

            critical_values.append(critical_value)

        return critical_values

    def _calc_statistics(self, data):
        _data = data[:]

        indices = []
        statistics = []
        for i in range(1, self.k + 1):
            idx, statistic = self._calc_statistic(_data)
            statistics.append(statistic)
            indices.append(idx)

            _data = np.delete(_data, idx)

        return indices, statistics

    def _calc_statistic(self, data):
        if self.hybrid:
            median = np.median(data)
            mad = stats.median_abs_deviation(data)
            statistics = np.asarray(np.abs(data - median) / mad)
        else:
            mean = np.mean(data)
            std = np.std(data)
            statistics = np.asarray(np.abs(data - mean) / std)

        idx = np.argmax(statistics)
        statistic = np.max(statistics)

        return idx, statistic

# 基于xmr控制图的异常检测算法
class AnolmalyDetectionOnXmr(AnomalyDetection):
    def __init__(self):
        return

    def detect(self, **kwargs):
        data = convert_2_ts(kwargs['data'])
        if_plot = kwargs['if_plot']

        diff_flag = 0
        is_stationarity = True

        if 'is_stationarity' in kwargs.keys():
            is_stationarity = kwargs['is_stationarity']

        anomaly_per = 0  # 检测前异常点比例为0
        stationarity_result = self.test_stationarity(data, if_plot=if_plot)
        data_test = data.copy()
        while not (diff_flag == 1):
            if is_stationarity and ((stationarity_result > 0.05) or (anomaly_per >= 0.2)):  # 若不平稳 则进行差分
                data_test = data_test.diff()
            diff_flag = 1
            data_length = len(data_test)  # 在这里存一下data_test的长度
            anomalies = pd.Series(dtype=float)  # 存储最终结果的series
            data_anomaly = pd.Series(dtype=float)  # 存储过程中异常值的series

            times = 0  # 循环次数
            while not (data_anomaly.empty) or (times == 0):
                print('循环次数', times)
                # data_anomaly = data_anomaly[:0]
                # print(data_anomaly)
                data_anomaly = data_anomaly.drop(data_anomaly.index)  # 将存储异常的series清空

                times += 1
                # 开始检测异常
                # 控制线的确定
                x_mean = data_test.mean()
                mr = data_test.diff().abs()  # 这里需要取绝对值
                # mr.dropna(inplace=True)

                mr_mean = mr.mean()
                # cl_x = x_mean
                ucl_x = x_mean + 2.66 * mr_mean
                lcl_x = x_mean - 2.66 * mr_mean
                # cl_mr = mr_mean
                ucl_mr = 3.27 * mr_mean
                # lcl_mr = 0

                # 判断是否异常点
                list_index1 = []  # 更新异常的list
                list_value1 = []
                list_index2 = []  # 更新非异常的list
                list_value2 = []
                for index, value in data_test.items():
                    print('mr rows: {}'.format(mr.shape[0]))
                    print('index : {}'.format(index))
                    print('data_test: {}'.format(data_test.shape[0]))
                    print(mr)

                    if value < lcl_x:
                        list_index1.append(index)
                        list_value1.append(value)

                    elif value > ucl_x:
                        list_index1.append(index)
                        list_value1.append(value)

                    elif pd.notnull(mr[index]):
                        if mr[index] > ucl_mr:
                            list_index1.append(index)
                            list_value1.append(value)
                        else:
                            list_index2.append(index)
                            list_value2.append(value)

                    else:
                        list_index2.append(index)
                        list_value2.append(value)

                # todo
                if len(list_value1) > 1:
                    print(1)
                # series_index_anomaly = []
                # for idx in list_index1:
                #     series_index_anomaly.append(data.index[idx])

                data_anomaly = data_anomaly.append(pd.Series(data=list_value1, index=list_index1))

                # series_index_nomaly = []
                # for idx in list_index2:
                #     series_index_nomaly.append(data.index[idx])

                data_test = pd.Series(data=list_value2, index=list_index2)

                # old
                # data_anomaly = data_anomaly.append(pd.Series(data=list_value1, index=list_index1))
                # data_test = data_test.drop(data_test.index)  # 将存储待检测的series清空
                # data_test = pd.Series(data=list_value2, index=list_index2)

                anomalies = anomalies.append(data_anomaly)
            anomalies_length = len(anomalies)
            if data_length > 0:
                anomaly_per = anomalies_length / data_length
            else:
                anomaly_per = 0
            if (anomaly_per < 0.2 and stationarity_result < 0.5) or (np.isnan(stationarity_result)):
                diff_flag = 1

        return anomalies

    def get_range(self, data, if_plot=True):
        data = data.astype(float)
        anomalies = self.detect(data, if_plot=if_plot)
        data_range = data.copy()
        data_range = data_range.drop(index=anomalies.index)
        #print(data)
        data_range.dropna()
        # print(len(data_range))
        x_mean = data_range.mean()
        #print('x_mean', x_mean)
        mr = data_range.diff().abs()  # 这里需要取绝对值
        mr_mean = mr.mean()
        # cl_x = x_mean
        ucl_x = x_mean + 2.66 * mr_mean
        lcl_x = x_mean - 2.66 * mr_mean
        range_df = data.to_frame()
        #print(len(range_df), len(data), len(data_range))
        range_df.insert(range_df.shape[1], 'low', lcl_x)
        range_df.insert(range_df.shape[1], 'high', ucl_x)
        range_df.columns = ['value', 'low', 'high']
        print(len(range_df), len(data), len(data_range))
        return range_df

# 基于主成分分析的异常检测方法
class AnolmalyDetectionOnPCA(AnomalyDetection, BaseEstimator):
    """
    Detection of anomalies over Time series: time series anomaly
    detection using Robust Principal Component Pursuit

    Robust Principal Component Pursuit is a matrix decomposition algorithm
    that seeks to separate a matrix X into the sum of three parts

    X = L + S + E.
    L is a low rank matrix representing a smooth X, S is a sparse matrix
    containing corrupted data, and E is noise.
    To convert a time series into the matrix X we take advantage of seasonality
    so that each column represents one full period, for example for weekly
    seasonality each row is a day of week and one column is one full week.

    While computing the low rank matrix L we take an SVD of X and soft threshold
    the singular values.

    This approach allows us to dampen all anomalies across the board simultaneously
    making the method robust to multiple anomalies. Most techniques such as time series
    regression and moving averages are not robust when there are two or more anomalies present.

    Empirical tests show that identifying anomalies is easier if X is stationary.
    The Augmented Dickey Fuller Test is used to test for stationarity - if X is not
    stationary then the time series is differenced before calling RPCP. While this
    test is abstracted away from the user differencing can be forced by setting the
    forcediff parameter.

    The thresholding values can be tuned for different applications, however we strongly recommend
    using the defaults which were proposed by Zhou. For more details on the choice of
    Lpenalty and Spenalty please refer to Zhou's 2010 paper on Stable
    Principal Component Pursuit.

    Inspired by Surus Project Netflix:https://github.com/Netflix/Surus
    """

    def __init__(
            self,
            frequency: int = 7,
            autodiff: bool = True,
            forcediff: bool = False,
            scale: bool = True,
            lpenalty: float = 1.0,
            spenalty: float = -1.0,
            verbose: bool = False):

        self.frequency = frequency
        self.autodiff = autodiff
        self.forcediff = forcediff
        self.scale = scale
        self.lpenalty = lpenalty
        self.spenalty = spenalty
        self.verbose = verbose
        self.usediff = False
        self.global_mean = 0
        self.global_sdt = 1

    def fit(self, X):
        """
        Fit estimador

        Parameters

        X: 1d array-like

        Returns
        -------
        self: object
              Fitted estimador
        """
        self._fit(X)
        return self

    def _fit(self, X):
        """
        TODO:Documetation
        """

        if self.spenalty == -1:
            self._spenalty = 1.4 / math.sqrt(max(self.frequency, len(X) / self.frequency))

        if isinstance(X, pd.Series):
            X = X.values.copy()

        X = _check_representation(frequency=self.frequency, X=X)

        if self.forcediff:
            self.usediff = True
            X = np.nan_to_num(np.diff(X, prepend=0))

        elif self.autodiff:
            X, flag = _autodiff(X)
            self.usediff = flag

        self.global_mean, self.global_sdt, X = _scale(self.scale, X)

        M = _reshape_pad(X=X, frequency=self.frequency)

        if self.verbose:
            print("..........Start Process..........")
            print("Time Series, frequency=%d and Num Periods= %d." % (M.shape[0], M.shape[1]))
        try:
            self._L, self._S, self._E = rpca(M, lpenalty=self.lpenalty, spenalty=self.spenalty,
                                             verbose=self.verbose)
        except ValueError as e:
            print(e)
            self._L, self._S, self._E = None, None, None

    def fit_transform(self, X):
        """
        Dispatch to the rPCA Function
        """
        self.fit(X)
        L_transform = (self._L.T.reshape((-1, 1)).ravel() * self.global_sdt) + self.global_mean
        S_transform = (self._S.T.reshape((-1, 1)).ravel() * self.global_sdt)
        E_transform = (self._E.T.reshape((-1, 1)).ravel() * self.global_sdt)

        return L_transform, S_transform, E_transform

    def transform(self):
        """
        Return the Matrix L, S and E as a 1d arrays.
        """

        check_is_fitted(self, attributes=['_S', '_L', '_E'])

        L_transform = (self._L.T.reshape((-1, 1)).ravel() * self.global_sdt) + self.global_mean
        S_transform = (self._S.T.reshape((-1, 1)).ravel() * self.global_sdt)
        E_transform = (self._E.T.reshape((-1, 1)).ravel() * self.global_sdt)

        return L_transform, S_transform, E_transform

    def detect(self, data):
        """
        Return the Outliers after the rPCA transformation
        """
        data_ = convert_2_ts(data)
        self.fit(X=data_)
        check_is_fitted(self, attributes=['_S'])

        anomalies = pd.Series(dtype='float')

        if self._S is None:
            return anomalies

        S = self._S.T.reshape((-1, 1)).ravel()
        anomaly_indices = []
        for i in range(len(S)-len(data_), len(S)):
            if abs(S[i]) > 0.0:
                anomaly_indices.append(i-len(S)+len(data_))

        if not len(anomaly_indices) == 0:
            anomalies = data_[anomaly_indices]

        return anomalies

        # return np.abs(self._S.T.reshape((-1, 1)).ravel() * self.global_sdt)

    def decision_function(self):
        """
        Return the Outliers with label 0 or 1.
        0 : Normal Values
        1 : Outlier

        """
        check_is_fitted(self, attributes=['_S'])
        S = self._S.T.reshape((-1, 1)).copy()
        return _mad_outlier(X=S)

    def to_frame(self, X, add_mad=True):
        """
        Return DataFrame withe the values of the matrices

        X=L+S+E

        """

        check_is_fitted(self, attributes=['_S', '_L', '_E'])

        X_len = len(X)
        L_len = len(self._L.T.reshape((-1, 1)).ravel())

        length_diff = abs(L_len - X_len)
        if length_diff > 0:
            X = np.pad(array=X, pad_width=(length_diff, 0), mode='constant', constant_values=0)

        if self.usediff:
            X = np.nan_to_num(np.diff(X, prepend=0))

        L_transform = (self._L.T.reshape((-1, 1)).ravel() * self.global_sdt) + self.global_mean
        S_transform = (self._S.T.reshape((-1, 1)).ravel() * self.global_sdt)
        E_transform = (self._E.T.reshape((-1, 1)).ravel() * self.global_sdt)

        Output = pd.DataFrame({'X_original': X,
                               'L_transform': L_transform,
                               'S_transform': S_transform,
                               'E_transform': E_transform})

        if add_mad:
            S = self._S.T.reshape((-1, 1)).ravel()
            Output['Mad_Outliers'] = _mad_outlier(S)
            return Output
        else:
            return Output

    def num_outliers(self):
        """
        Number of Outliers
        """
        check_is_fitted(self, attributes=['_S'])
        S = self._S.T.reshape((-1, 1)).ravel()
        return sum(np.abs(S) > 0)

    def plot(self, figsize=(10, 6)):
        """
        Plot of the Time Series after rPCA transformation

        Parameters
        ----------

        figsize : Size of the plot

        Returns
        -------
        matplotlib plot
        """
        check_is_fitted(self, attributes=['_S', '_L', '_E'])

        L_transform = (self._L.T.reshape((-1, 1)).ravel() * self.global_sdt) + self.global_mean
        S_transform = (self._S.T.reshape((-1, 1)).ravel() * self.global_sdt)
        E_transform = (self._E.T.reshape((-1, 1)).ravel() * self.global_sdt)

        fig = plt.figure(facecolor='w', figsize=figsize)
        ax = fig.add_subplot(111)

        ax.plot(range(len(L_transform, )), L_transform + E_transform, label='Time Serie rPCA', ls='--', c='royalblue')
        ax.plot(range(len(L_transform)), np.abs(S_transform), c='red', label='Outliers')
        ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
        ax.legend()
        fig.tight_layout()
        return fig.show()

def handle_mbr_base_line(data, result_path, year_month, dim_cols):
    # 0值太多
    data_tmp = data.groupby(dim_cols)['value'].apply(
        lambda x: count_zeros(x.values)
    ).reset_index(name='is_zeros')
    data_tmp = pd.merge(data, data_tmp.query("is_zeros == 'No'"))

    # 空值太多
    data_tmp_1 = data.groupby(dim_cols)['value'].apply(
        lambda x: count_na(x.values)
    ).reset_index(name='is_na')
    data_tmp = pd.merge(data_tmp, data_tmp_1.query("is_na == 'No'"))

    # 众数不是异常
    result_mode_tmp = data_tmp.groupby(dim_cols + ['value']).size().reset_index()
    result_mode_tmp.columns = dim_cols + ['value', 'count']
    result_mode = result_mode_tmp.groupby(dim_cols)['count'].max().reset_index()
    result_mode.columns = dim_cols + ['max_count']
    result_mode = pd.merge(result_mode_tmp, result_mode)
    result_mode = result_mode.loc[result_mode.loc[:, 'count'] == result_mode.loc[:, 'max_count'], :]

    result_count = data_tmp.groupby(dim_cols).size().reset_index(name='all_count')
    result_mode = pd.merge(result_mode, result_count)
    result_mode.loc[:, 'ratio'] = result_mode.loc[:, 'max_count'] / result_mode.loc[:, 'all_count']
    result_mode = result_mode.query("ratio >= 0.1")
    result_mode = result_mode.loc[:, dim_cols + ['value', 'max_count', 'ratio']]

    # test
    anomaly_pca = AnolmalyDetectionOnPCA()

    if os.path.exists(result_path+'{}_pca.csv'.format(year_month)):
        data_ts_esd_pca = pd.read_csv(result_path+'{}_pca.csv'.format(year_month))
        data_ts_esd_pca.loc[:, 'Month'] = pd.to_datetime(data_ts_esd_pca.loc[:, 'Month'], format='%Y-%m-%d')
    else:
        data_ts_esd_pca = data_tmp.sort_values(by='Month').groupby(dim_cols).apply(
            lambda x: anomaly_pca.detect(data=pd.Series(x['value'].values, index=x['Month'].values))
        ).reset_index()

        data_ts_esd_pca.columns = dim_cols + ['Month', 'anomaly_pca']
        data_ts_esd_pca.to_csv(result_path + '{}_pca.csv'.format(year_month), encoding='utf_8_sig', index=False)

    # s-head-esd检测
    anomaly_esd = AnomalyDetectionOnSeasonalESD(anomaly_ratio=0.1, hybrid=True, alpha=0.05)
    anomaly_esd_xmr = AnolmalyDetectionOnXmr()
    #
    if os.path.exists(result_path+'{}_stl.csv'.format(year_month)):
        data_ts_esd_stl = pd.read_csv(result_path+'{}_stl.csv'.format(year_month))
        data_ts_esd_stl.loc[:, 'Month'] = pd.to_datetime(data_ts_esd_stl.loc[:, 'Month'], format='%Y-%m-%d')
    else:
        data_ts_esd_stl = data_tmp.sort_values(by='Month').groupby(dim_cols).apply(
            lambda x: anomaly_esd.detect(data=pd.Series(x['value'].values, index=x['Month'].values), if_plot=False,
                                         method='stl')
        ).reset_index()

        data_ts_esd_stl.columns = dim_cols + ['Month', 'anomaly_esd_stl']
        data_ts_esd_stl.to_csv(result_path+'{}_stl.csv'.format(year_month), encoding='utf_8_sig', index=False)

    if os.path.exists(result_path+'{}_xmr.csv'.format(year_month)):
        data_ts_esd_xmr = pd.read_csv(result_path+'{}_xmr.csv'.format(year_month))
        data_ts_esd_xmr.loc[:, 'Month'] = pd.to_datetime(data_ts_esd_xmr.loc[:, 'Month'], format='%Y-%m-%d')
    else:
        data_ts_esd_xmr = data_tmp.sort_values(by='Month').groupby(dim_cols).apply(
            lambda x: anomaly_esd_xmr.detect(data=pd.Series(x['value'].values, index=x['Month'].values), if_plot=False)
        ).reset_index()

        data_ts_esd_xmr.columns = dim_cols + ['Month', 'anomaly_xmr']
        data_ts_esd_xmr.to_csv(result_path+'{}_xmr.csv'.format(year_month), encoding='utf_8_sig', index=False)

    if os.path.exists(result_path+'{}_prophet.csv'.format(year_month)):
        data_ts_esd_prophet = pd.read_csv(result_path+'{}_prophet.csv'.format(year_month))
        data_ts_esd_prophet.loc[:, 'Month'] = pd.to_datetime(data_ts_esd_prophet.loc[:, 'Month'], format='%Y-%m-%d')
    else:
        data_ts_esd_prophet = data_tmp.sort_values(by='Month').groupby(dim_cols).apply(
            lambda x: anomaly_esd.detect(data=pd.Series(x['value'].values, index=x['Month'].values), if_plot=False,
                                         method='prophet')
        ).reset_index()

        data_ts_esd_prophet.columns = dim_cols + ['Month', 'anomaly_esd_prophet']
        data_ts_esd_prophet.to_csv(result_path+'{}_prophet.csv'.format(year_month), encoding='utf_8_sig', index=False)

    data_ts_esd_anomaly = pd.merge(data_ts_esd_prophet, data_ts_esd_stl, how='outer')
    data_ts_esd_anomaly = pd.merge(data_ts_esd_anomaly, data_ts_esd_xmr, how='outer')
    data_ts_esd_anomaly = pd.merge(data_ts_esd_anomaly, data_ts_esd_pca, how='outer')

    data_ts_esd_anomaly.loc[:, 'is_anomaly_prophet'] = data_ts_esd_anomaly.loc[:, 'anomaly_esd_prophet'].apply(
        lambda x: 0 if pd.isnull(x) else 1
    )
    data_ts_esd_anomaly.loc[:, 'is_anomaly_stl'] = data_ts_esd_anomaly.loc[:, 'anomaly_esd_stl'].apply(
        lambda x: 0 if pd.isnull(x) else 1
    )
    data_ts_esd_anomaly.loc[:, 'is_anomaly_xmr'] = data_ts_esd_anomaly.loc[:, 'anomaly_xmr'].apply(
        lambda x: 0 if pd.isnull(x) else 1
    )
    data_ts_esd_anomaly.loc[:, 'is_anomaly_pca'] = data_ts_esd_anomaly.loc[:, 'anomaly_pca'].apply(
        lambda x: 0 if pd.isnull(x) else 1
    )

    data_ts_esd_anomaly.loc[:, 'model_score'] = data_ts_esd_anomaly.loc[:,
                                                'is_anomaly_prophet'] + data_ts_esd_anomaly.loc[:,
                                                                        'is_anomaly_stl'] + data_ts_esd_anomaly.loc[:,
                                                                                            'is_anomaly_xmr'] + data_ts_esd_anomaly.loc[
                                                                                                                :,
                                                                                                                'is_anomaly_pca']

    # todo
    result_range = data_tmp.sort_values(by='Month').groupby(dim_cols).apply(
        lambda x: anomaly_esd.get_range_boxplot(pd.Series(x['value'].values, index=x['Month'].values), if_plot=False)
    ).reset_index()
    result_range.rename(columns={'ds': 'Month'}, inplace=True)

    output = pd.merge(result_range, data_ts_esd_anomaly, how='left')
    output = pd.merge(data_tmp, output, how='left')

    output.to_csv(result_path+'result_mbr{}.csv'.format(year_month), encoding='utf_8_sig', index=False)

    return


def handle_mbr_chanyan_line(result_path, month, data_file):
    # dateparse = lambda dates: pd.datetime.strptime(dates, '%Y%m')
    data = pd.read_csv(data_file, sep='	')
    data.loc[:, '时间（按月统计）'] = pd.to_datetime(data.loc[:, '时间（按月统计）'], format='%Y-%m')

    # data.loc[:, '指标辅助列'] = data.loc[:, '指标辅助列'].apply(
    #     lambda x: 'my_define' if pd.isnull(x) else x
    # )
    # data.loc[:, '口径说明'] = data.loc[:, '口径说明'].apply(
    #     lambda x: 'my_define' if pd.isnull(x) else x
    # )
    dim_cols = ['部门', '部门code', '部门层级', '是否末级节点', 'JobFamily', '职级']

    # 0值太多
    data_tmp = data.groupby(dim_cols)['离职率'].apply(
        lambda x: count_zeros(x.values)
    ).reset_index(name='is_zeros')
    print(data_tmp.shape[0])
    data_tmp = pd.merge(data, data_tmp.query("is_zeros == 'No'"))

    # 众数不是异常
    result_mode_tmp = data_tmp.groupby(dim_cols).size().reset_index()
    result_mode_tmp.columns = dim_cols + ['count']
    data_tmp = pd.merge(data_tmp, result_mode_tmp)

    print(data_tmp.shape[0])

    data_tmp = data_tmp.query("count > 12")
    print(data_tmp.shape[0])
    # test
    anomaly_pca = AnolmalyDetectionOnPCA()

    if os.path.exists(result_path + '{}_pca.csv'.format(month)):
        data_ts_esd_pca = pd.read_csv(result_path + '{}_pca.csv'.format(month))
        data_ts_esd_pca.loc[:, '时间（按月统计）'] = pd.to_datetime(data_ts_esd_pca.loc[:, '时间（按月统计）'], format='%Y-%m-%d')
    else:
        data_ts_esd_pca = data_tmp.sort_values(by='时间（按月统计）').groupby(dim_cols).apply(
            lambda x: anomaly_pca.detect(data=pd.Series(x['离职率'].values, index=x['时间（按月统计）'].values))
        ).reset_index()

        data_ts_esd_pca.columns = dim_cols + ['时间（按月统计）', 'anomaly_pca']
        data_ts_esd_pca.to_csv(result_path + '{}_pca.csv'.format(month), encoding='utf_8_sig', index=False)

    # s-head-esd检测
    anomaly_esd = AnomalyDetectionOnSeasonalESD(anomaly_ratio=0.1, hybrid=True, alpha=0.05)
    anomaly_esd_xmr = AnolmalyDetectionOnXmr()

    if os.path.exists(result_path + '{}_stl.csv'.format(month)):
        data_ts_esd_stl = pd.read_csv(result_path + '{}_stl.csv'.format(month))
        data_ts_esd_stl.loc[:, '时间（按月统计）'] = pd.to_datetime(data_ts_esd_stl.loc[:, '时间（按月统计）'], format='%Y-%m-%d')
    else:
        data_ts_esd_stl = data_tmp.sort_values(by='时间（按月统计）').groupby(dim_cols).apply(
            lambda x: anomaly_esd.detect(data=pd.Series(x['离职率'].values, index=x['时间（按月统计）'].values), if_plot=False,
                                         method='stl')
        ).reset_index()

        data_ts_esd_stl.columns = dim_cols + ['时间（按月统计）', 'anomaly_esd_stl']
        data_ts_esd_stl.to_csv(result_path + '{}_stl.csv'.format(month), encoding='utf_8_sig', index=False)

    if os.path.exists(result_path + '{}_xmr.csv'.format(month)):
        data_ts_esd_xmr = pd.read_csv(result_path + '{}_xmr.csv'.format(month))
        data_ts_esd_xmr.loc[:, '时间（按月统计）'] = pd.to_datetime(data_ts_esd_xmr.loc[:, '时间（按月统计）'], format='%Y-%m-%d')
    else:
        data_ts_esd_xmr = data_tmp.sort_values(by='时间（按月统计）').groupby(dim_cols).apply(
            lambda x: anomaly_esd_xmr.detect(data=pd.Series(x['离职率'].values, index=x['时间（按月统计）'].values), if_plot=False)
        ).reset_index()

        data_ts_esd_xmr.columns = dim_cols + ['时间（按月统计）', 'anomaly_xmr']
        data_ts_esd_xmr.to_csv(result_path + '{}_xmr.csv'.format(month), encoding='utf_8_sig', index=False)

    if os.path.exists(result_path + '{}_prophet.csv'.format(month)):
        data_ts_esd_prophet = pd.read_csv(result_path + '{}_prophet.csv'.format(month))
        data_ts_esd_prophet.loc[:, '时间（按月统计）'] = pd.to_datetime(data_ts_esd_prophet.loc[:, '时间（按月统计）'], format='%Y-%m-%d')
    else:
        data_ts_esd_prophet = data_tmp.sort_values(by='时间（按月统计）').groupby(dim_cols).apply(
            lambda x: anomaly_esd.detect(data=pd.Series(x['离职率'].values, index=x['时间（按月统计）'].values), if_plot=False,
                                         method='prophet')
        ).reset_index()

        data_ts_esd_prophet.columns = dim_cols + ['时间（按月统计）', 'anomaly_esd_prophet']
        data_ts_esd_prophet.to_csv(result_path + '{}_prophet.csv'.format(month), encoding='utf_8_sig', index=False)

    data_ts_esd_anomaly = pd.merge(data_ts_esd_prophet, data_ts_esd_stl, how='outer')
    data_ts_esd_anomaly = pd.merge(data_ts_esd_anomaly, data_ts_esd_xmr, how='outer')
    data_ts_esd_anomaly = pd.merge(data_ts_esd_anomaly, data_ts_esd_pca, how='outer')

    data_ts_esd_anomaly.loc[:, 'is_anomaly_prophet'] = data_ts_esd_anomaly.loc[:, 'anomaly_esd_prophet'].apply(
        lambda x: 0 if pd.isnull(x) else 1
    )
    data_ts_esd_anomaly.loc[:, 'is_anomaly_stl'] = data_ts_esd_anomaly.loc[:, 'anomaly_esd_stl'].apply(
        lambda x: 0 if pd.isnull(x) else 1
    )
    data_ts_esd_anomaly.loc[:, 'is_anomaly_xmr'] = data_ts_esd_anomaly.loc[:, 'anomaly_xmr'].apply(
        lambda x: 0 if pd.isnull(x) else 1
    )
    data_ts_esd_anomaly.loc[:, 'is_anomaly_pca'] = data_ts_esd_anomaly.loc[:, 'anomaly_pca'].apply(
        lambda x: 0 if pd.isnull(x) else 1
    )

    data_ts_esd_anomaly.loc[:, 'model_score'] = data_ts_esd_anomaly.loc[:,
                                                'is_anomaly_prophet'] + data_ts_esd_anomaly.loc[:,
                                                                        'is_anomaly_stl'] + data_ts_esd_anomaly.loc[:,
                                                                                            'is_anomaly_xmr'] + data_ts_esd_anomaly.loc[
                                                                                                                :,
                                                                                                                'is_anomaly_pca']

    # todo
    # result_range = data_tmp.sort_values(by='时间（按月统计）').groupby(dim_cols).apply(
    #     lambda x: anomaly_esd.get_range_boxplot(pd.Series(x['离职率'].values, index=x['Month'].values), if_plot=False)
    # ).reset_index()
    # result_range.rename(columns={'ds': '时间（按月统计）'}, inplace=True)

    # output = pd.merge(result_range, data_ts_esd_anomaly, how='left')
    # output = pd.merge(data_tmp, output, how='left')

    data_ts_esd_anomaly.to_csv(result_path + 'result_mbr{}.csv'.format(month), encoding='utf_8_sig', index=False)

    return

if __name__ == '__main__':
    # 修改时间
    month = '2024-07'
    data_file = 'D:/新建文件夹/anomaly_detection/data/experiment_data/original_data/MBR-202407.csv'

    # dim_cols = ['年月', '部门', '序列分类']
    dim_cols = ['指标名称', 'by BG/平台', 'by序列分类', '维度']
    index_col = '指标名称'
    value_col = 'value'
    order_col = '年月'

    data_long = process_data(data_file, dim_cols=dim_cols, value_col=value_col, order_col=order_col)

    result_path = 'D:/新建文件夹/anomaly_detection/result/'

    dim_cols = ['指标名称', 'by BG/平台', 'by序列分类', '维度']

    handle_mbr_base_line(data=data_long, result_path=result_path, year_month=month, dim_cols=dim_cols)

    # result_path = '/Users/xiefang09/xiefang/python/time_series-20230328/time_series/data/anomaly/'
    # result_file = result_path + 'result_mbr{}.csv'.format(month)
    # result = pd.read_csv(result_file)
    # dim_cols = ['部门', '序列分类', '指标名称']

    # result_path = '/Users/xiefang09/xiefang/python/time_series-20230328/time_series/data/anomaly/'
    # dim_cols = ['指标名称', 'by BG/平台', 'by序列分类']
    #
    # data_file = '/Users/xiefang09/xiefang/python/time_series-20230328/time_series/data/anomaly/input.txt'
    # handle_mbr_chanyan_line(result_path, "____", data_file)
































