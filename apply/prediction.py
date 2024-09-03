# coding: UTF-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fbprophet import Prophet

import datetime


from apply.anomaly_detaction_base import AnomalyDetection
from tools.holt_winters import HoltWinters

error = []
class PredictionBase(AnomalyDetection):
    def __init__(self):
        return object.__init__(self)

class TimeSeriesPrediction(PredictionBase):
    def __init__(self, test_num=3, p=12, format='%Y%m'):
        self.test_num = test_num
        self.p = p
        self.format = format
        self.holt_winters = HoltWinters(p=p, test_num=test_num)

        return PredictionBase.__init__(self)

    def set_test_num(self, test_num):
        self.test_num = test_num
        self.holt_winters.set_test_num(test_num)

    def set_cycle_period(self, p):
        self.p = p
        self.holt_winters.set_test_num(p)

    def train(self, data, if_plot=True):
        m = Prophet(yearly_seasonality=False,
                    # seasonality_prior_scale=1,
                    weekly_seasonality=False,
                    holidays_prior_scale=0.08,
                    growth='logistic',
                    changepoint_prior_scale=0.05)
        m.add_country_holidays(country_name='CN')

        df = pd.DataFrame(data)
        df = df.reset_index()
        df.columns = ['ds', 'y']

        mean_ = df.iloc[:-self.test_num, 1].mean()
        std = df.iloc[:-self.test_num, 1].std()
        cap = mean_ + std
        # floor = max(mean_ - std, 0)
        floor = mean_ - std

        df['cap'] = cap
        df['floor'] = floor
        m.fit(df.iloc[0: df.shape[0] - self.test_num, :])

        future = m.make_future_dataframe(periods=self.test_num + 1, freq='M', include_history=True)
        future['cap'] = cap
        future['floor'] = floor
        forecast = m.predict(future)

        df.drop(columns=['cap', 'floor'], inplace=True)
        future.drop(columns=['cap', 'floor'], inplace=True)

        resid = df.loc[:, 'y'] - forecast.loc[:, 'yhat']
        p_value_noisy = self.test_noisy(resid)
        p_value_stationarity = self.test_stationarity(resid, if_plot=False)

        forecast_resid = None

        # 如果不是白噪声，对残差利用holt_winters进行修正
        if p_value_noisy <= 0.05 and p_value_stationarity <= 0.05:
            resid_df = pd.DataFrame()
            resid_df.loc[:, 'ds'] = df.loc[:, 'ds']
            resid_df.loc[:, 'y'] = pd.Series(resid)

            self.holt_winters.grid_search(pd.Series(resid_df.loc[:, 'y'].values, index=resid_df.loc[:, 'ds'].values))

            predict_series = pd.Series(self.holt_winters.best_pred,
                                       index=resid_df.loc[:, 'ds'].values[resid.shape[0] - self.test_num:])
            all_resid = self.holt_winters.ts_train.append(predict_series)
            forecast_resid = pd.DataFrame(all_resid).reset_index()
            forecast_resid.columns = ['ds', 'pred']

        if forecast_resid is not None:
            forecast.loc[:, 'yhat'] = forecast.loc[:, 'yhat'] + forecast_resid.loc[:, 'pred']

        err = HoltWinters.compute_mse(df.loc[df.shape[0] - self.test_num:, 'y'],
                                      forecast.loc[forecast.shape[0] - self.test_num:, 'yhat'])
        error.append(err)

        if if_plot:
            m.plot(forecast)
            m.plot_components(forecast)
            self.plot(m, df, forecast)

        return forecast, error

    def plot(self, m, true_value, prediction_value, ahead=0):
        plot_cap = True
        figsize = (10, 6)
        fig = plt.figure(facecolor='w', figsize=figsize)
        ax = fig.add_subplot(111)
        fig = ax.get_figure()
        fcst_t = prediction_value['ds'].dt.to_pydatetime()
        ax.plot(true_value['ds'].dt.to_pydatetime(), true_value['y'], 'k.')
        ax.plot(fcst_t, prediction_value['yhat'], ls='-', c='#0072B2')
        ax.plot(fcst_t[-ahead:], prediction_value['yhat'][-ahead:], 'r.', markersize=10)
        if 'cap' in prediction_value and plot_cap:
            ax.plot(fcst_t, prediction_value['cap'], ls='--', c='k')
        if m.logistic_floor and 'floor' in prediction_value and plot_cap:
            ax.plot(fcst_t, prediction_value['floor'], ls='--', c='k')

        # Specify formatting to workaround matplotlib issue #12925
        from matplotlib.dates import AutoDateLocator, AutoDateFormatter
        locator = AutoDateLocator(interval_multiples=False)
        formatter = AutoDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
        xlabel = 'ds'
        ylabel = 'y'
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.tight_layout()

    def predict(self, data, ahead=2):
        self.train(data, if_plot=False)

        m = Prophet(yearly_seasonality=False,
                         # seasonality_prior_scale=1,
                         weekly_seasonality=False,
                         holidays_prior_scale=0.08,
                         growth='logistic',
                         changepoint_prior_scale=0.05)
        m.add_country_holidays(country_name='CN')

        df = pd.DataFrame(data)
        df = df.reset_index()
        df.columns = ['ds', 'y']

        mean_ = df.loc[:, 'y'].mean()
        std = df.loc[:, 'y'].std()
        cap = mean_ + std
        # floor = max(mean_ - std, 0)
        floor = mean_ - std

        df['cap'] = cap
        df['floor'] = floor
        m.fit(df)

        future = m.make_future_dataframe(periods=ahead+1, freq='M', include_history=True)
        future['cap'] = cap
        future['floor'] = floor
        forecast = m.predict(future)

        m.plot(forecast)
        m.plot_components(forecast)

        df.drop(columns=['cap', 'floor'], inplace=True)
        future.drop(columns=['cap', 'floor'], inplace=True)

        resid = df.loc[:, 'y'] - forecast.loc[0: forecast.shape[0]-ahead-1:, 'yhat']
        p_value_noisy = self.test_noisy(resid)
        p_value_stationarity = self.test_stationarity(resid, if_plot=False)

        forecast_resid = None

        # 如果不是白噪声，对残差进行一次修正预测
        if p_value_noisy <= 0.05 and p_value_stationarity <= 0.05:
            resid_df = pd.DataFrame()
            resid_df.loc[:, 'ds'] = df.loc[:, 'ds']
            resid_df.loc[:, 'y'] = pd.Series(resid)

            date_list = []

            now_date = df.loc[:, 'ds'].values[-1]
            time_stamp = (now_date - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
            date_time_stamp = datetime.datetime.fromtimestamp(time_stamp)
            date_time = date_time_stamp.strftime(self.format)

            for i in range(ahead):
                # year, month = int(date_time.split('-')[0]), int(date_time.split('-')[1])
                year, month = int(date_time[0:4]), int(date_time[-2:])
                if month == 12:
                    year += 1
                    month = 1
                else:
                    month += 1
                month_str = str(month)
                if month < 10:
                    month_str = '0'+str(month)
                if '-' in date_time:
                    date_time = '{}-{}'.format(year, month_str)
                else:
                    date_time = '{}{}'.format(year, month_str)
                date_list.append(pd.to_datetime(date_time, format=self.format))

            predict_series = pd.Series(self.holt_winters.predict_values(ahead*2)[-ahead:], index=pd.Series(date_list))
            all_resid = pd.Series(resid_df.loc[:, 'y'].values, index=resid_df.loc[:, 'ds'].values).append(predict_series)
            forecast_resid = pd.DataFrame(all_resid).reset_index()
            forecast_resid.columns = ['ds', 'pred']

        if forecast_resid is not None:
            forecast.loc[:, 'yhat'] = forecast.loc[:, 'yhat'] + forecast_resid.loc[:, 'pred']
            forecast.loc[:, 'ds'] = forecast_resid.loc[:, 'ds']

        self.plot(m, df, forecast, ahead)
        print(self.holt_winters.best_alpha)

        return forecast

    def calculate_precision(self, true, prediction):
        return ((prediction - true)/true).abs().sum()/len(true)

if __name__ == '__main__':
    predict_ = TimeSeriesPrediction()
    data = pd.read_csv('D:/code/time_series-20230112/time_series/data/prediction/quit_rate.csv')
    data.loc[:, 'Month'] = pd.to_datetime(data.loc[:, '月份'], format='%Y%m')
    data_ts = data.sort_values(by='Month').groupby(['序列']).apply(
        lambda x: predict_.predict(pd.Series(x['自愿离职率'].values, index=x['Month'].values))
    ).reset_index()

    # data_ts = data.sort_values(by='Month').groupby(['序列']).apply(
    #     lambda x: predict_.predict(pd.Series(x['自愿离职率'].values, index=x['Month'].values))
    # ).reset_index()

    for err in error:
        print('error: {}'.format(err))


