
import numpy as np
import pandas as pd

data_path='D:/新建文件夹/活性/HR_data.csv'
df = pd.read_csv(data_path,encoding='utf-8-sig')
month_list = ['2023/6/30','2023/7/31','2023/8/31','2023/9/30','2023/10/31','2023/11/30','2023/12/31',
              '2024/1/31','2024/2/29','2024/3/31','2024/4/30','2024/5/31','2024/6/30']
df_months = pd.DataFrame({'month': month_list})
dimensions = df.iloc[:, :6]
df_combined = dimensions.assign(key=1).merge(df_months.assign(key=1), on='key').drop('key', axis=1)
data_input=df.iloc[:,6:]
length=len(data_input)
df_combined['value']=np.nan
df_combined['is_anomaly']=np.nan
df_combined.to_csv('D:/新建文件夹/活性/HR_prediction.csv', encoding='utf-8-sig', index=False)


def convert_2_ts(series_):
    df = series_.to_frame().reset_index()
    df.columns = ['ds', 'value']
    df.index = df.loc[:, 'ds']

    df.drop('ds', axis=1, inplace=True)

    ds = df.loc[:, 'value']

    return ds

class AnolmalyDetectionOnXmr:
    def __init__(self):
        return

    def detect(self, data):
        data = convert_2_ts(data)

        anomaly_per = 0  # 检测前异常点比例为0
        data_test = data.copy()
        data_test = data_test.diff()
        data_length = len(data_test)  # 在这里存一下data_test的长度
        anomalies = pd.Series(dtype='float64')  # 存储最终结果的series
        data_anomaly = pd.Series(dtype='float64')  # 存储过程中异常值的series

        times = 0  # 循环次数
        while not (data_anomaly.empty) or (times == 0):
            # print('循环次数', times)
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
                # print('mr rows: {}'.format(mr.shape[0]))
                # print('index : {}'.format(index))
                # print('data_test: {}'.format(data_test.shape[0]))
                # print(mr)

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

        return anomalies

    def get_range(self, data):
        data = data.astype(float)
        anomalies = self.detect(data)
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
        # print(len(range_df), len(data), len(data_range))
        return range_df


for i in range(length):
    data=data_input.loc[i]
    detector=AnolmalyDetectionOnXmr()
    anomalies = detector.detect(data)
    print("检测到的异常点:")
    print(anomalies)
    # 调用 get_range 方法
    range_df = detector.get_range(data)
    # print("数据范围 (带上下限):")
    print(range_df)

    df = pd.read_csv('D:/新建文件夹/活性/HR_prediction.csv',encoding='utf-8-sig')
    empty_rows = df[df['value'].isnull()].index[:13]
    print(len(empty_rows))  # 打印 empty_rows 的长度
    print(len(range_df['value'].values))  # 打印 range_df['value'].values 的长度
    df.loc[empty_rows, 'value'] = range_df['value'].values
    if not anomalies.empty:
        matching_rows = df.loc[empty_rows][df['month'].isin(anomalies.index)]
        df.loc[matching_rows.index, 'is_anomaly'] = 1
    # 4. 重新保存CSV文件
    df.to_csv('D:/新建文件夹/活性/HR_prediction.csv', encoding='utf-8-sig', index=False)

