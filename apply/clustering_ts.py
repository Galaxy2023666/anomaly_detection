# coding: UTF-8

from tslearn.clustering import TimeSeriesKMeans

import numpy
import matplotlib.pyplot as plt
import pandas as pd
import math
import os

from tslearn.clustering import TimeSeriesKMeans

#处理数据 删除没有信息的样本
from apply.ts_apply import TsApplyBase


from tools.statistics_calculation import *
from tools.preprocess import convert_to_zeros


class ClusteringTsOnKmeans(TsApplyBase):
    def __init__(self, seed, n_clusters, metric='euclidean'):
        self.km = TimeSeriesKMeans(n_clusters=n_clusters, metric=metric, verbose=True, random_state=seed)
        self.metric = metric

        self.n_clusters = n_clusters

        TsApplyBase.__init__(self)

        return

    def predict(self, data, group_cols, value_name, is_norm, length, figure_file=None):
        data_list = []

        if is_norm:
            data.loc[:, value_name + '_norm'] = data.groupby(group_cols)[value_name].apply(
                lambda x: (x - x.mean()) / x.std()
            )
            data_group = data.loc[:, group_cols + [value_name + '_norm']].groupby(group_cols)[value_name + '_norm']
        else:
            data_group = data.groupby(group_cols)[value_name]

        key_list = []

        for key, value in data_group:
            if len(value) != length:
                continue
            if math.isnan(value.tolist()[0]):
                continue
            data_list.append(value.tolist())
            key_list.append(key)

        data_array = numpy.asarray(data_list)
        sz = data_array.shape[1]

        y_pred = self.km.fit_predict(data_array)

        plt.figure()
        for yi in range(self.n_clusters):
            plt.subplot(1, self.n_clusters, yi + 1)
            for xx in data_array[y_pred == yi]:
                plt.plot(xx.ravel(), "k-", alpha=.2)
            plt.plot(self.km.cluster_centers_[yi].ravel(), "r-")
            plt.xlim(0, sz)
            plt.ylim(-4, 4)
            plt.text(0.55, 0.85, 'Cluster %d' % (yi + 1), transform=plt.gca().transAxes)
            if yi == 1:
                plt.title(self.metric)
        if figure_file:
            plt.savefig(figure_file)
        else:
            plt.show()

        y_pred_list = y_pred.tolist()
        result = []
        for key, value in zip(key_list, y_pred_list):
            result.append(list(key) + [value])

        result_df = pd.DataFrame(result)
        result_df.columns = group_cols + ['cluster_id']

        # result_df = pd.merge(data, result_df)

        # result_df.to_csv('/Users/xiefang09/xiefang/python/time_series-20230328/time_series/data/anomaly/cluster.csv',
        #                  encoding='utf_8_sig', index=False)

        return result_df


if __name__ == '__main__':
    #修改
    month = '2024-07'
    cluster_file = 'D:/新建文件夹/anomaly_detection/result/cluster2024-07.csv'
    cluster_std_file = 'D:/新建文件夹/anomaly_detection/result/cluster_std2024-07.csv'
    cluster_index_count_file = 'D:/新建文件夹/anomaly_detection/result/cluster_index_count2024-07.csv'
    anomaly_file = 'D:/新建文件夹/anomaly_detection/result/result_mbr2024-07.csv'
    data_file = 'D:/新建文件夹/anomaly_detection/data/experiment_data/original_data/MBR-202407.csv'

    is_update = True

    dim_cols = ['指标名称', 'by BG/平台', 'by序列分类', '维度']
    if not os.path.exists(data_file) or is_update:
        data = pd.read_csv(data_file)
        # dim_cols = ['指标名称', 'by BG/平台', 'by序列分类']

        data.loc[:, '维度'] = data.loc[:, '维度'].apply(
            lambda x: 'my_define' if pd.isnull(x) else x
        )

        # col_name = ['2019-01', '2019-02', '2019-03', '2019-04', '2019-05', '2019-06', '2019-07', '2019-08', '2019-09',
        #             '2019-10', '2019-11', '2019-12', '2020-01', '2020-02', '2020-03', '2020-04', '2020-05', '2020-06',
        #             '2020-07', '2020-08', '2020-09', '2020-10', '2020-11', '2020-12', '2021-01', '2021-02', '2021-03',
        #             '2021-04', '2021-05', '2021-06', '2021-07', '2021-08', '2021-09', '2021-10', '2021-11', '2021-12',
        #             '2022-01', '2022-02', '2022-03', '2022-04', '2022-05', '2022-06', '2022-07', '2022-08', '2022-09',
        #             '2022-10', '2022-11', '2022-12', '2023-01', '2023-02', '2023-03', '2023-04', '2023-05', '2023-06']
        # if month not in col_name:
        #     col_name.append(month)
        #
        # col_name = dim_cols + col_name

        # data_long = pd.melt(data.loc[:, col_name], id_vars=dim_cols, var_name="time", value_name="value")
        data_long = pd.melt(data, id_vars=dim_cols, var_name="年月", value_name="value")
        data_long.loc[:, 'Month'] = pd.to_datetime(data_long.loc[:, '年月'], format='%Y-%m')
        data_long.drop('年月', axis=1, inplace=True)

        data_long.dropna(inplace=True)
        data_long.loc[:, 'value'] = data_long.loc[:, 'value'].apply(lambda x: replace_blank(x))
        data_long = data_long.query("value!='-' and value!='/' and value!=''")
        data_long_tmp = data_long.query("Month >= '2021-06-01 00:00:00'")

        data_long_sum = data_long_tmp.groupby(['指标名称', 'by BG/平台', 'by序列分类'])['value'].count().reset_index(name='count')
        data_long_sum = data_long_sum.query("count >= 25")

        data_long = pd.merge(data_long, data_long_sum)
        data_long.loc[:, 'value'] = data_long.loc[:, 'value'].apply(convert_to_zeros)

        seed = 0
        numpy.random.seed(seed)
        n_clusters = 7
        metric = 'dtw'
        cluster = ClusteringTsOnKmeans(seed=seed, n_clusters=n_clusters)
        # todo 需要修改length
        length = 67
        cluster_result = cluster.predict(data_long, dim_cols, 'value', True, length,
                                         'D:/新建文件夹/anomaly_detection/result/cluster202407.png')
        cluster_result.to_csv(cluster_file, encoding='utf_8_sig', index=False)
    else:
        cluster_result = pd.read_csv(cluster_file)

    cluster_index_count = \
    cluster_result.loc[:, dim_cols + ['cluster_id']].drop_duplicates().loc[:, ['指标名称', 'by序列分类', 'cluster_id']].groupby(
        ['指标名称', 'by序列分类', 'cluster_id'])['cluster_id'].count()
    cluster_index_count = cluster_index_count.reset_index(name='count')

    cluster_index_count_all = \
    cluster_result.loc[:, dim_cols + ['cluster_id']].drop_duplicates().loc[:, ['指标名称', 'by序列分类', 'cluster_id']].groupby(
        ['指标名称', 'by序列分类'])['cluster_id'].count()
    cluster_index_count_all = cluster_index_count_all.reset_index(name='count_all')

    cluster_index_count = pd.merge(cluster_index_count, cluster_index_count_all)
    cluster_index_count.loc[:, 'ratio'] = cluster_index_count.loc[:, 'count'] / cluster_index_count.loc[:, 'count_all']
    cluster_index_count.to_csv(cluster_index_count_file, encoding='utf_8_sig', index=False)


    # 定义排序和生成序号的函数
    def sort_and_rank(df_group, col_name):
        df_group_sorted = df_group.sort_values(by=col_name, ascending=False)
        df_group_sorted['order'] = range(1, len(df_group_sorted) + 1)
        return df_group_sorted


    # 按group列进行分组，并根据value列进行排序并为每个组生成序号
    result = cluster_index_count.groupby(['指标名称', 'by序列分类']).apply(
        lambda t: sort_and_rank(t, col_name='ratio')
    )
    result.index = pd.Series(range(result.shape[0]))

    # 探索指标聚类分散度
    cluster_index_std = cluster_index_count.groupby(['指标名称'])['count'].apply(
        lambda x: x.std()
    ).reset_index(name='std')

    result = pd.merge(result, cluster_index_std, on=['指标名称'])

    result.to_csv(cluster_std_file, encoding='utf_8_sig', index=False)


    # cluster_index_std.to_csv(cluster_std_file, encoding='utf_8_sig', index=False)

    def yingshe_b(x):
        if int(x[:2]) == 0:
            return '美团'
        if int(x[:2]) <= 9:
            return '到店事业群'
        if int(x[:2]) <= 17:
            return '到家事业群'
        if int(x[:2]) == 18:
            return '优选事业部'
        if int(x[:2]) == 19:
            return '快驴事业部'
        if int(x[:2]) == 20:
            return '小象事业部'
        if int(x[:2]) == 21:
            return '骑行事业部'
        if int(x[:2]) == 22:
            return 'SaaS事业部'
        if int(x[:2]) == 23:
            return '充电宝业务部'
        if int(x[:2]) == 24:
            return '无人机业务部'
        if int(x[:2]) == 25:
            return '自动车配送部'
        if int(x[:2]) == 26:
            return '境外业务'
        if int(x[:2]) == 27:
            return '金融服务平台'
        if int(x[:2]) == 28:
            return '点评事业部'
        if int(x[:2]) <= 32:
            return '美团平台'
        if int(x[:2]) == 33:
            return '基础研发平台'
        if int(x[:2]) == 34:
            return '公司事务平台'
        if int(x[:2]) == 35:
            return '人力资源'
        if int(x[:2]) == 36:
            return '财务平台'
        if int(x[:2]) == 37:
            return '战略与投资平台'
        if int(x[:2]) == 38:
            return '总办'
        if int(x[:2]) == 39:
            return 'GN06'


    cluster = pd.read_csv(cluster_file)
    # cluster_std = pd.read_csv(cluster_std_file)
    # cluster_count = pd.read_csv(cluster_index_count_file)
    anormaly = pd.read_csv(anomaly_file)
    # df = pd.read_excel('./指标信息映射.xlsx')

    # 异动模型结果有重复值，需要先去重
    anormaly = anormaly[~anormaly['指标名称'].isna()]
    anormaly = anormaly.drop_duplicates()

    year_mo = ['2019-01', '2019-02', '2019-03',
               '2019-04', '2019-05', '2019-06', '2019-07', '2019-08', '2019-09',
               '2019-10', '2019-11', '2019-12', '2020-01', '2020-02', '2020-03',
               '2020-04', '2020-05', '2020-06', '2020-07', '2020-08', '2020-09',
               '2020-10', '2020-11', '2020-12', '2021-01', '2021-02', '2021-03',
               '2021-04', '2021-05', '2021-06', '2021-07', '2021-08', '2021-09',
               '2021-10', '2021-11', '2021-12', '2022-01', '2022-02', '2022-03',
               '2022-04', '2022-05', '2022-06', '2022-07', '2022-08', '2022-09',
               '2022-10', '2022-11', '2022-12', '2023-01', '2023-02', '2023-03',
               '2023-04', '2023-05', '2023-06', '2023-07', '2023-08', '2023-09',
               '2023-10', '2023-11', '2023-12', '2024-01', '2024-02', '2024-03',
               '2024-04', '2024-05', '2024-06', '2024-07']

    last_last_month = anormaly[anormaly['Month'] == year_mo[-3] + '-01'].copy()
    last_month = anormaly[anormaly['Month'] == year_mo[-2] + '-01'].copy()
    this_month = anormaly[anormaly['Month'] == year_mo[-1] + '-01'].copy()

    last_last_month['是否异动1'] = last_last_month['model_score'].apply(lambda x: 1 if x > 0 else 0)
    last_month['是否异动1'] = last_month['model_score'].apply(lambda x: 1 if x > 0 else 0)
    this_month['是否异动1'] = this_month['model_score'].apply(lambda x: 1 if x > 0 else 0)

    last_last_month['是否异动'] = last_last_month['model_score'].apply(lambda x: '是' if x > 0 else '否')
    last_month['是否异动'] = last_month['model_score'].apply(lambda x: '是' if x > 0 else '否')
    this_month['是否异动'] = this_month['model_score'].apply(lambda x: '是' if x > 0 else '否')

    last_last_month_result = last_last_month.groupby(['指标名称']).agg({'是否异动1': 'sum', '维度': 'count'}).reset_index()
    last_last_month_result['异动占比'] = last_last_month_result['是否异动1'] / last_last_month_result['维度']

    last_month_result = last_month.groupby(['指标名称']).agg({'是否异动1': 'sum', '维度': 'count'}).reset_index()
    last_month_result['异动占比'] = last_month_result['是否异动1'] / last_month_result['维度']

    this_month_result = this_month.groupby(['指标名称']).agg({'是否异动1': 'sum', '维度': 'count'}).reset_index()
    this_month_result['异动占比'] = this_month_result['是否异动1'] / this_month_result['维度']

    # 整合后进异动模型的表3
    df_babp = pd.read_csv(data_file)

    cols = ['指标名称', 'by BG/平台', 'by序列分类', '维度', '2022-01',
            '2022-02', '2022-03', '2022-04', '2022-05', '2022-06', '2022-07',
            '2022-08', '2022-09', '2022-10', '2022-11', '2022-12', '2023-01',
            '2023-02', '2023-03', '2023-04', '2023-05', '2023-06', '2023-07',
            '2023-08', '2023-09', '2023-10', '2023-11', '2023-12', '2024-01',
            '2024-02', '2024-03', '2024-04', '2024-05', '2024-06', '2024-07'
    ]
    df_babp = df_babp[df_babp['by BG/平台'] != '未知']

    df_babp['year_mo'] = '2024_07'  # 这一行每个月需要更改一下

    df_babp['source'] = ''

    cols = ['year_mo', '指标名称', 'by BG/平台', 'by序列分类', '维度', 'source', '2022-01',
            '2022-02', '2022-03', '2022-04', '2022-05', '2022-06', '2022-07',
            '2022-08', '2022-09', '2022-10', '2022-11', '2022-12', '2023-01',
            '2023-02', '2023-03', '2023-04', '2023-05', '2023-06', '2023-07',
            '2023-08', '2023-09', '2023-10', '2023-11', '2023-12', '2024-01',
            '2024-02', '2024-03', '2024-04', '2024-05', '2024-06', '2024-07'
            ]

    df_babp = df_babp[cols]

    # df_babp = pd.merge(left=df_babp,right =last_month,how = 'left'
    #                    ,left_on=['指标名称_y','by BG/平台','by序列分类']
    #                    ,right_on=['指标名称_y','by BG/平台','by序列分类'],suffixes = ('','02') )
    anomaly_cols = ['指标名称', 'by BG/平台', 'by序列分类', '是否异动1', 'model_score']

    # 拼接这个月是否异动的信息
    df_babp = pd.merge(left=df_babp, right=this_month[anomaly_cols], how='left'
                       , left_on=['指标名称', 'by BG/平台', 'by序列分类']
                       , right_on=['指标名称', 'by BG/平台', 'by序列分类'], suffixes=('', '_thismonth'))

    # 生成上传表中的一个字段
    df_babp = df_babp.loc[df_babp.loc[:, 'by BG/平台'] != '-', :]
    df_babp['b_02_dept_name'] = df_babp['by BG/平台'].apply(lambda x: yingshe_b(x))

    # 拼接指标的聚类信息
    df_babp = pd.merge(left=df_babp, right=cluster, how='left'
                       , left_on=['指标名称', 'by BG/平台', 'by序列分类']
                       , right_on=['指标名称', 'by BG/平台', 'by序列分类'], suffixes=('', '_thismonth_cluster'))

    # 拼接指标的方差信息
    df_babp = pd.merge(left=df_babp, right=cluster_index_std, how='left'
                       , left_on=['指标名称']
                       , right_on=['指标名称'], suffixes=('', '_thismonth_clusterstd'))

    # df_babp = pd.merge(left =df_babp
    #                    ,right = last_last_month_result,how = 'left'
    #                    ,left_on=['指标名称']
    #                    ,right_on=['指标名称'],suffixes = ('','02'))

    # 拼接上个月异动占比
    df_babp = pd.merge(left=df_babp
                       , right=last_month_result, how='left'
                       , left_on=['指标名称']
                       , right_on=['指标名称'], suffixes=('', 'last_month_rate'))

    # 拼接这个月异动占比
    df_babp = pd.merge(left=df_babp
                       , right=this_month_result, how='left'
                       , left_on=['指标名称']
                       , right_on=['指标名称'], suffixes=('', 'this_month_rate'))

    # 拼接上上月异动占比
    df_babp = pd.merge(left=df_babp
                       , right=last_last_month_result, how='left'
                       , left_on=['指标名称']
                       , right_on=['指标名称'], suffixes=('', 'last_last_month_rate'))

    # df_babp = pd.merge(left=df_babp,right =last_month,how = 'left'
    #                    ,left_on=['指标名称_y','by BG/平台','by序列分类']
    #                    ,right_on=['指标名称_y','by BG/平台','by序列分类'],suffixes = ('','02') )
    anomaly_cols = ['指标名称', 'by BG/平台', 'by序列分类', '是否异动', 'model_score']

    # 拼接这个月是否异动的信息
    df_babp = pd.merge(left=df_babp, right=this_month[anomaly_cols], how='left'
                       , left_on=['指标名称', 'by BG/平台', 'by序列分类']
                       , right_on=['指标名称', 'by BG/平台', 'by序列分类'], suffixes=('', '_thismonth'))

    # 拼接上个月是否异动的信息
    df_babp = pd.merge(left=df_babp, right=last_month[anomaly_cols], how='left'
                       , left_on=['指标名称', 'by BG/平台', 'by序列分类']
                       , right_on=['指标名称', 'by BG/平台', 'by序列分类'], suffixes=('', '_lastmonth'))

    # 拼接上上个月是否异动的信息
    df_babp = pd.merge(left=df_babp, right=last_last_month[anomaly_cols], how='left'
                       , left_on=['指标名称', 'by BG/平台', 'by序列分类']
                       , right_on=['指标名称', 'by BG/平台', 'by序列分类'], suffixes=('', '_lastlastmonth'))

    # 计算是否连续三个月异动
    df_babp['是否连续三个月异动'] = df_babp[['是否异动', '是否异动_lastmonth', '是否异动_lastlastmonth']].apply(
        lambda x: '是' if x[0] == '是' and x[1] == '是' and x[2] == '是' else '否', axis=1)

    # 计算是否连续两个月异动
    df_babp['是否连续两个月异动'] = df_babp[['是否异动', '是否异动_lastmonth']].apply(
        lambda x: '是' if x[0] == '是' and x[1] == '是' else '否', axis=1)

    col_check = ['year_mo', 'zhibiao', 'bg', 'xulie', 'dim', 'source', 'last_year_month_01', 'last_year_month_02',
                 'last_year_month_03',
                 'last_year_month_04', 'last_year_month_05', 'last_year_month_06', 'last_year_month_07',
                 'last_year_month_08', 'last_year_month_09',
                 'last_year_month_10', 'last_year_month_11', 'last_year_month_12', 'this_year_month_01',
                 'this_year_month_02', 'this_year_month_03',
                 'this_year_month_04', 'this_year_month_05', 'this_year_month_06', 'this_year_month_07',
                 'this_year_month_08', 'this_year_month_09',
                 'this_year_month_10', 'this_year_month_11', 'this_year_month_12', 'mom', 'yoy01', 'yoy02', 'yoy03',
                 'yoy04', 'yoy05', 'yoy06', 'yoy07', 'yoy08', 'yoy09', 'yoy10', 'yoy11',
                 'yoy12', 'is_abnormal', 'model_score', 'b_02_dept_name', 'cluster_id', 'std', 'last_m_abnor_cnt',
                 'last_m_cnt', 'last_m_abnor_rt', 'this_m_abnor_cnt',
                 'this_m_cnt', 'this_m_abnor_rt', 'last_last_m_abnor_cnt', 'last_last_m_cnt', 'last_last_m_abnor_rt',
                 'this_m_is_abnormal', 'this_m_abnormal_score',
                 'last_m_is_abnormal', 'last_m_abnormal_score', 'last_last_m_is_abnormal', 'last_last_m_abnormal_score',
                 'if_three_month_abnormal', 'if_two_month_abnormal']

    df_babp = df_babp.rename({'指标名称': 'zhibiao', "by BG/平台": 'bg', 'by序列分类': 'xulie', '维度': 'dim',
                              '2023-01': 'last_year_month_01',
                              '2023-02': 'last_year_month_02',
                              '2023-03': 'last_year_month_03',
                              '2023-04': 'last_year_month_04',
                              '2023-05': 'last_year_month_05',
                              '2023-06': 'last_year_month_06',
                              '2023-07': 'last_year_month_07',
                              '2023-08': 'last_year_month_08',
                              '2023-09': 'last_year_month_09',
                              '2023-10': 'last_year_month_10',
                              '2023-11': 'last_year_month_11',
                              '2023-12': 'last_year_month_12',
                              '2024-01': 'this_year_month_01',
                              '2024-02': 'this_year_month_02',
                              '2024-03': 'this_year_month_03',
                              '2024-04': 'this_year_month_04',
                              '2024-05': 'this_year_month_05',
                              '2024-06': 'this_year_month_06',
                              '2024-07': 'this_year_month_07',
                              '2024-08': 'this_year_month_08',
                              '2024-09': 'this_year_month_09',
                              '2024-10': 'this_year_month_10',
                              '2024-11': 'this_year_month_11',
                              '2024-12': 'this_year_month_12',
                              '是否异动1': 'is_abnormal',
                              'model_score': 'model_score',
                              'b_02_dept_name': 'b_02_dept_name',
                              'cluster_id': 'cluster_id',
                              'std': 'std',
                              '是否异动1last_month_rate': 'last_m_abnor_cnt',
                              '维度last_month_rate': 'last_m_cnt',
                              '异动占比': 'last_m_abnor_rt',
                              '是否异动1this_month_rate': 'this_m_abnor_cnt',
                              '维度this_month_rate': 'this_m_cnt',
                              '异动占比this_month_rate': 'this_m_abnor_rt',
                              '是否异动1last_last_month_rate': 'last_last_m_abnor_cnt',
                              '维度last_last_month_rate': 'last_last_m_cnt',
                              '异动占比last_last_month_rate': 'last_last_m_abnor_rt',
                              '是否异动': 'this_m_is_abnormal',
                              'model_score_thismonth': 'this_m_abnormal_score',
                              '是否异动_lastmonth': 'last_m_is_abnormal',
                              'model_score_lastmonth': 'last_m_abnormal_score',
                              '是否异动_lastlastmonth': 'last_last_m_is_abnormal',
                              'model_score_lastlastmonth': 'last_last_m_abnormal_score',
                              '是否连续三个月异动': 'if_three_month_abnormal',
                              '是否连续两个月异动': 'if_two_month_abnormal', }, axis=1)
    for i in col_check:
        if i not in df_babp.columns:
            df_babp[i] = ''
    df_babp = df_babp[col_check]

    # df_babp.to_excel('./BABP.xlsx') 改动日期
    # df_babp.to_csv('D:/新建文件夹/anomaly_detection/result/202406BABP-V4.csv',
    #                encoding='utf_8_sig', index=False, sep='\t')
    df_babp.to_excel('D:/新建文件夹/anomaly_detection/result/202407BABP-V4.xlsx',
                     encoding='utf_8_sig', index=False)
    print(df_babp.columns)












