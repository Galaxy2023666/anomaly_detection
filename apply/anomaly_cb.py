import numpy
import matplotlib.pyplot as plt
import pandas as pd
import math
import os

from apply.clustering_ts import ClusteringTsOnKmeans


if __name__ == '__main__':
    cluster_file = '/Users/xiefang09/xiefang/python/time_series-20230328/time_series/data/cb/csv/cluster_{}.csv'
    cluster_figure_file = '/Users/xiefang09/xiefang/python/time_series-20230328/time_series/data/cb/png/cluster_figure_{}.png'
    cluster_std_file = '/Users/xiefang09/xiefang/python/time_series-20230328/time_series/data/cb/csv/cluster_std_{}.csv'
    data_file = '/Users/xiefang09/xiefang/python/time_series-20230328/time_series/data/cb/2023M07_TP.csv'

    is_update = True

    dim_cols = ['code', 'DEPTDESCR', 'JOB_FUNCTION_DESCR', 'JOB_SUB_FUNC_DESCR', 'SUPV_LVL_ID', 'LOCATION_DESCR',
                'LAST_HIRE_DT', 'NEWHIRE', 'HR_STATUS', 'SUBJECT', 'ID_HASH']


    if not os.path.exists(data_file) or is_update:
        # 定义排序和生成序号的函数
        def sort_and_rank(df_group, col_name):
            df_group_sorted = df_group.sort_values(by=col_name, ascending=False)
            df_group_sorted['order'] = range(1, len(df_group_sorted) + 1)
            return df_group_sorted

        data = pd.read_csv(data_file)
        # data.index = data.loc[:, 'code']
        # data.drop('code', inplace=True, axis=1)

        col_name = ['2021M08', '2021M09', '2021M10', '2021M11', '2021M12', '2022M01', '2022M02', '2022M03', '2022M04',
                    '2022M05', '2022M06', '2022M07', '2022M08', '2022M09', '2022M10', '2022M11', '2022M12', '2023M01',
                    '2023M02', '2023M03', '2023M04', '2023M05', '2023M06', '2023M07']

        col_name = dim_cols + col_name

        data_long = pd.melt(data.loc[:, col_name], id_vars=dim_cols, var_name="time",
                            value_name="value")

        subject_set = set(data_long.loc[:, 'SUBJECT'].values.tolist())

        for subject in subject_set:
            seed = 0
            numpy.random.seed(seed)
            n_clusters = 7
            metric = 'dtw'
            cluster = ClusteringTsOnKmeans(seed=seed, n_clusters=n_clusters)

            cluster_result = cluster.predict(data_long.query("SUBJECT == '{}'".format(subject)), dim_cols, 'value',
                                             False, 24, figure_file=cluster_figure_file.format(subject))
            cluster_result.to_csv(cluster_file.format(subject), encoding='utf_8_sig', index=False)

            dim_group_cols = ['JOB_SUB_FUNC_DESCR', 'SUPV_LVL_ID']

            cluster_index_count = \
                cluster_result.loc[:, dim_group_cols + ['cluster_id']].groupby(
                    dim_group_cols + ['cluster_id'])['cluster_id'].count()
            cluster_index_count = cluster_index_count.reset_index(name='count')

            cluster_index_count_all = \
                cluster_result.loc[:, dim_group_cols + ['cluster_id']].groupby(
                    dim_group_cols)['cluster_id'].count()
            cluster_index_count_all = cluster_index_count_all.reset_index(name='count_all')

            cluster_index_count = pd.merge(cluster_index_count, cluster_index_count_all)
            cluster_index_count.loc[:, 'ratio'] = cluster_index_count.loc[:, 'count'] / cluster_index_count.loc[:,
                                                                                        'count_all']

            # 按group列进行分组，并根据value列进行排序并为每个组生成序号
            # result = cluster_index_count.groupby(dim_group_cols).apply(
            #     lambda t: sort_and_rank(t, col_name='ratio')
            # )
            # result.index = pd.Series(range(result.shape[0]))

            # 探索指标聚类分散度
            cluster_index_std = cluster_index_count.groupby(dim_group_cols)['count'].apply(
                lambda x: x.std()
            ).reset_index(name='std')

            result = pd.merge(cluster_index_count, cluster_index_std, on=dim_group_cols)

            result.to_csv(cluster_std_file.format(subject), encoding='utf_8_sig', index=False)





