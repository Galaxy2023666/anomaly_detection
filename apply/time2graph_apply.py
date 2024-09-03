import math
import os

import pandas
import pandas as pd
import numpy as np
import torch
import networkx as nx

# from gensim.models import Word2Vec
from sklearn.cluster import AgglomerativeClustering
from tslearn.clustering import TimeSeriesKMeans

import argparse
from pathos.helpers import mp

import ruptures as rpt

# from tools.change_point_detection.change_point_finder import Qdetector

from tools.shapelets_base.time2graph_plus.time2graph.core.model_embeds import Time2GraphEmbed
from tools.shapelets_base.time2graph_plus.time2graph.core.model_gat import Time2GraphGAT
from tools.shapelets_base.time2graph_plus.config import module_path, STB, model_args, xgb_args, njobs

from tools.shapelets_base.time2graph_plus.time2graph.utils.base_utils import Debugger, evaluate_performance
from tools.shapelets_base.time2graph_plus.config import module_path
# from tools.shapelets_base.time2graph_plus.archive.load_usr_dataset import load_usr_dataset_by_name

# from tools.shapelets_base.shapelet_transformation.src.SVDD.SVDD import SVDD

from apply.ts_apply import TsApplyBase
from tools.preprocess import convert_to_zeros

from tools.plot import GraphForTs

# from apply.anomaly_detaction_base import AnolmalyDetectionOnXmr, AnomalyDetectionOnSeasonalESD

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, default='mbr-2023-06')
parser.add_argument('--n_splits', type=int, default=5)
parser.add_argument('--nhidden', type=int, default=8)
parser.add_argument('--nheads', type=int, default=8)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--relu', type=float, default=0.2)
parser.add_argument('--data_size', type=int, default=1)
parser.add_argument('--opt_metric', type=str, default='f1')

parser.add_argument('--niter', type=int, default=1000)
parser.add_argument('--njobs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--percentile', type=int, default=80)

parser.add_argument('--diff', action='store_true', default=False)
parser.add_argument('--standard_scale', action='store_true', default=False)

parser.add_argument('--softmax', action='store_true', default=False)
parser.add_argument('--append', action='store_true', default=False)
parser.add_argument('--sort', action='store_true', default=False)
parser.add_argument('--ft_xgb', action='store_true', default=False)
parser.add_argument('--aggregate', action='store_true', default=False)
parser.add_argument('--feat_flag', action='store_true', default=False)
parser.add_argument('--feat_norm', action='store_true', default=False)
parser.add_argument('--debug', action='store_true', default=False)

parser.add_argument('--model_cache', action='store_true', default=False)
parser.add_argument('--shapelet_cache', action='store_true', default=True)
parser.add_argument('--gpu_enable', action='store_true', default=False)
parser.add_argument('--pretrain', action='store_true', default=False)
parser.add_argument('--finetune', action='store_true', default=False)

# parser.add_argument('--measurement', type=str, default='gw')

parser.add_argument('--seg_length', type=int, default=STB['seg_length'])
parser.add_argument('--num_segment', type=int, default=STB['num_segment'])

args = parser.parse_args()
general_options = {
    'kernel': 'lr',
    # 'kernel': 'xgb',
    'opt_metric': args.opt_metric,
    'init': 0,
    'warp': 2,
    'tflag': True,
    'mode': 'embedding',
    'candidate_method': 'greedy'
}

model_options = model_args[args.dataset]
xgb_options = xgb_args.get(args.dataset, {})
xgb_options['n_jobs'] = njobs
pretrain = None if not args.pretrain else xgb_options

tflag_str = 'time_aware' if general_options['tflag'] else 'static'

class Time2GraphMbr(TsApplyBase):
    def __init__(self, shapelets_model, embedding_model):
        self.shapelets_model = shapelets_model
        self.embedding_model = embedding_model

        self.shapelets_distance = None
        self.shapelets = None
        self.shapelets_edgeslist = None

        object.__init__(self)

    def set_model(self, shapelets_model):
        self.shapelets_model = shapelets_model

    def set_edgelist(self, shapelets_edgeslist):
        self.shapelets_edgeslist = shapelets_edgeslist

    def learn_shapelets(self, x_train, y_train, shapelet_cache, shapelets_distance_cache):
        if not os.path.isfile(shapelet_cache) and args.shapelet_cache:
            Debugger.info_print('train_size {}, label size {}'.format(x_train.shape, y_train.shape))
            self.shapelets_model.learn_shapelets(x=x_train, y=y_train, num_segment=model_options['num_segment'],
                                       data_size=args.data_size)
            self.shapelets_model.save_shapelets(shapelet_cache)
            self.shapelets_distance, _ = self.shapelets_model.get_distance(x_train)
            torch.save(self.shapelets_distance, shapelets_distance_cache)

            Debugger.info_print('learn {} shapelets done...'.format(tflag_str))
        else:
            self.shapelets_model.load_shapelets(shapelet_cache, map_location=torch.device('cpu'))

        self.shapelets = torch.load(shapelet_cache, map_location=torch.device('cpu'))

        if not os.path.isfile(shapelets_distance_cache):
            self.shapelets_distance, _ = self.shapelets_model.get_distance(x_train)
            torch.save(self.shapelets_distance, shapelets_distance_cache)

        self.shapelets_distance = torch.load(shapelets_distance_cache, map_location=torch.device('cpu'))

        self.embedding_model.load_shapelets(fpath=shapelet_cache)

        return

    def get_shapelets_from_ts_with_least_distance(self, x_data):
        """
        :param x_data: 原始时序数据
        :param is_match: 是否严格匹配
        :return:
        """
        max_index_list = []

        shapelet_t = []
        for shapelet in self.shapelets:
            shapelet_t.append(shapelet[0].reshape(1, -1)[0])

        for id in range(len(x_data)):
            x_data_t = x_data[id].reshape(1, -1)[0]

            if len(shapelet_t) > 0 and len(x_data_t) > 0:
                e_len = len(shapelet_t[0])
                nums, _ = divmod(len(x_data_t), e_len)
            else:
                continue

            max_index_list.append({})

            for num in range(nums):
                x = x_data_t[num * (e_len): (num + 1) * (e_len)]
                dist_max = -1.1
                dist_min = 0
                for shapelet_str_index in range(len(shapelet_t)):
                    # 余弦相似性
                    y = shapelet_t[shapelet_str_index]
                    # dist_enc = x.dot(y) / (np.linalg.norm(x) * np.linalg.norm(y))
                    import scipy.stats as stats
                    dist_enc, p_value = stats.pearsonr(x, y)

                    # todo 需要优化和shapelet距离都很远的时间序列字段的处理
                    if dist_enc > dist_max:
                        dist_max = dist_enc
                        if dist_enc >= 0.99 and abs(np.mean(x)-np.mean(y)) < 0.1:
                            max_index_list[id][num] = [shapelet_str_index, 'match', dist_enc]
                        else:
                            max_index_list[id][num] = [shapelet_str_index, 'relation', dist_enc]

        return max_index_list

    def get_shortest_length_of_ts(self, max_index):
        shortest_length = []

        g = nx.DiGraph()

        self.shapelets_edgeslist.apply(lambda x: g.add_weighted_edges_from([(str(int(x['Source'])),
                                                                             str(int(x['Target'])),
                                                                             1.0 / (float( x['Weight']) + 0.000001))]),
                                       axis=1)

        reachable_dict = {}
        for shapelet_index in max_index:
            length = 0.0
            for index in range(len(shapelet_index) - 1):
                if str(shapelet_index[index]) not in reachable_dict.keys():
                    reachable_dict[str(shapelet_index[index])] = []
                    for reachable_node in nx.dfs_postorder_nodes(g, source=str(shapelet_index[index])):
                        reachable_dict[str(shapelet_index[index])].append(reachable_node)
                if str(shapelet_index[index + 1]) in reachable_dict[str(shapelet_index[index])]:
                    length += nx.dijkstra_path_length(g, str(shapelet_index[index]), str(shapelet_index[index + 1]))
                else:
                    length += math.inf

            shortest_length.append(length)

        return shortest_length

    def pick_anomaly_change(self):
        """
        找到异常的shapelet之间的跳转
        :return:
        """
        pass

    def get_shapelets_cluster(self, n_clusters=7): #embedding_cache,
        # model = Word2Vec()
        # embeds = model.wv.load_word2vec_format(embedding_cache).vectors

        shapelet_t = []
        for shapelet in self.shapelets:
            shapelet_t.append(shapelet[0].reshape(1, -1)[0])

        cluster_model = AgglomerativeClustering(n_clusters=None, distance_threshold=2)
        cluster_model = cluster_model.fit(shapelet_t)

        # C = 1 / (self.shapelets_distance.shape[0] * 0.1)
        #
        # svdd = SVDD(C=C, kernel='linear', zero_center=True, tol=1e-6, verbose=False)
        # svdd.fit(np.asarray(shapelet_t))
        # svdd_label = svdd.predict(np.asarray(shapelet_t))

        metric = 'euclidean'
        ts_km = TimeSeriesKMeans(n_clusters=len(set(cluster_model.labels_)), metric=metric)
        ts_km_label = ts_km.fit_predict(np.asarray(shapelet_t))

        # import scipy.cluster.hierarchy as shc
        # shc.dendrogram((shc.linkage(shapelet_t, method='ward')))

        return cluster_model.labels_, ts_km_label

def handle_time2graph(time2graph_mbr, x_train):
    max_index_list = time2graph_mbr.get_shapelets_from_ts_with_least_distance(x_train)

    match_ts_index = []
    match_ts_index_with_x = []
    max_index = []
    max_index_with_x = []

    for index in range(len(max_index_list)):
        max_index_tuple = sorted(max_index_list[index].items(), key=lambda s: s[0])

        match_ts_index_tmp = []
        match_ts_index_with_x_tmp = []
        max_index_tmp = []
        max_index_with_x_tmp = []
        for tuple_index in max_index_tuple:
            max_index_tmp.append(tuple_index[1][0])
            max_index_with_x_tmp.append(tuple_index[0])
            if tuple_index[1][1] == 'match':
                match_ts_index_tmp.append(tuple_index[1][0])
                match_ts_index_with_x_tmp.append(tuple_index[0])
        match_ts_index.append(match_ts_index_tmp)
        match_ts_index_with_x.append(match_ts_index_with_x_tmp)
        max_index.append(max_index_tmp)
        max_index_with_x.append(max_index_with_x_tmp)

    shortest_length = time2graph_mbr.get_shortest_length_of_ts(max_index)

    return match_ts_index, match_ts_index_with_x, max_index, max_index_with_x, shortest_length

def cluster_to_shapelets_df(shapelets, cluster_list):
    cluster_label_dict = {}
    for idx in range(len(cluster_list)):
        if cluster_list[idx] not in cluster_label_dict.keys():
            cluster_label_dict[cluster_list[idx]] = [[idx] + shapelets[idx][0].reshape(1, -1)[0].tolist()]
        else:
            cluster_label_dict[cluster_list[idx]].append([idx] + shapelets[idx][0].reshape(1, -1)[0].tolist())

    # for key, values in cluster_label_dict.items():
    #     svdd = SVDD(kernel='linear', zero_center=True, tol=1e-6, verbose=False)
    #     list_for_svdd = []
    #     for value in values:
    #         list_for_svdd.append(value[1:])
    #     if len(list_for_svdd) > 2:
    #         svdd.fit(np.asarray(list_for_svdd))
    #         svdd_label = svdd.predict(np.asarray(list_for_svdd))
    #         print('cluster: {}'.format(key))
    #         print(svdd_label)
    #     else:
    #         print('length of cluster: {} is smaller than 3'.format(key))

    df = pd.DataFrame()
    for key, values in cluster_label_dict.items():
        df_tmp = pd.DataFrame(values)
        df_tmp.insert(0, 'cluster_id', key, allow_duplicates=True)
        df = df.append(df_tmp)

    df = df.reset_index(drop=True)
    df.columns = ['cluster_id'] + ['shapelet_id'] + [idx for idx in range(df.shape[1]-2)]

    df = pd.melt(df, id_vars=['cluster_id', 'shapelet_id'], value_vars=[0, 1, 2, 3, 4, 5], value_name='value')

    return df


def handle_shapelets_cluster(time2graph_mbr, shapelets, n_clusters=7):
    cluster_label, ts_km_label = time2graph_mbr.get_shapelets_cluster(n_clusters=n_clusters)

    # assert len(cluster_label) == len(svdd_label)

    cluster_label_dict = {}
    for idx in range(len(cluster_label)):
        if cluster_label[idx] not in cluster_label_dict.keys():
            cluster_label_dict[cluster_label[idx]] = [idx]
        else:
            cluster_label_dict[cluster_label[idx]].append(idx)

    data_for_anomaly = cluster_to_shapelets_df(shapelets, cluster_label)
    # anomaly_esd_xmr = AnolmalyDetectionOnXmr()

    # data_ts_esd_xmr = data_for_anomaly.sort_values(by='variable').groupby(['cluster_id', 'shapelet_id']).apply(
    #     lambda x: anomaly_esd_xmr.detect(data=pd.Series(x['value'].values, index=x['variable'].values),
    #                                      if_plot=False, is_stationarity=False)
    # ).reset_index()

    # detector = Qdetector(0.8)
    # data_ts_esd_xmr = data_for_anomaly.sort_values(by='variable').groupby(['cluster_id', 'shapelet_id']).apply(
    #     lambda x: detector.detect(data=pd.Series(x['value'].values, index=x['variable'].values))
    # ).reset_index()

    def find_change_point_by_ruptures(data, n_bkps=1):
        anomalies = pd.Series(dtype=float)
        model = "l2"
        # algo = rpt.Binseg(model=model, min_size=1, jump=2).fit(np.asarray(data))
        # my_bkps = algo.predict(n_bkps=n_bkps)
        algo = rpt.Pelt(model=model, min_size=1, jump=2).fit(np.asarray(data))
        my_bkps = algo.predict(pen=n_bkps)

        if my_bkps[0] < len(data):
            anomalies = anomalies.append(pd.Series(data=[data[my_bkps[0]]], index=[my_bkps[0]]))
        else:
            data_diff = np.abs(np.diff(data))
            idx_max = np.argmax(data_diff)

            if idx_max == len(data_diff) - 1:
                anomalies = anomalies.append(pd.Series(data=[data[my_bkps[0]-1]], index=[my_bkps[0]-1]))

        return anomalies

    data_ts_esd_xmr = data_for_anomaly.sort_values(by='variable').groupby(['cluster_id', 'shapelet_id']).apply(
        lambda x: find_change_point_by_ruptures(data=pd.Series(x['value'].values, index=x['variable'].values))
    ).reset_index()

    data_ts_esd_xmr.columns = ['cluster_id', 'shapelet_id', 'index', 'value']

    return data_ts_esd_xmr, cluster_label_dict

if __name__ == '__main__':
    mp.set_start_method('spawn')
    data_file = '/Users/xiefang09/xiefang/python/time_series-20230328/time_series/data/anomaly/baseline_202306.csv'
    cluster_file = '/Users/xiefang09/xiefang/python/time_series-20230328/time_series/data/anomaly/cluster.csv'
    cluster_std_file = '/Users/xiefang09/xiefang/python/time_series-20230328/time_series/data/anomaly/cluster_std.csv'

    tmp_file = '/Users/xiefang09/xiefang/python/time_series-20230328/time_series/data/anomaly/tmp.csv'
    match_tmp_file = '/Users/xiefang09/xiefang/python/time_series-20230328/time_series/data/anomaly/shapelet_tmp.csv'

    cluster_std_file = '/Users/xiefang09/xiefang/python/time_series-20230328/time_series/data/anomaly/cluster_std.csv'
    cluster_std = pd.read_csv(cluster_std_file)

    cluster_std_target = cluster_std.query("ratio>0.5 and order==1")
    index_name_list = list(set(cluster_std_target.loc[:, '指标名称'].values.tolist()))
    sequence_list = list(set(cluster_std_target.loc[:, 'by序列分类'].values.tolist()))
    cluster_list = list(set(cluster_std_target.loc[:, 'cluster_id'].values.tolist()))

    data = pd.read_csv(data_file)
    cluster_df = pd.read_csv(cluster_file)
    data = pd.merge(data, cluster_df)

    data.loc[:, 'is_index'] = data.loc[:, '指标名称'].apply(
        lambda x: True if x in index_name_list else False
    )
    data = data.loc[data.loc[:, 'is_index'], :]
    data.drop('is_index', axis=1, inplace=True)

    data.loc[:, 'is_sequence'] = data.loc[:, 'by序列分类'].apply(
        lambda x: True if x in sequence_list else False
    )
    data = data.loc[data.loc[:, 'is_sequence'], :]
    data.drop('is_sequence', axis=1, inplace=True)

    data.loc[:, 'is_cluster'] = data.loc[:, 'cluster_id'].apply(
        lambda x: True if x in cluster_list else False
    )
    data = data.loc[data.loc[:, 'is_cluster'], :]
    data.drop('is_cluster', axis=1, inplace=True)

    data = data.query("cluster_id == 3 or cluster_id == 4 or cluster_id == 6")
    data['cluster_id'] = data['cluster_id'].apply(lambda x: 0 if x == 3 else 1)

    data.to_csv(tmp_file, encoding='utf_8_sig', index=False)

    row_number = data.shape[0]
    col_number = data.shape[1]-1
    value_cols = data.columns[4: col_number]
    for value_col in value_cols:
        data[value_col] = data[value_col].apply(convert_to_zeros)
    for i in range(row_number):
        if data.iloc[i, 4: col_number].max() - data.iloc[i, 4: col_number].min() == 0.0:
            continue
        else:
            data.iloc[i, 4: col_number] = (data.iloc[i, 4: col_number] - data.iloc[i, 4: col_number].mean()) / (data.iloc[i, 4: col_number].std())
            # data.iloc[i, 5:] = (data.iloc[i, 5:] - data.iloc[i, 5:].min()) / (data.iloc[i, 5:].max() - data.iloc[i, 5:].min())

    data_train = data.iloc[0: int(data.shape[0]), ]
    data_test = data.iloc[int(data.shape[0]):, ]

    model = Time2GraphGAT(
        gpu_enable=args.gpu_enable, n_hidden=args.nhidden, n_heads=args.nheads, dropout=args.dropout, lk_relu=args.relu,
        out_clf=True, softmax=args.softmax, data_size=args.data_size, dataset=args.dataset, njobs=args.njobs,
        niter=args.niter, batch_size=args.batch_size, append=args.append, sort=args.sort, ft_xgb=args.ft_xgb,
        ggregate=args.aggregate, feat_flag=args.feat_flag, feat_norm=args.feat_norm, pretrain=pretrain, diff=args.diff,
        standard_scale=args.standard_scale, percentile=args.percentile, **general_options,
        **model_options, debug=args.debug
    )#, measurement=args.measuremen

    t2g = Time2GraphEmbed(kernel=general_options['kernel'], seg_length=model_options['seg_length'], tflag=True)

    x_train, y_train = data_train.values[:, 4:col_number].astype(np.float).reshape(-1, data.shape[1] - 5, 1), \
                       data_train.values[:, col_number].astype(np.int)

    # model.learn_shapelets(x=x_train, y=y_train, num_segment=model_options['num_segment'], data_size=args.data_size)

    x_test, y_test = data_test.values[:, 4:col_number].astype(np.float).reshape(-1, data.shape[1] - 5, 1), \
                     data_test.values[:, col_number].astype(np.int)

    shapelet_cache = '{}/scripts/cache/{}_{}_{}_{}_shapelets.cache'.format(
        module_path, args.dataset, model_options['K'], model_options['seg_length'], tflag_str)
    shapelets_distance_cache = '{}/scripts/cache/{}_{}_{}_{}_shapelets_distance.cache'.format(
        module_path, args.dataset, model_options['K'], model_options['seg_length'], tflag_str)
    edgelist_cache = '{}/scripts/cache/edgelist/edgelist.csv'.format(module_path)

    cache_path = '{}/scripts/cache'.format(module_path)
    embedding_cache = '{}/embeds/{}.embeddings'.format(cache_path, 0)

    time2graph_mbr = Time2GraphMbr(model, t2g)
    time2graph_mbr.learn_shapelets(x_train, y_train, shapelet_cache, shapelets_distance_cache)

    shapelets_edgeslist = pd.read_csv(edgelist_cache)
    time2graph_mbr.set_edgelist(shapelets_edgeslist)

    match_ts_index, match_ts_index_with_x, max_index, max_index_with_x, shortest_length = \
        handle_time2graph(time2graph_mbr, x_train)

    # 查找严格匹配的shapelet在哪个数据里
    shapelet_to_ts = {}
    for match_ts_index_idx in range(len(match_ts_index)):
        if len(match_ts_index[match_ts_index_idx]) > 0:
            for indx in match_ts_index[match_ts_index_idx]:
                if indx not in shapelet_to_ts.keys():
                    shapelet_to_ts[indx] = [match_ts_index_idx]
                else:
                    shapelet_to_ts[indx].append(match_ts_index_idx)

    match_df_index = []
    for match_ts_index_idx in range(len(match_ts_index)):
        if len(match_ts_index[match_ts_index_idx]) > 0:
            match_df_index.append(match_ts_index_idx)

    data_train.iloc[list(set(match_df_index)), :].to_csv(match_tmp_file, encoding='utf_8_sig', index=False)

    shapelets = torch.load(shapelet_cache, map_location=torch.device('cpu'))
    shapelets_distance = torch.load(shapelets_distance_cache, map_location=torch.device('cpu'))
    shapelets_edgeslist = pd.read_csv(edgelist_cache)

    graph_ts = GraphForTs(shapelets_distance, shapelets, shapelets_edgeslist)

    data_ts_esd_xmr, cluster_label_dict = handle_shapelets_cluster(time2graph_mbr, shapelets)
    print(shapelet_to_ts)

    shapelets_anomaly_dict = {}
    for idx, value in data_ts_esd_xmr.iterrows():
        if value['shapelet_id'] not in shapelets_anomaly_dict.keys():
            shapelets_anomaly_dict[int(value['shapelet_id'])] = [int(value['index'])]
        else:
            shapelets_anomaly_dict[int(value['shapelet_id'])].append(int(value['index']))

    idx = 0
    for key, values in shapelet_to_ts.items():
        for value in values:
            graph_ts.plot_shapelets_in_ts(x_train,
                                          order=value,
                                          label='{}_{}'.format(value, key),
                                          max_index=max_index,
                                          max_index_with_x=max_index_with_x,
                                          match_ts_index=match_ts_index,
                                          shapelets_anomaly_dict=shapelets_anomaly_dict,
                                          if_plot_shapelet=True)


    for key, value in cluster_label_dict.items():
        if key in data_ts_esd_xmr.loc[:, 'cluster_id'].tolist():
            graph_ts.plot_all_shapelets(label="culster: {}".format(key), shapelets_index=value,
                                        shapelets_anomaly_dict=shapelets_anomaly_dict)









