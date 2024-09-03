# coding: UTF-8

import numpy as np
from plotly.graph_objs import *
import plotly.graph_objects as go

import networkx as nx
import torch
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    plt.title("Hierarchical Clustering Dendrogram")

    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()

def list_split(ls, n):
    return [ls[i:i+n] for i in range(0, len(ls), n)]

class GraphForTs(object):
    def __init__(self, shapelets_distance, shapelets, shapelets_edgeslist):
        self.shapelets_distance = shapelets_distance
        self.shapelets = shapelets
        self.shapelets_edgeslist = shapelets_edgeslist
        object.__init__(self)

    def plot_anomalies(self, x_data, y_data, x_anomalies, y_anomalies, path=None):
        """
            :param data: 原始数据
            :param anomaly_indices: 原始数据异常值索引
            :param anomalies: 异常值
            :return:
            """
        layout = Layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        fig = go.Figure(layout=layout)
        fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines', name='原始数据'))

        if not len(x_anomalies) == 0:
            fig.add_trace(go.Scatter(x=x_anomalies, y=y_anomalies, mode='markers', name='shapelets点'))
        fig.add_trace(go.Scatter(x=x_anomalies, y=y_anomalies, mode='lines', name='shapelets线'))
        fig.show()

        if path:
            fig.write_image(path)

    def plot_shapelets_in_ts(self, x_data, order, label, max_index, max_index_with_x, match_ts_index,
                             shapelets_anomaly_dict=None, n=9, path=None, if_plot_shapelet=True):
        """
        :param x_data:
        :param order:
        :param label:
        :param max_index: 根据距离确定的shapelets
        :param n:
        :param path:
        :return:
        """
        assert len(x_data) == len(self.shapelets_distance)

        x_data_sub_index = []
        x_data_sub_index_with_x = []
        y_data = []
        max_index_plot = max_index[order][0: n]

        x_data_index = [i for i in range(len(x_data[order]))]
        y = x_data[order].reshape(1, -1)[0]

        for shapelet_index in max_index_plot:
            shapelet = self.shapelets[int(shapelet_index)][0]

            y_data.append(shapelet.reshape(1, -1)[0])

        for index in range(min(len(max_index_with_x[order]), n)):
            x_data_sub_index.append([i for i in range(index * self.shapelets[0][0].shape[0],
                                                      (index + 1) * self.shapelets[0][0].shape[0])])
            x_data_sub_index_with_x.append(max_index_with_x[order][index])

        layout = Layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title=go.layout.Title(
                text='title: {}'.format(label),
                xref="paper",
                x=0
            )
        )

        fig = go.Figure(layout=layout)

        fig.add_trace(go.Scatter(x=x_data_index, y=y, mode='lines', name='原始数据'))

        if if_plot_shapelet:
            for id in range(len(x_data_sub_index)):
                name = str(x_data_sub_index_with_x[id])
                if max_index_plot[id] in match_ts_index[order]:
                    name = name + '_' + str(max_index_plot[id]) + '_match'
                else:
                    name = name + '_' + str(max_index_plot[id]) + '_relation'
                fig.add_trace(go.Scatter(x=x_data_sub_index[id], y=y_data[id], mode='lines', name=name))

        anomaly_index_list = []
        anomaly_value_list = []

        for shapelet_id in range(len(max_index_plot)):
            if max_index_plot[shapelet_id] in shapelets_anomaly_dict.keys():
                for value in shapelets_anomaly_dict[max_index_plot[shapelet_id]]:
                    index_tmp = shapelet_id * len(x_data_sub_index[shapelet_id]) + value
                    anomaly_index_list.append(index_tmp)
                    anomaly_value_list.append(x_data[order].reshape(1, -1)[0][index_tmp])

        fig.add_trace(go.Scatter(x=anomaly_index_list, y=anomaly_value_list, mode='markers', name='anomaly'))

        fig.show()

        if path:
            fig.write_image(path)

    def plot_shapelets_network(self, edge_ratio=0.05):
        g = nx.DiGraph()

        self.shapelets_edgeslist.apply(lambda x: g.add_weighted_edges_from([(str(int(x['Source'])),
                                                                             str(int(x['Target'])),
                                                                             float(x['Weight']))]), axis=1)

        # pr = nx.pagerank(g)
        #
        # df_pr = pd.DataFrame([[x[0], x[1]] for x in sorted(pr.items(), key=lambda s: -s[1])],
        #                      columns=["Id", "pagerank"])

        weight_list = sorted(self.shapelets_edgeslist.loc[:, 'Weight'].tolist(), reverse=True)
        weight_list = weight_list[0: int(self.shapelets_edgeslist.shape[0]*edge_ratio)]
        if len(weight_list) == 0:
            weight_min = 0.0
        else:
            weight_min = min(weight_list)

        shapelets_edge_tmp = self.shapelets_edgeslist.query("Weight >= {}".format(weight_min))

        g_sub = nx.DiGraph()
        shapelets_edge_tmp.apply(lambda x: g_sub.add_weighted_edges_from([(str(int(x['Source'])),
                                                                           str(int(x['Target'])),
                                                                           float(x['Weight']))]), axis=1)

        node_size = [g_sub.degree(i)*10 for i in g_sub.nodes()]
        node_color = [g_sub.degree(i) for i in g_sub.nodes()]
        edge_width = [g.get_edge_data(*e)['weight']*50 for e in g.edges()]

        plt.figure(figsize=(10, 6))

        options = {
            'pos': nx.spring_layout(g_sub),
            'node_size': node_size,
            'node_color': node_color,
            'edge_color': 'gray',
            'width': edge_width,
            'font_size': 14,
            'font_color': 'red'
        }

        nx.draw(g_sub, with_labels=True, **options)
        plt.show()

    def get_stop_shapelet(self):
        pass

    def plot_all_shapelets(self, label, shapelets_index=None, path=None, shapelets_anomaly_dict=None):
        shapelet_t = []
        for shapelet in self.shapelets:
            shapelet_t.append(shapelet[0].reshape(1, -1)[0])

        shapelet_t_plot = []
        if shapelets_index:
            for idx in shapelets_index:
                shapelet_t_plot.append(shapelet_t[idx])
        else:
            shapelet_t_plot = shapelet_t
            shapelets_index = [i for i in range(len(shapelet_t))]

        if len(shapelet_t_plot) == 1:
            title = 'title: {}-{}'.format(label, (shapelets_index[0]))
        else:
            title = 'title: {}'.format(label)

        layout = Layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title=go.layout.Title(
                text=title,
                xref="paper",
                x=0
            )
        )

        fig = go.Figure(layout=layout)

        for shapelet_id in range(len(shapelet_t_plot)):
            shapelet = shapelet_t_plot[shapelet_id]
            fig.add_trace(go.Scatter(x=[i for i in range(len(shapelet))], y=shapelet, mode='lines',
                                     name='shapelet: {}'.format(shapelets_index[shapelet_id]))) # line=dict(color='rgb(204, 204, 204)')

        anomaly_index_list = []
        anomaly_value_list = []

        for shapelet_id in range(len(shapelets_index)):
            if shapelets_index[shapelet_id] in shapelets_anomaly_dict.keys():
                for value in shapelets_anomaly_dict[shapelets_index[shapelet_id]]:
                    anomaly_index_list.append(value)
                    anomaly_value_list.append(shapelet_t_plot[shapelet_id][value])

        fig.add_trace(go.Scatter(x=anomaly_index_list, y=anomaly_value_list, mode='markers', name='anomaly'))

        fig.show()

        if path:
            fig.write_image(path)

        return shapelet_t_plot












