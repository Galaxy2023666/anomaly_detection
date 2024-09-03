import os
import sys
import pandas as pd
import numpy as np
import json
from functools import reduce
from datetime import datetime
import copy

from tools.statistics_calculation import *

# Data folders
output_folder = '../../data/experiment_data/processed'
data_folder = 'data'

datasets = ['synthetic', 'SMD', 'SWaT', 'SMAP', 'MSL', 'WADI', 'MSDS', 'UCR', 'MBA', 'NAB']
wadi_drop = ['2_LS_001_AL', '2_LS_002_AL', '2_P_001_STATUS', '2_P_002_STATUS']

train_start_month = '2019-01-01'
train_end_month = '2022-09-01'
k_loader_start_month = '2022-10-01'
k_loader_end_month = '2023-12-01'
val_start_month = '2023-01-01'
val_end_month = '2023-03-01'
test_start_month = '2023-04-01'
test_end_month = '2023-09-01'

def convert_to_zeros(x):
    # 为了兼容之前的输入，需要同时考虑
    if pd.isnull(x):
        return 0.0
    elif type(x) == float:
        return x

    if type(x) == str:
        x = x.replace(' ', '')
    if x == '- ' or x == '/' or x == '' or pd.isnull(x) or x == '-':
        return 0.0
    else:
        return x


def process_data(data, year_month, dataset_folder):
    dim_cols = ['指标名称', 'by BG/平台', 'by序列分类', '维度', 'method']
    col_name = ['2019-01', '2019-02', '2019-03', '2019-04', '2019-05', '2019-06', '2019-07', '2019-08', '2019-09',
                '2019-10', '2019-11', '2019-12', '2020-01', '2020-02', '2020-03', '2020-04', '2020-05', '2020-06',
                '2020-07', '2020-08', '2020-09', '2020-10', '2020-11', '2020-12', '2021-01', '2021-02', '2021-03',
                '2021-04', '2021-05', '2021-06', '2021-07', '2021-08', '2021-09', '2021-10', '2021-11', '2021-12',
                '2022-01', '2022-02', '2022-03', '2022-04', '2022-05', '2022-06', '2022-07', '2022-08', '2022-09',
                '2022-10', '2022-11', '2022-12', '2023-01', '2023-02', '2023-03', '2023-04', '2023-05', '2023-06',
                '2023-07', '2023-08', '2023-09']
    if year_month not in col_name:
        col_name.append(year_month)

    col_name = dim_cols + col_name

    data_long = pd.melt(data.loc[:, col_name], id_vars=dim_cols, var_name="time",
                        value_name="value")
    data_long.loc[:, 'Month'] = pd.to_datetime(data_long.loc[:, 'time'], format='%Y-%m')
    data_long.drop('time', axis=1, inplace=True)
    # data_long.loc[:, 'value'] = pd.to_numeric(data_long['value'], errors='coerce').fillna(0)
    data_long.loc[:, 'value'] = data_long.loc[:, 'value'].apply(
        lambda x: 0.0 if pd.isnull(x) else x
    )

    data_long.loc[:, 'value'] = data_long.loc[:, 'value'].apply(convert_to_zeros)

    data_tmp = data_long.groupby(dim_cols)['value'].apply(
        lambda x: count_zeros(x.values)
    ).reset_index(name='is_zeros')
    data_long = pd.merge(data_long, data_tmp.query("is_zeros == 'No'"))

    data_long_tmp = copy.deepcopy(data_long)
    data_long_tmp.loc[:, 'level'] = data_long_tmp.loc[:, 'by BG/平台'].apply(lambda t: t.count('-'))
    data_wide_tmp = data_long_tmp.query("level == 1").pivot_table(index=['指标名称', 'Month', 'by序列分类', 'method'],
                                                                  columns=['by BG/平台'],
                                                                  values='value', fill_value=0).reset_index()
    data_wide_tmp.to_csv(dataset_folder + '/all_1.csv', encoding='utf_8_sig', index=False)

    data_mean = data_long.groupby(dim_cols)['value'].mean().reset_index(name='mean')
    data_std = data_long.groupby(dim_cols)['value'].std().reset_index(name='std')

    data_long = pd.merge(data_long, data_mean)
    data_long = pd.merge(data_long, data_std)
    data_long.loc[:, 'value'] = (data_long.loc[:, 'value'] - data_long.loc[:, 'mean'])/data_long.loc[:, 'std']

    data_long.drop(['mean', 'std'], inplace=True, axis=1)

    data_long.index = pd.Series(range(data_long.shape[0]))
    data_long.loc[:, 'level'] = data_long.loc[:, 'by BG/平台'].apply(lambda t: t.count('-'))

    level_list = set(data_long.loc[:, 'level'].values.tolist())
    method_list = set(data_long.loc[:, 'method'].values.tolist())

    file_list = []
    for level in level_list:
        for method in method_list:
            data_long_tmp = data_long.query("level == {} and method == '{}'".format(level, method))
            data_wide_tmp = data_long_tmp.pivot_table(index=['指标名称', 'Month', 'by序列分类'], columns=['by BG/平台'],
                                                      values='value', fill_value=0).reset_index()

            # 排序以便后续进行批量学习
            data_wide_tmp = data_wide_tmp.sort_values(by=['指标名称', 'by序列分类', 'Month'])

            # todo 区分训练集和测试集，且因为每个生成的数据都是依赖指标分类的，所以最好根据时间范围来确定
            data_wide_tmp_train = data_wide_tmp.query(
                "Month >= '{}' and Month <= '{}'".format(train_start_month, train_end_month))

            if data_wide_tmp_train.shape[0] > 1000:
                file_list.append('{}_level_{}'.format(method, level))

                data_wide_tmp_k_loader = data_wide_tmp.query(
                    "Month >= '{}' and Month <= '{}'".format(k_loader_start_month, k_loader_end_month))
                data_wide_tmp_val = data_wide_tmp.query(
                    "Month >= '{}' and Month <= '{}'".format(val_start_month, val_end_month))
                data_wide_tmp_test = data_wide_tmp.query(
                    "Month >= '{}' and Month <= '{}'".format(test_start_month, test_end_month))

                data_wide_tmp_train.to_csv(dataset_folder + '/{}_level_{}_train.csv'.format(method, level),
                                           encoding='utf_8_sig', index=False)
                data_wide_tmp_k_loader.to_csv(dataset_folder + '/{}_level_{}_k_loader.csv'.format(method, level),
                                              encoding='utf_8_sig', index=False)
                data_wide_tmp_val.to_csv(dataset_folder + '/{}_level_{}_val.csv'.format(method, level),
                                         encoding='utf_8_sig', index=False)
                data_wide_tmp_test.to_csv(dataset_folder + '/{}_level_{}_test.csv'.format(method, level),
                                          encoding='utf_8_sig', index=False)

    file_df = pd.DataFrame(file_list)
    file_df.columns = ['file_name']

    file_df.to_csv(dataset_folder + '/file_names.csv', encoding='utf_8_sig', index=False)


def load_and_save(category, filename, dataset, dataset_folder):
    temp = np.genfromtxt(os.path.join(dataset_folder, category, filename),
                         dtype=np.float64,
                         delimiter=',')
    print(dataset, category, filename, temp.shape)
    np.save(os.path.join(output_folder, f"SMD/{dataset}_{category}.npy"), temp)
    return temp.shape


def load_and_save2(category, filename, dataset, dataset_folder, shape):
    temp = np.zeros(shape)
    with open(os.path.join(dataset_folder, 'interpretation_label', filename), "r") as f:
        ls = f.readlines()
    for line in ls:
        pos, values = line.split(':')[0], line.split(':')[1].split(',')
        start, end, indx = int(pos.split('-')[0]), int(pos.split('-')[1]), [int(i) - 1 for i in values]
        temp[start - 1:end - 1, indx] = 1
    print(dataset, category, filename, temp.shape)
    np.save(os.path.join(output_folder, f"SMD/{dataset}_{category}.npy"), temp)


def normalize(a):
    a = a / np.maximum(np.absolute(a.max(axis=0)), np.absolute(a.min(axis=0)))
    return (a / 2 + 0.5)


def normalize2(a, min_a=None, max_a=None):
    if min_a is None: min_a, max_a = min(a), max(a)
    return (a - min_a) / (max_a - min_a), min_a, max_a


def normalize3(a, min_a=None, max_a=None):
    if min_a is None: min_a, max_a = np.min(a, axis=0), np.max(a, axis=0)
    return (a - min_a) / (max_a - min_a + 0.0001), min_a, max_a


def convertNumpy(df):
    x = df[df.columns[3:]].values[::10, :]
    return (x - x.min(0)) / (x.ptp(0) + 1e-4)


def load_data(dataset):
    folder = os.path.join(output_folder, dataset)
    if os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    if dataset == 'synthetic':
        train_file = os.path.join(data_folder, dataset, 'synthetic_data_with_anomaly-s-1.csv')
        test_labels = os.path.join(data_folder, dataset, 'test_anomaly.csv')
        dat = pd.read_csv(train_file, header=None)
        split = 10000
        train = normalize(dat.values[:, :split].reshape(split, -1))
        test = normalize(dat.values[:, split:].reshape(split, -1))
        lab = pd.read_csv(test_labels, header=None)
        lab[0] -= split
        labels = np.zeros(test.shape)
        for i in range(lab.shape[0]):
            point = lab.values[i][0]
            labels[point - 30:point + 30, lab.values[i][1:]] = 1
        test += labels * np.random.normal(0.75, 0.1, test.shape)
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file))
    elif dataset == 'SMD':
        dataset_folder = '../../data/experiment_data/original_data/SMD'
        file_list = os.listdir(os.path.join(dataset_folder, "train"))
        for filename in file_list:
            if filename.endswith('.txt'):
                load_and_save('train', filename, filename.strip('.txt'), dataset_folder)
                s = load_and_save('test', filename, filename.strip('.txt'), dataset_folder)
                load_and_save2('labels', filename, filename.strip('.txt'), dataset_folder, s)
    elif dataset == 'UCR':
        dataset_folder = '../../data/experiment_data/original_data/UCR'
        file_list = os.listdir(dataset_folder)
        for filename in file_list:
            if not filename.endswith('.txt'): continue
            vals = filename.split('.')[0].split('_')
            dnum, vals = int(vals[0]), vals[-3:]
            vals = [int(i) for i in vals]
            temp = np.genfromtxt(os.path.join(dataset_folder, filename),
                                 dtype=np.float64,
                                 delimiter=',')
            min_temp, max_temp = np.min(temp), np.max(temp)
            temp = (temp - min_temp) / (max_temp - min_temp)
            train, test = temp[:vals[0]], temp[vals[0]:]
            labels = np.zeros_like(test)
            labels[vals[1] - vals[0]:vals[2] - vals[0]] = 1
            train, test, labels = train.reshape(-1, 1), test.reshape(-1, 1), labels.reshape(-1, 1)
            for file in ['train', 'test', 'labels']:
                np.save(os.path.join(folder, f'{dnum}_{file}.npy'), eval(file))
    elif dataset == 'NAB':
        dataset_folder = '../../data/experiment_data/original_data/NAB'
        file_list = os.listdir(dataset_folder)
        with open(dataset_folder + '/labels.json') as f:
            labeldict = json.load(f)
        for filename in file_list:
            if not filename.endswith('.csv'): continue
            df = pd.read_csv(dataset_folder + '/' + filename)
            vals = df.values[:, 1]
            labels = np.zeros_like(vals, dtype=np.float64)
            for timestamp in labeldict['realKnownCause/' + filename]:
                tstamp = timestamp.replace('.000000', '')
                index = np.where(((df['timestamp'] == tstamp).values + 0) == 1)[0][0]
                labels[index - 4:index + 4] = 1
            min_temp, max_temp = np.min(vals), np.max(vals)
            vals = (vals - min_temp) / (max_temp - min_temp)
            train, test = vals.astype(float), vals.astype(float)
            train, test, labels = train.reshape(-1, 1), test.reshape(-1, 1), labels.reshape(-1, 1)
            fn = filename.replace('.csv', '')
            for file in ['train', 'test', 'labels']:
                np.save(os.path.join(folder, f'{fn}_{file}.npy'), eval(file))
    elif dataset == 'MSDS':
        dataset_folder = '../../data/experiment_data/original_data/MSDS'
        df_train = pd.read_csv(os.path.join(dataset_folder, 'train.csv'))
        df_test = pd.read_csv(os.path.join(dataset_folder, 'test.csv'))
        df_train, df_test = df_train.values[::5, 1:], df_test.values[::5, 1:]
        _, min_a, max_a = normalize3(np.concatenate((df_train, df_test), axis=0))
        train, _, _ = normalize3(df_train, min_a, max_a)
        test, _, _ = normalize3(df_test, min_a, max_a)
        labels = pd.read_csv(os.path.join(dataset_folder, 'labels.csv'))
        labels = labels.values[::1, 1:]
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file).astype('float64'))
    elif dataset == 'SWaT':
        dataset_folder = '../../data/experiment_data/original_data/SWaT'
        file = os.path.join(dataset_folder, 'series.json')
        df_train = pd.read_json(file, lines=True)[['val']][3000:6000]
        df_test = pd.read_json(file, lines=True)[['val']][7000:12000]
        train, min_a, max_a = normalize2(df_train.values)
        test, _, _ = normalize2(df_test.values, min_a, max_a)
        labels = pd.read_json(file, lines=True)[['noti']][7000:12000] + 0
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file))
    elif dataset in ['SMAP', 'MSL']:
        dataset_folder = '../../data/experiment_data/original_data/SMAP_MSL'
        file = os.path.join(dataset_folder, 'labeled_anomalies.csv')
        values = pd.read_csv(file)
        values = values[values['spacecraft'] == dataset]
        filenames = values['chan_id'].values.tolist()
        for fn in filenames:
            train = np.load(f'{dataset_folder}/train/{fn}.npy')
            test = np.load(f'{dataset_folder}/test/{fn}.npy')
            train, min_a, max_a = normalize3(train)
            test, _, _ = normalize3(test, min_a, max_a)
            np.save(f'{folder}/{fn}_train.npy', train)
            np.save(f'{folder}/{fn}_test.npy', test)
            labels = np.zeros(test.shape)
            indices = values[values['chan_id'] == fn]['anomaly_sequences'].values[0]
            indices = indices.replace(']', '').replace('[', '').split(', ')
            indices = [int(i) for i in indices]
            for i in range(0, len(indices), 2):
                labels[indices[i]:indices[i + 1], :] = 1
            np.save(f'{folder}/{fn}_labels.npy', labels)
    elif dataset == 'CIRCE':
        dataset_folder = '../../data/experiment_data/original_data/CIRCE/Sin fundamental'
        prefalta = np.load(os.path.join(dataset_folder, f'DB_CT802_PulsoPrefalta_4000.npz'))
        register = prefalta['tipos']
        # SIN FILTRAR LA FUNDAMENTAL
        # dataset_folder = 'data/CIRCE/CSV'
        # file = os.path.join(dataset_folder, 'PF_P_Error2.csv')
        # df = pd.read_csv(file)
        # R_prefalta=df['Prefault R (V)']
        # S_prefalta=df['Prefault S (V)']
        # T_prefalta=df['Prefault T (V)']

        # register = np.concatenate((R_prefalta[1030:6030], S_prefalta[51030:56030], T_prefalta[101030:106030]))
        R = register[:4000]
        S = register[4000:8000]
        T = register[8000:]

        register = np.vstack((R, S, T))
        train = np.transpose(register)
        train, min_a, max_a = normalize3(train)

        prefalta = np.load(os.path.join(dataset_folder, f'DB_CT802_PulsoFalta_4000.npz'))
        register = prefalta['tipos'][1, :]
        # R_falta=df['Fault R (V)']
        # S_falta=df['Fault S (V)']
        # T_falta=df['Fault T (V)']
        # register = np.vstack((R_falta[52030:57030], S_falta[102030:107030], T_falta[152030:157030]))
        R = register[:4000]
        S = register[4000:8000]
        T = register[8000:]

        register = np.vstack((R, S, T))
        test = np.transpose(register)
        test, _, _ = normalize3(test, min_a, max_a)

        labels = np.zeros((4000, 3))
        np.save(f'{folder}/CIRCE_train.npy', train)
        np.save(f'{folder}/CIRCE_test.npy', test)
        np.save(f'{folder}/CIRCE_labels.npy', labels)
    elif dataset == 'WADI':
        dataset_folder = '../../data/experiment_data/original_data/WADI'
        ls = pd.read_csv(os.path.join(dataset_folder, 'WADI_attacklabels.csv'))
        train = pd.read_csv(os.path.join(dataset_folder, 'WADI_14days.csv'), skiprows=1000, nrows=2e5)
        test = pd.read_csv(os.path.join(dataset_folder, 'WADI_attackdata.csv'))
        train.dropna(how='all', inplace=True);
        test.dropna(how='all', inplace=True)
        train.fillna(0, inplace=True);
        test.fillna(0, inplace=True)
        test['Time'] = test['Time'].astype(str)
        test['Time'] = pd.to_datetime(test['Date'] + ' ' + test['Time'])
        labels = test.copy(deep=True)
        for i in test.columns.tolist()[3:]: labels[i] = 0
        for i in ['Start Time', 'End Time']:
            ls[i] = ls[i].astype(str)
            ls[i] = pd.to_datetime(ls['Date'] + ' ' + ls[i])
        for index, row in ls.iterrows():
            to_match = row['Affected'].split(', ')
            matched = []
            for i in test.columns.tolist()[3:]:
                for tm in to_match:
                    if tm in i:
                        matched.append(i);
                        break
            st, et = str(row['Start Time']), str(row['End Time'])
            labels.loc[(labels['Time'] >= st) & (labels['Time'] <= et), matched] = 1
        train, test, labels = convertNumpy(train), convertNumpy(test), convertNumpy(labels)
        print(train.shape, test.shape, labels.shape)
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file))
    elif dataset == 'MBA':
        dataset_folder = '../../data/experiment_data/original_data/MBA'
        ls = pd.read_excel(os.path.join(dataset_folder, 'labels.xlsx'))
        train = pd.read_excel(os.path.join(dataset_folder, 'train.xlsx'))
        test = pd.read_excel(os.path.join(dataset_folder, 'test.xlsx'))
        train, test = train.values[1:, 1:].astype(float), test.values[1:, 1:].astype(float)
        train, min_a, max_a = normalize3(train)
        test, _, _ = normalize3(test, min_a, max_a)
        ls = ls.values[:, 1].astype(int)
        labels = np.zeros_like(test)
        for i in range(-20, 20):
            labels[ls + i, :] = 1
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file))
    # todo
    elif dataset == 'MBR':
        dataset_folder = '/Users/xiefang09/xiefang/python/time_series-20230328/time_series/data/anomaly_on_deeplearning/MBR'
        data_set = pd.read_csv(dataset_folder + '/baseline_202309.csv')
        data_classification = pd.read_csv(dataset_folder + '/classification.csv')
        data_set = pd.merge(data_set, data_classification)
        process_data(data_set, '2023-09', dataset_folder)
    else:
        raise Exception(f'Not Implemented. Check one of {datasets}')


if __name__ == '__main__':
    dataset = 'MSDS'
    original_data_path = '/Users/xiefang09/xiefang/python/time_series-20230328/time_series/data/experiment_data/original_data'
    if not os.path.exists(os.path.join(original_data_path, dataset, 'labels.csv')):
        files = os.listdir(os.path.join(original_data_path, dataset, 'metrics'))
        dfs = []

        # Read csv files

        for file in files:
            if '.csv' in file and 'wally' in file:
                df = pd.read_csv(os.path.join(original_data_path, dataset, 'metrics', file))
                df = df.drop(columns=['load.cpucore', 'load.min1', 'load.min5', 'load.min15'])
                dfs.append(df)

        # Process dataframes
        start = dfs[0].min()['now']
        end = dfs[0].max()['now']
        for df in dfs:
            if df.min()['now'] > start:
                start = df.min()['now']

        id_vars = ['now']
        dfs2 = []
        for df in dfs:
            df = df.drop(np.argwhere(list(df['now'] < start)).reshape(-1))
            df = df.drop(np.argwhere(list(df['now'] > end)).reshape(-1))
            melted = df.melt(id_vars=id_vars).dropna()
            df = melted.pivot_table(index=id_vars, columns="variable", values="value")
            dfs2.append(df)
        dfs = dfs2

        df_merged = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True), dfs)

        # Change timezone string format

        ni = []
        for i in df_merged.index:
            dt = datetime.strptime(i[:-5], '%Y-%m-%d %H:%M:%S')
            ni.append(dt.strftime('%Y-%m-%dT%H:%M:%SZ'))
        df_merged.index = ni

        # Save train and test sets

        start = round(df_merged.shape[0] * 0.1)
        df_merged = df_merged[start:]
        split = round(df_merged.shape[0] / 2)
        df_merged[:split].to_csv(os.path.join(original_data_path, dataset, 'train.csv'))
        df_merged[split:].to_csv(os.path.join(original_data_path, dataset, 'test.csv'))
        d = pd.DataFrame(0, index=np.arange(df_merged[split:].shape[0]), columns=df_merged.columns)
        d.to_csv(os.path.join(original_data_path, dataset, 'labels.csv'))

    # commands = sys.argv[1:]
    # commands = ['SMAP', 'MSL', 'SWaT', 'SMD', 'UCR', 'MBA', 'NAB', 'WADI', 'MSDS']
    commands = ['MBR']
    load = []
    if len(commands) > 0:
        for d in commands:
            load_data(d)
    else:
        print("Usage: python preprocess.py <datasets>")
        print(f"where <datasets> is space separated list of {datasets}")
