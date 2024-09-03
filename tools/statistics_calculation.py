# coding: UTF-8

import numpy as np
import scipy.stats as stats
import pandas as pd

def count_zeros(data, ratio=0.2):
    """计算含有0的比例，超过一定比例返回False
    :param data:
    :param ratio:
    :return:
    """
    max_len = len(data)

    if max_len == 0:
        return 'Yes'
    zero_len = 0
    for i in data:
        if i == 0.0:
            zero_len += 1

    if zero_len/max_len > ratio:
        return 'Yes'

    return 'No'

def count_length(data):
    """计算序列长度，小于3返回Yes
    """
    data_len = len(data)

    if data_len <= 3:
        return 'Yes'
    return 'No'

def count_na(data):
    max_len = len(data)

    if max_len == 0:
        return 'Yes'
    na_len = 0
    for i in data:
        if pd.isnull(i):
            na_len += 1
    print(max_len, na_len, na_len/max_len)
    if na_len > 0:
        return 'Yes'

    return 'No'

# 归一化
def mean_norm_value(df, value_name):
    df[value_name + '_mean_norm'] = df[value_name].apply(
        lambda x: (x-x.mean())/x.std()
    )
    return df

def test_stat(y, iteration):
    std_dev = np.std(y)
    avg_y = np.mean(y)
    abs_val_minus_avg = abs(y - avg_y)
    max_of_deviations = max(abs_val_minus_avg)
    max_ind = np.argmax(abs_val_minus_avg)
    cal = max_of_deviations/ std_dev
    print('Test {}'.format(iteration))
    print("Test Statistics Value(R{}) : {}".format(iteration, cal))
    return cal, max_ind

def calculate_critical_value(size, alpha, iteration):
    t_dist = stats.t.ppf(1 - alpha / (2 * size), size - 2)
    numerator = (size - 1) * np.sqrt(np.square(t_dist))
    denominator = np.sqrt(size) * np.sqrt(size - 2 + np.square(t_dist))
    critical_value = numerator / denominator
    print("Critical Value(λ{}): {}".format(iteration, critical_value))
    return critical_value


def check_values(R, C, inp, max_index, iteration):
    if R > C:
        print('{} is an outlier. R{} > λ{}: {:.4f} > {:.4f} \n'.format(inp[max_index], iteration, iteration, R, C))
    else:
        print('{} is not an outlier. R{}> λ{}: {:.4f} > {:.4f} \n'.format(inp[max_index], iteration, iteration, R, C))

    return

def replace_blank(x):
    if type(x) == str:
        x = x.replace(' ', '')
        x = x.replace(',', '')

    return x
