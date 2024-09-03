import pandas as pd

# 删除没有信息的样本，包括shapelet一致且包含一个值
def drop_useless_samples(data, shapelet_drop_index, top, ratio):
    """
    :param data:
    :param shapelet_drop_index: 通过shapelet识别出来的信息不高的样本
    :param top: 出现次数多的前多少高
    :param ratio: 出现次数多的前多少高加起来超过次数的占比
    :return:
    """
    # 1、占比太多
    drop_index = []
    for row_index, row_data in data.iterrows():
        if len(shapelet_drop_index) == 0:
            count_dict = {}
            len_ = 0
            for col_index, col_data in row_data.iteritems():
                len_ += 1
                if col_data in count_dict:
                    count_dict[col_data] += 1
                else:
                    count_dict[col_data] = 1
            count_list = sorted(count_dict.items(), key=lambda s: -s[1])
            top_ = min(len(count_list), top)

            sum_ = 0.0
            for value in count_list:
                if top_ > 0:
                    sum_ += value[1]
                    top_ -= 1
                else:
                    break
            if sum_ / len_ > ratio:
                drop_index.append(row_index)
        else:
            if row_index in shapelet_drop_index:
                count_dict = {}
                len_ = 0
                for col_index, col_data in row_data.iteritems():
                    len_ += 1
                    if col_data in count_dict:
                        count_dict[col_data] += 1
                    else:
                        count_dict[col_data] = 1
                count_list = sorted(count_dict.items(), key=lambda s: -s[1])
                top_ = min(len(count_list), top)

                sum_ = 0.0
                for value in count_list:
                    if top_ > 0:
                        sum_ += value[1]
                        top_ -= 1
                    else:
                        break
                if sum_ / len_ > ratio:
                    drop_index.append(row_index)

    # 2、只有没几个值
    return drop_index

class TsApplyBase(object):
    def __init__(self):
        return object.__init__(self)