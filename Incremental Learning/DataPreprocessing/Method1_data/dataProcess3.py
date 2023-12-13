### Process the data, such doing fillna 
import pandas as pd
import gzip
import pickle
import torch
import pdb
import time 
# pdb.set_trace()

# 读取数据
time_start=time.time()
# data_df = pd.read_hdf('DF90_jiemian.h5', 'df')
with open('HT90_jiemian.pkl', 'rb') as file:
    data_df = pickle.load(file)
time_end=time.time()
print('读取数据用时：',time_end-time_start,'s')
label_df = pd.read_hdf('YDataSets20.h5', 'data')


# 创建一个空字典来存储结果
result_dict = {}

# 遍历数据，按交易日期分组
for trading_date, group_data in data_df.groupby(level='tradingDate'):
    # pdb.set_trace()
    # if trading_date > 20220901:
    #     pdb.set_trace()
        # 获得股票代码
    codes = group_data.index.get_level_values('code').tolist()

    # 数据中的缺失值补0
    group_data.fillna(0, inplace=True)

    # 从标签 DataFrame 中获取与当前交易日期和股票代码匹配的标签
    labels = label_df.loc[trading_date].reindex(codes)

    # 如果标签中有缺失值，则补零
    labels.fillna(0, inplace=True)

    # 将数据重塑为(股票数量, 90, 特征数量)的数组，并转换为PyTorch张量
    reshaped_data = group_data.values.reshape(-1, 90, 15)
    tensor_data = torch.tensor(reshaped_data, dtype=torch.float32)

    # 转换为PyTorch张量
    tensor_labels = torch.tensor(labels.values, dtype=torch.float32)

    # 创建一个股票代码与索引的映射
    code_index_mapping = pd.Series(range(len(codes)), index=codes)

    # 创建一个字典条目，其中包括数据、股票代码和标签
    result_dict[trading_date] = [tensor_data, tensor_labels, code_index_mapping, None]

    # # 获得股票代码
    # codes = group_data.index.get_level_values('code').tolist()

    # #数据中的缺失值去除
    # valid_data = group_data.dropna()

    # # 重新获取有效的股票代码
    # valid_codes = valid_data.index.get_level_values('code').tolist()

    # # 从标签 DataFrame 中获取与当前交易日期和股票代码匹配的标签
    # labels = label_df.loc[trading_date].reindex(valid_codes)

    # # 如果标签中有缺失值，则补零
    # labels.fillna(0, inplace=True)

    # # 将有效数据重塑为(股票数量, 90, 特征数量)的数组，并转换为PyTorch张量
    # reshaped_data = valid_data.values.reshape(-1, 90, 15)
    # tensor_data = torch.tensor(reshaped_data, dtype=torch.float32)

    # # 转换为PyTorch张量
    # tensor_labels = torch.tensor(labels.values, dtype=torch.float32)

    # # 创建一个股票代码与索引的映射
    # code_index_mapping = pd.Series(range(len(valid_codes)), index=valid_codes)

    # # 创建一个字典条目，其中包括数据、股票代码和标签
    # result_dict[trading_date] = [tensor_data, tensor_labels, code_index_mapping, None]


with open('HT90dataSet.pkl', 'wb') as file:
    pickle.dump(result_dict, file)
