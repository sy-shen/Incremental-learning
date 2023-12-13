### Put the strong factor into the dataset
import pickle
import pandas as pd
import torch
import numpy as np
import pdb
import gzip

# 读取新的数据
# df1 = pd.read_hdf('280W.h5', 'ADJValue')
df1 = pd.read_hdf('residuals_1922.h5', 'residuals')
# with open('new_dataSet.pkl', 'rb') as file:
#     result_dict = pickle.load(file)
with open('HT90dataSet.pkl', 'rb') as file:
    result_dict = pickle.load(file)

for trading_date, data_list in result_dict.items():
    # pdb.set_trace()
    tensor_data, tensor_labels, code_index_mapping, _ = data_list
    
    # 获取与当前交易日期匹配的新数据，并重置索引以便于访问‘code’列
    new_data_for_date = df1.loc[str(trading_date)].reset_index() ##For regression
    # new_data_for_date = df1.loc[trading_date].reset_index()
    # 创建一个与code_index_mapping大小匹配的新数据数组，并填充新数据
    new_data_array = np.zeros(len(code_index_mapping))
    for index, row in new_data_for_date.iterrows():
        # code = row['code']
        code = "{:06d}".format(int(row['code']))
        if code in code_index_mapping:  # 检查code是否存在于映射中
            # pdb.set_trace()
            idx = code_index_mapping[code]
            # new_data_array[idx] = row['ALPHA480W20R02']
            new_data_array[idx] = row['residuals']

    # 转换新数据数组为PyTorch张量
    tensor_new_data = torch.tensor(new_data_array, dtype=torch.float32)

    # 将新数据添加到result_dict中的相应位置
    result_dict[trading_date][3] = tensor_new_data

with open('HT90dataSet_Residual.pkl', 'wb') as file:
    pickle.dump(result_dict, file)

