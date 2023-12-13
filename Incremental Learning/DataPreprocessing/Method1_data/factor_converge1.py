### Converge factors into one file

import pandas as pd
import gzip
import pickle
import torch
import pdb
# pkl

# fileName = 'dataSets.pkl.gz'
# f = gzip.open(fileName, 'rb')
# result_dict = pickle.load(f)
# start_date = 20190211
# end_date = 20211223

# filtered_data = {date: data for date, data in result_dict.items() if start_date <= date <= end_date}
# with open('filtered_Xdata.pkl', 'wb') as file:
#     pickle.dump(filtered_data, file)


# with open('X90dataSet.pkl', 'rb') as file:
#     data = pickle.load(file)
# print(data.keys())




# df = pd.read_hdf('DF/DFl2c0.h5', 'Data')
# print(df)
# pdb.set_trace()
import pandas as pd

# 示例因子文件列表
factor_files = [
"HT/HTAMTNETFLOWBIGORDERRATIO.h5",
"HT/HTASKEXPBIDBIGORDERSD0.h5",
"HT/HTBBIGORDERSD0.h5",
"HT/HTBigOrderAMNetBidAmtRatio.h5",
"HT/HTBIGORDERSD0.h5",
"HT/HTBIGTRANSACTIONBIDSD.h5",
"HT/HTCORRNETBIDSPM.h5",
"HT/HTDPINASKAM.h5",
"HT/HTMEANAMOPENLEVEL1.h5",
"HT/HTMOMBIGORDER.h5",
"HT/HTNETINTENTIONAM.h5",
"HT/HTNETOPENAMTRATIO.h5",
"HT/HTNETOPENAMTSTRENGTH.h5",
"HT/HTNETSTRENGTHAM.h5",
"HT/HTTRANSACTIONBIDCR.h5",
]

# 初始化主数据框
master_df = pd.read_hdf(factor_files[0], 'Data')

# 遍历剩余文件并合并
for file in factor_files[1:]:
    temp_df = pd.read_hdf(file, 'Data')
    master_df = master_df.merge(temp_df, on=['tradingDate', 'code'], how='outer')

print(master_df)

master_df.to_hdf('HT_converge.h5', key='Data', mode='w', complevel=9, complib='zlib')
