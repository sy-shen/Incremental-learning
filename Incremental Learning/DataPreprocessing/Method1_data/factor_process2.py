### Set n_days rolling window for each factor


import pandas as pd
import pdb
# pdb.set_trace()
master_df = pd.read_hdf('HT_converge.h5', 'Data')

# 设置双重索引
master_df.set_index('code', append=True, inplace=True)

# currentdf = master_df.sort_index()
currentdf = master_df
df = currentdf.loc['20180104':'20221130']

def zscore_group(df_group):
    return (df_group - df_group.mean()) / df_group.std()

df_normalized = df.groupby('tradingDate').apply(zscore_group).reset_index(level=0, drop=True)
# pdb.set_trace()
currentdf = df_normalized.sort_index()
dataFrameList=[]
currentdfUnstack=currentdf.unstack(1)

groupPeriod = 1
trainTsPeriod = 90

for i in range(groupPeriod*trainTsPeriod-groupPeriod,0,-groupPeriod):
    dataFrameList.append(currentdfUnstack.shift(i).stack())
# pdb.set_trace()    
dataFrameList.append(currentdf)
XDataSetsAll = pd.concat(dataFrameList, axis=1)
XDataSetsAll.columns=pd.MultiIndex.from_product([list(range(trainTsPeriod)),list(currentdf.columns.get_level_values(0)) ])
XDataSetsAll = XDataSetsAll.sort_index().swaplevel(axis=1)
XDataSetsAll = XDataSetsAll.loc['20190221':'20221130']
# XDataSetsAll.to_hdf('DF90_jiemian.h5', key='df', mode='w', complevel=9, complib='zlib')
XDataSetsAll.to_pickle("HT90_jiemian.pkl")
