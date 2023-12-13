### Regress the label on the strong factor
import pickle as pickle
import torch
import pdb
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

a = pd.read_hdf('YDataSets20.h5', 'data')
b = pd.read_hdf('280W.h5', 'ADJValue')


# pdb.set_trace()
a = a.to_frame()
df0 = a[(a.index.get_level_values(0) >= 20190211) & (a.index.get_level_values(0) <= 20221130)]

df0 = df0.reset_index(level=[0,1]) # Reset all levels of the index
df0.columns = ['tradingDate', 'code', 'label'] # Rename the columns
df0 = df0.set_index(['tradingDate', 'code']) # Set the new index

df0.index = df0.index.set_levels(df0.index.levels[0].astype(str), level=0)
df0.index = df0.index.set_levels(df0.index.levels[1].astype(int), level=1)


# pdb.set_trace()
df1 = b[(b.index.get_level_values(0) >= 20190211) & (b.index.get_level_values(0) <= 20221130)]
df1 = df1.reset_index()
df1.columns = ['tradingDate', 'code', 'value'] # Rename the columns
df1 = df1.set_index(['tradingDate', 'code']) # Set the new index
df1.index = df1.index.set_levels(df1.index.levels[0].astype(str), level=0)
df1.index = df1.index.set_levels(df1.index.levels[1].astype(int), level=1)


# 假设第一个数据集是 df_label，第二个数据集是 df_value
# pdb.set_trace()
# 合并数据集
merged_df = pd.merge(df0, df1, left_index=True, right_index=True)
merged_df =merged_df.dropna()

# 取出标签和值
labels = merged_df['label'].values.reshape(-1, 1)
values = merged_df['value'].values.reshape(-1, 1)

# 进行线性回归
reg = LinearRegression().fit(values, labels)

# 获取预测值
predictions = reg.predict(values)

# 计算残差
residuals = labels - predictions

# 将残差加入到合并的数据框中
merged_df['residuals'] = residuals

print(merged_df['residuals'])

# 将残差保存为 .h5 文件
residuals_df = merged_df[['residuals']]
residuals_df.to_hdf('residuals_1922.h5', key='residuals', mode='w')
