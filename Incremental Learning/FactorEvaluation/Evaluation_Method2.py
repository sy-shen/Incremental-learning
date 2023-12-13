### Evaluate the performance of the factors, such as IC, CAP_IC, and Group_IC for Method2

import pandas as pd
import numpy as np
import pdb
import openpyxl

def calPerformance(predictY, testY, df_cap, num_cap=5, num_group=10):
        
        index_intersection = predictY.index.intersection(testY.index)
        predictY = predictY.reindex(index_intersection)
        testY = testY.reindex(index_intersection)

        weakFactorColumns = list(predictY.columns)
        testY.name = 'objectivReturnName'
        objectivReturnName = testY.name
        df = pd.concat([predictY, testY], join='inner', axis=1)
        df['mean'] = df[weakFactorColumns].mean(axis=1)
        calFactorNames = []
        calFactorNames.extend(weakFactorColumns)
        calFactorNames.append('mean')
        df = df.sort_index()
        df.index.names = ['tradingDate', 'code']
        
        # IC
        testIC = df.groupby("tradingDate").apply(lambda x: x[calFactorNames].apply(lambda y: y.corr(x[objectivReturnName], method='spearman')))

        # ICIR
        testICIR = (testIC.mean() / testIC.std()).round(3)

        # Correlation
        corr = df[weakFactorColumns].corr(method='pearson')
        mean = df[weakFactorColumns].mean(axis=1)
        std = df[weakFactorColumns].std()
        pdb.set_trace()
        
        # CAP
        ds_cap = df_cap.reset_index().set_index(['tradingDate', 'code'])['CAP']
        ds_cap.index.names = ['tradingDate', 'code']
        index_intersection_cap = df.index.intersection(ds_cap.index)
        ds_cap = ds_cap.reindex(index_intersection_cap)
        df_cap = df.reindex(index_intersection_cap)
        ds_cap_divide = ds_cap.groupby('tradingDate').apply(lambda x: pd.qcut(x, q=num_cap, labels=False, duplicates='drop'))
        list_ds_ic_cap = []

        for date, group in df_cap.groupby(level='tradingDate'):
            sub_ds_cap_divide = ds_cap_divide.xs(date, level=0)
            
            if not group.index.equals(sub_ds_cap_divide.index):
                sub_ds_cap_divide = sub_ds_cap_divide.droplevel(0)
               
            list_ds_ic_cap.append(group.groupby(sub_ds_cap_divide).apply(lambda x: x[calFactorNames].corrwith(x[objectivReturnName], method='spearman')))
        
        ic_cap = pd.concat(list_ds_ic_cap)
        ic_cap_group_mean = ic_cap.groupby(level='CAP').mean()
        
        # Group alpha
        
        ds_alpha = testY - testY.groupby('tradingDate').mean()
        list_ds_alpha_group = []
        for _factor in predictY.columns:
            ds_predict = predictY[_factor]
            ds_predict_divide = ds_predict.groupby('tradingDate').apply(lambda x: pd.qcut(x, q=num_group, labels=False, duplicates='drop'))

            # 移除多级索引中的额外的tradingDate层
            ds_predict_divide = ds_predict_divide.droplevel(0)

            # 合并ds_alpha和ds_predict_divide
            df_temp = pd.concat([ds_alpha, ds_predict_divide], axis=1)
            df_temp.columns = ['alpha', 'group']

            # 使用新的临时df_temp DataFrame来计算每个组的平均alpha
            ds_alpha_group = df_temp.groupby(['tradingDate', 'group'])['alpha'].mean()

            list_ds_alpha_group.append(ds_alpha_group)
        # pdb.set_trace()
        selected_columns = [col for col in df.columns if col != 'objectivReturnName']
        group_alpha = pd.concat(list_ds_alpha_group, axis=1, keys=selected_columns)
        group_alpha.index.names = ['tradingDate', 'group']
        group_alpha_mean = group_alpha.groupby(level='group').mean()

        return testIC, corr, mean, std, testICIR, ic_cap_group_mean , group_alpha, group_alpha_mean

b = pd.read_hdf('f17.h5', 'ADJValue')

df2 = b[(b.index.get_level_values(0) >= 20220104) & (b.index.get_level_values(0) <= 20221129)]
df2 = df2.reset_index()
df2.columns = ['tradingDate', 'code', 'value'] # Rename the columns
df2 = df2.set_index(['tradingDate', 'code']) # Set the new index
df2.index = df2.index.set_levels(df2.index.levels[0].astype(int), level=0)
df2.index = df2.index.set_levels(df2.index.levels[1].astype(int), level=1)


a ='BiGRU_HT_Residual_1.xlsx'
predictY = pd.read_excel(a,index_col=[0,1],sheet_name='factor')
predictY.index = predictY.index.set_levels(predictY.index.levels[0].astype(int), level=0)
predictY.index = predictY.index.set_levels(predictY.index.levels[1].astype(int), level=1)

index_intersection = predictY.index.intersection(df2.index)
predictY = predictY.reindex(index_intersection)
df2 = df2.reindex(index_intersection)
# pdb.set_trace()
df = pd.concat([predictY, df2], join='inner', axis=1)
weights = [1,1]
df = df.dot(weights)
    
df = df.reset_index()
df.columns = ['tradingDate', 'code', 'value'] # Rename the columns
df = df.set_index(['tradingDate', 'code']) # Set the new index
df.index = df.index.set_levels(df.index.levels[0].astype(int), level=0)
df.index = df.index.set_levels(df.index.levels[1].astype(int), level=1)


label = pd.read_hdf('label_ret20.h5', 'data')
df0 = label.to_frame()
df0 = df0.reset_index(level=[0,1]) # Reset all levels of the index
df0.columns = ['tradingDate', 'code', 'objectivReturnName'] # Rename the columns
df0 = df0.set_index(['tradingDate', 'code']) # Set the new index
df0.index = df0.index.set_levels(df0.index.levels[0].astype(int), level=0)
df0.index = df0.index.set_levels(df0.index.levels[1].astype(int), level=1)
testY = df0

df_cap = pd.read_hdf("CAP.h5", 'RawValue')
# df_cap['tradingDate'] = df_cap['tradingDate'].astype(int)
df_cap['code'] = df_cap['code'].astype(int)


testIC, corr, mean, std, testICIR, ic_cap_group_mean , group_alpha, group_alpha_mean =calPerformance(df, testY, df_cap, num_cap=5, num_group=10)
# pdb.set_trace()

fileName = a
with pd.ExcelWriter(fileName, mode='a', engine='openpyxl') as writer:
        testIC.mean().to_excel(writer, sheet_name='RetIC')
        # testIC.mean().to_excel(writer, sheet_name='testIC.mean()')
        
        
        testICIR.to_excel(writer, sheet_name='icir')
        
        ic_cap_group_mean.to_excel(writer, sheet_name='ic_cap_group_mean')
        
        # group_alpha.to_excel(writer, sheet_name='group_alpha')
        group_alpha_mean.to_excel(writer, sheet_name='group_alpha_mean')