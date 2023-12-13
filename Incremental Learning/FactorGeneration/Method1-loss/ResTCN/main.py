
if __name__ == '__main__':
    import pickle as pickle
    import logging
    from NNModel import RESTCN
    import torch
    import gzip
    import pandas as pd
    import numpy as np
    import  datetime

    FORMAT = "%(asctime)s %(thread)d %(message)s"
    logging.basicConfig(level=logging.INFO,filename='app.log',
                        format='%(asctime)-6s:%(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s', )


    # fileName = 'dataSets.pkl.gz'
    # f = gzip.open(fileName, 'rb')
    # dataSets = pickle.load(f)
    # f.close()
    # with open('XdataSet1.pkl', 'rb') as file: #RawBar数据，将强因子放在risk
    #     dataSets = pickle.load(file)
    with open('../HT90dataSet_280W.pkl', 'rb') as file: #17数据，将强因子放在risk
        dataSets = pickle.load(file)

    # modelDate = 20211230 #20190221开始
    modelDate = 20201231
    tSets, vSets = RESTCN.getDataSets(modelDate, dataSets, 0.2)   #获取直到‘modelDate’前25天的data，并按股票随机区分训练集和验证集
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inputSize = tSets[0][1].shape[2]
    weakNum = 16
    model = RESTCN(num_inputs=inputSize,
                   num_channels=[64,64,64],
                   output_size= weakNum,
                   kernel_size=3,
                   dropout=0.2,
                   Length=[3,4,5]).to(device)
    model.init()
    model.checkPoint['modelName'] = 0
    # model.checkPoint['factorParaDic'] = factorParaDic
    model.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # testDataSets = RESTCN.getPeriodDataSets(20220104, 20221130, dataSets) #测试集 目前20220104-20221130
    testDataSets = RESTCN.getPeriodDataSets(20210104, 20211223, dataSets)
    checkPoint = RESTCN.trainAndEvaluateModel(model, tSets, vSets, 100)
    testPredictY , testWeak = model.predictModel(testDataSets)
    # YDataSet = pd.read_hdf('labels.h5', 'data')
    YDataSet = pd.read_hdf('../YDataSets20.h5', 'data') #新标签
    testIC, testCorr, testMean, testStd = RESTCN.calPerformance(testPredictY, YDataSet) #强弱因子合成后的因子的IC
    weakIC, weakCorr, weakMean, weakStd, weight = model.calPerformance1(testWeak, YDataSet)  #弱因子的IC，Corr

    print(testIC.mean())
    np.fill_diagonal(testCorr.values, 0)
    testCorr = np.abs(testCorr)
    np.fill_diagonal(weakCorr.values, 0)
    weakCorr = np.abs(weakCorr)

    # trainTorchSets = model.getPeriodDataSets(20190211, 20211230, dataSets) #训练集
    trainTorchSets = model.getPeriodDataSets(20190211, 20201231, dataSets) #训练集
    trainPredictY , _ = model.predictModel(trainTorchSets)
    trainIC, trainCorr, trainMean, trainStd = RESTCN.calPerformance(trainPredictY, YDataSet)
    np.fill_diagonal(trainCorr.values, 0)
    trainCorr = np.abs(trainCorr)

    fileName = f'ResTCN_HT_2_21.xlsx'
    with pd.ExcelWriter(fileName) as writer:
        trainIC.to_excel(writer, sheet_name='trainIC')
        trainIC.mean().to_excel(writer, sheet_name='trainICmean')       
        # trainCorr.to_excel(writer, sheet_name='trainCorr')
        # trainCorr.mean().to_excel(writer, sheet_name='trainCorr.mean()')

        testIC.to_excel(writer, sheet_name='testIC')
        testIC.mean().to_excel(writer, sheet_name='testIC.mean()')
        # testCorr.to_excel(writer, sheet_name='testCorr')
        # testCorr.mean().to_excel(writer, sheet_name='testCorr.mean()')
        testMean.to_excel(writer, sheet_name='factor')
        weakMean.to_excel(writer, sheet_name='weak_factor')
        # weakIC.to_excel(writer, sheet_name='weakIC')
        weakIC.mean().to_excel(writer, sheet_name='weakIC.mean()')       
        # weakCorr.to_excel(writer, sheet_name='weakCorr')
        weakCorr.mean().to_excel(writer, sheet_name='weakCorr.mean()')
        pd.Series(weight).to_excel(writer, sheet_name='weights')  #各因子权重，最后一维是强因子权重
