if __name__ == '__main__':
    import pickle as pickle
    import logging
    from NNModel import RecurrentAndNeural
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
    with open('../HT90dataSet_Residual.pkl', 'rb') as file:
        dataSets = pickle.load(file)
    modelDate = 20211230
    # modelDate = 20201230
    tSets, vSets = RecurrentAndNeural.getDataSets(modelDate, dataSets, 0.2)   #获取直到‘modelDate’前25天的data，并按股票随机区分训练集和验证集
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inputSize = tSets[0][1].shape[2]
    weakNum = 16
    model = RecurrentAndNeural(input_size=inputSize,
                               hidden_size=inputSize,
                               fc_nn1=inputSize,
                               num_factors=weakNum).to(device)
    model.init()
    model.checkPoint['modelName'] = 0
    # model.checkPoint['factorParaDic'] = factorParaDic
    model.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # testDataSets = RecurrentAndNeural.getPeriodDataSets(20210104, 20211223, dataSets)
    testDataSets = RecurrentAndNeural.getPeriodDataSets(20220104, 20221130, dataSets)
    checkPoint = RecurrentAndNeural.trainAndEvaluateModel(model, tSets, vSets, 100)
    testPredictY = model.predictModel(testDataSets)
    # YDataSet = pd.read_hdf('labels.h5', 'data')
    YDataSet = pd.read_hdf('../YDataSets20.h5', 'data')
    testIC, testCorr, testMean, testStd = RecurrentAndNeural.calPerformance(testPredictY, YDataSet)

    print(testIC.mean())
    np.fill_diagonal(testCorr.values, 0)
    testCorr = np.abs(testCorr)

    # trainTorchSets = model.getPeriodDataSets(20071230, 20201230, dataSets)
    trainTorchSets = model.getPeriodDataSets(20071230, 20211230, dataSets)
    trainPredictY = model.predictModel(trainTorchSets)
    trainIC, trainCorr, trainMean, trainStd = RecurrentAndNeural.calPerformance(trainPredictY, YDataSet)
    np.fill_diagonal(trainCorr.values, 0)
    trainCorr = np.abs(trainCorr)

    fileName = f'BiGRU_HT_Residual_3.xlsx'
    with pd.ExcelWriter(fileName) as writer:
        trainIC.to_excel(writer, sheet_name='trainIC')
        trainIC.mean().to_excel(writer, sheet_name='trainICmean')       
        trainCorr.to_excel(writer, sheet_name='trainCorr')
        trainCorr.mean().to_excel(writer, sheet_name='trainCorr.mean()')
        testIC.to_excel(writer, sheet_name='testIC')
        testIC.mean().to_excel(writer, sheet_name='testIC.mean()')
        testCorr.to_excel(writer, sheet_name='testCorr')
        testCorr.mean().to_excel(writer, sheet_name='testCorr.mean()')

        testMean.to_excel(writer, sheet_name='factor')

 

