import datetime
import time

import pandas as pd
import math
import logging
from torch.nn.utils import weight_norm
import os
from torch import nn
import  torch
import pdb
import numpy as np
def zscore_norm(tensor, dim=0):
    mean = tensor.mean(dim=dim, keepdim=True)
    std = tensor.std(dim=dim, keepdim=True)
    return (tensor - mean) / (std + 1e-5) 


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # 如果输入和输出通道数不同，则通过1x1卷积匹配它们
        self.match_channels = None
        if in_channels != out_channels:
            self.match_channels = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 如果需要，应用1x1卷积以匹配残差连接的通道数
        if self.match_channels:
            residual = self.match_channels(residual)
        
        out += residual
        out = self.relu(out)
        return out


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.LeakyReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout, length):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = length**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)[..., -1]
class RESTCN(nn.Module):
   
    def __init__(self, num_inputs, num_channels, output_size, kernel_size, dropout, Length):
        super(RESTCN, self).__init__()
        self.resnet0 = ResidualBlock(num_inputs,num_channels[0])
        resnet_layers = [ResidualBlock(num_channels[0], num_channels[0], kernel_size=3) for _ in range(2)]
        self.resnet = nn.Sequential(*resnet_layers)
        

        self.tcn1 = TemporalConvNet(num_channels[0], num_channels, kernel_size, dropout, Length[0])
        self.tcn2 = TemporalConvNet(num_channels[0], num_channels, kernel_size, dropout, Length[1])
        self.tcn3 = TemporalConvNet(num_channels[0], num_channels, kernel_size, dropout, Length[2])

        self.out = nn.Sequential(
            nn.BatchNorm1d(num_channels[-1]*len(Length)),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(num_channels[-1]*len(Length), output_size),
            nn.BatchNorm1d(output_size) 
        )


        # self.final_linear_layer = nn.Linear(num_factors+1, 1)    
        # self.input_size = input_size
        # self.hidden_size = hidden_size
        

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.fixed_weights = torch.ones(output_size, device=self.device) *0.1
        self.fixed_weights.requires_grad = False

        self.last_weight = nn.Parameter(torch.tensor([1.0], device=self.device))




    def forward(self, x, risk):
        # pdb.set_trace()
        x = x.transpose(1,2)
        x = self.resnet0(x)
        x = self.resnet(x)
        x = zscore_norm(x,dim=0)
        x1 = self.tcn1(x)
        x2 = self.tcn2(x)
        x3 = self.tcn3(x)
        x1 = zscore_norm(x1,dim=0)
        x2 = zscore_norm(x2,dim=0)
        x3 = zscore_norm(x3,dim=0)

        x_out = torch.cat([x1,x2,x3],dim=1)
        weakFactors = self.out(x_out)        
        # pdb.set_trace()
        # gru_out = self.rnn(x)[0][:, -1, :]


        # weakFactors = self.out(gru_out) #弱因子
        # r = risk.unsqueeze(1)
        # concatenated_factors = torch.cat((weakFactors, r), dim=1) # 拼接因子
        # final_output = self.final_linear_layer(concatenated_factors) #线性层

        combined_weights = torch.cat([self.fixed_weights, self.last_weight], dim=0)
        final_output = torch.sum(combined_weights * torch.cat((weakFactors, risk.unsqueeze(1)), dim=1), dim=1, keepdim=True)

        return final_output, weakFactors


    def init(self):
        self.toleranceMinLoss = 1000
        self.toleranceNum = 0
        self.earyStopModel = None
        self.checkPoint = {}
        self.optimizer = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 验证集合容忍次数
        self.tolerance_max = 3
        self.toleranceNum = 0
        self.minLr = 0.000016
        self.toleranceMinLoss = 100
        self.modelPath = ''
        self.trainLossList = []
        self.evalLossList = []
        self.earyStopModel = None
        self.lr_scheduler = None
        self.checkPoint = {}
        torch.set_default_tensor_type(torch.FloatTensor)


    def trainModel(self, dataSets):
        lossList = []
        l2lossList = []
        icLossList = []
        self.train()

        for batchNum, X, y, codes, risk in dataSets:

            xValues, weakFactors= self(X.to(self.device), risk.to(self.device))
            yValues = y.to(self.device)

            ICloss = ToolFactor.torch_ic_weighted(xValues, yValues)  
            l2loss = ToolFactor.torch_similarity_penalty(weakFactors, penalty_type='l2')
            # 检查是否有nan值
            if math.isnan(ICloss) or math.isnan(l2loss):
                pdb.set_trace()
            # print(self.last_weight)
            if torch.isnan(l2loss).any():  # 检测l2loss是否包含NaN
                loss = -ICloss
            else:
                loss = l2loss * 0.1 - ICloss

            self.optimizer.zero_grad()
            loss.backward()
            # print(self.last_weight.grad)
            # for name, param in self.rnn.named_parameters():
            #     torch.nn.utils.clip_grad_norm_(param, 4.0)
            torch.nn.utils.clip_grad_norm_(self.parameters(), 4.0)
            self.optimizer.step()
            l2lossList.append(np.round(l2loss.item(), 6))
            lossList.append(np.round(loss.item(), 6))
            icLossList.append(np.round(ICloss.item(), 6))
        # print('Final linear layer weights: ', self.final_linear_layer.weight.data)
        print('train  loss : ' + str(np.mean(lossList))+' l2loss : ' + str(np.mean(l2lossList))+' icloss : ' + str(np.mean(icLossList)))


    def evalulateModel(self, dataSets):
        import copy
        self.eval()
        lossList = []
        l2lossList = []
        icLossList = []
        riskLossList = []
        with torch.no_grad():
            for batchNum, X, y, codes, risk  in dataSets:
                xValues, weakFactors= self(X.to(self.device), risk.to(self.device))
                yValues = y.to(self.device)

                ICloss = ToolFactor.torch_ic_weighted(xValues, yValues)  
                l2loss = ToolFactor.torch_similarity_penalty(weakFactors, penalty_type='l2')
                
                
                loss = l2loss * 0.1 - ICloss  


                lossList.append(np.round(loss.item(), 6))
                l2lossList.append(np.round(l2loss.item(), 6))
                icLossList.append(np.round(ICloss.item(), 6))
        meanLoss = np.mean(lossList)
        print('evalulate loss: ' + str(meanLoss) + ' l2loss : ' + str(np.mean(l2lossList)) + '  icloss: ' + str(
            np.mean(icLossList)) + ' riskLoss ' + str(np.mean(riskLossList)))


        if meanLoss < self.toleranceMinLoss:

            self.toleranceMinLoss = meanLoss
            self.toleranceNum = 0
            self.checkPoint['net'] = copy.deepcopy(self.state_dict())  # 不知道是否需要深度复制
            self.checkPoint['optimizer'] = self.optimizer.state_dict()
            self.checkPoint['loss'] = self.toleranceMinLoss
        else:
            self.toleranceNum = self.toleranceNum + 1
            if self.toleranceNum >= self.tolerance_max:
                currentLr = self.optimizer.param_groups[0]['lr']
                if currentLr == self.minLr:
                    logging.info('学习率已经达到最小提前停止')
                    return True
                else:
                    # 可以设置一套动态，不断减小学习率的高效算法：例如到学习率小于一定数值时，最大容忍次数可以降低，这样免得浪费时间。同时又可以让模型不断减小学习率达到极限
                    nextLr = currentLr * 0.5 if currentLr * 0.5 > self.minLr else self.minLr
                    for params in self.optimizer.param_groups:
                        params['lr'] = nextLr
                    self.toleranceNum = 0

                    self.load_state_dict(self.checkPoint['net'])
                    logging.info('下降学习率到 {} ,模型参数返回到上次最优参数'.format(nextLr))

        return False


    def predictModel(self, dataSets):
        self.eval()
        result = None
        result1 = None
        outputs = []
        outputs1 = []
        indexs = None
        indexs1 = None
        dateArrays = []
        dateArrays1 = []
        codesArrays = []
        codesArrays1 = []
        with torch.no_grad():
            for batchNum, X ,y,codes,risk in dataSets:
                # 注意将risk也传递给模型
                output , weak = self(X.to(self.device), risk.to(self.device))

                outputs.append(output)
                dateArrays.extend([batchNum] * len(codes))
                codesArrays.extend(codes)

                outputs1.append(weak)
                dateArrays1.extend([batchNum] * len(codes))
                codesArrays1.extend(codes)

        indexs = pd.MultiIndex.from_tuples(zip(dateArrays, codesArrays))
        torchResuts = torch.vstack(outputs)
        result = pd.DataFrame(data=torchResuts.detach().cpu().numpy(), index=indexs)
        result = result.sort_index()
        result.index.names = ['tradingDate', 'code']
        
        indexs1 = pd.MultiIndex.from_tuples(zip(dateArrays1, codesArrays1))
        torchResuts1 = torch.vstack(outputs1)
        result1 = pd.DataFrame(data=torchResuts1.detach().cpu().numpy(), index=indexs1)
        result1 = result1.sort_index()
        result1.index.names = ['tradingDate', 'code']        
        
        
        
        
        return result, result1
        
    
   
    def trainAndEvaluateModel(model, tSets, vSets, epoches):
        # if epoches==0:
        # print('第一个模型不做')
        # return
        earylyStop = False
        for epoch in range(epoches):
            # print(model.checkPoint.keys())
            print('模型' + str(model.checkPoint['modelName']) + '轮次' + str(epoch))
            logging.info('train 模型{} epoch {} '.format(model.checkPoint['modelName'], epoch))
            model.trainModel(tSets)
            earylyStop = model.evalulateModel(vSets)
            # logging.info('evalulate 模型{} epoch {} loss: {} '.format(model.checkPoint['modelName'], epoch, model.checkPoint['loss']))
            if earylyStop:
                logging.info('使用早停最佳模型')
                break
        if not earylyStop:
            logging.info('最大训练轮次到，用最后的模型')

        return model.checkPoint
    
    @staticmethod
    def getDataSets(modelDate,torchDataSets, tRatio):
        torch.set_default_tensor_type(torch.FloatTensor)

        trainDataSets = []
        validDataSets = []

        dataStoredMinDate = min(torchDataSets.keys())
        tradingDates=list(torchDataSets.keys())
        import random

        midBatchCodes = torchDataSets[modelDate][2].index.values
        batchSizeLen = len(midBatchCodes)
        selectedCodeInt = random.sample(range(0, batchSizeLen), int(batchSizeLen * (1 - tRatio)))

        selectedCodes = midBatchCodes[selectedCodeInt]
        modelPos=tradingDates.index(modelDate)
        for i in range(0, modelPos-25):
            currentDate = tradingDates[i]
            currentDataSets=torchDataSets[currentDate]
            currentCodeSeries=currentDataSets[2]
            traindCodesSeries=currentCodeSeries.loc[currentCodeSeries.index.isin(selectedCodes)]
            validCodesSeries=currentCodeSeries.loc[~currentCodeSeries.index.isin(selectedCodes)]
            traindLocation=list(traindCodesSeries.values)
            trainCodes=list(traindCodesSeries.index.values)
            trainXDataSets = currentDataSets[0][traindLocation]
            trainYDataSets = None if currentDataSets[1] is None else currentDataSets[1][traindLocation]
            trainRiskDataSets= None if currentDataSets[3] is None else currentDataSets[3][traindLocation]

            validLocation=list(validCodesSeries.values)
            validCodes = list(validCodesSeries.index.values)
            validXDataSets = currentDataSets[0][validLocation]
            validYDataSets = None if currentDataSets[1] is None else currentDataSets[1][validLocation]
            validRiskDataSets = None if currentDataSets[3] is None else currentDataSets[3][validLocation]

            trainDataSets.append([currentDate, trainXDataSets, trainYDataSets, trainCodes, trainRiskDataSets])
            validDataSets.append([currentDate, validXDataSets, validYDataSets, validCodes, validRiskDataSets])
        return trainDataSets, validDataSets
        # return self.changeDevice(trainDataSets),self.changeDevice(validDataSets)

    @staticmethod
    def getPeriodDataSets(startDate,endDate,torchDataSets):
        tradingDates=list(torchDataSets.keys())
        dataSets=[]
        torchDataSets = torchDataSets
        startDateS=min([d for d in tradingDates if d>=startDate])
        endDateS = max([d for d in tradingDates if d<= endDate])
        endPos = tradingDates.index(endDateS)
        startPos = tradingDates.index(startDateS)
        for i in range(startPos, endPos):
            currentDate = tradingDates[i]
            if currentDate in torchDataSets:
                dataSets.append([currentDate, torchDataSets[currentDate][0], torchDataSets[currentDate][1],  list(torchDataSets[currentDate][2].index.values), torchDataSets[currentDate][3]])
        return dataSets



    @staticmethod
    def calPerformance(predictY,testY):
        weakFactorColumns=list(predictY.columns)
        testY.name='objectivReturnName'
        objectivReturnName=testY.name
        df = pd.concat([predictY, testY], join='inner', axis=1)
        df['mean']=df[weakFactorColumns].mean(axis=1)
        calFactorNames=[]
        calFactorNames.extend(weakFactorColumns)
        calFactorNames.append('mean')
        df=df.sort_index()
        df.index.names = ['tradingDate', 'code']
        testIC = df.groupby("tradingDate").apply(
            lambda x: x[calFactorNames].apply(
                lambda y: y.corr(x[objectivReturnName], method='spearman')))
        corr = df[weakFactorColumns].corr(method ='pearson')
        mean=df[weakFactorColumns].mean(axis=1)
        std = df[weakFactorColumns].std()
        return testIC,corr,mean,std

    def calPerformance1(self, predictY, testY):  
        #计算弱因子加权后
        weakFactorColumns = list(predictY.columns)
        testY.name = 'objectivReturnName'
        objectivReturnName = testY.name
        df = pd.concat([predictY, testY], join='inner', axis=1)

    # # 获取整个权重向量，并转换为Numpy数组
        # weights = self.final_linear_layer.weight.data[0].cpu().numpy()

        # 获取整个权重向量，并转换为Numpy数组 正权重
        fixed_weights_np = self.fixed_weights.cpu().numpy()
        last_weight_np = self.last_weight.detach().cpu().numpy()
        weights = np.concatenate([fixed_weights_np, last_weight_np])

    

        # 用权重（除去最后一个元素）乘以每个因子，然后求和以得到加权平均
        df['weighted_mean'] = df[weakFactorColumns].values @ weights[:-1]

        calFactorNames = []
        calFactorNames.extend(weakFactorColumns)
        calFactorNames.append('weighted_mean')

        df = df.sort_index()
        df.index.names = ['tradingDate', 'code']
        testIC = df.groupby("tradingDate").apply(
            lambda x: x[calFactorNames].apply(
                lambda y: y.corr(x[objectivReturnName], method='spearman')))
        corr = df[weakFactorColumns].corr(method='pearson')
        mean = df[weakFactorColumns].mean(axis=1)
        std = df[weakFactorColumns].std()
        return testIC, corr, mean, std, weights


class ToolFactor(object):
    """
    该类主要保存因子处理过程中用到的主要工具和方法
    """
    @staticmethod
    def format_inputs(x: torch.Tensor, y: torch.Tensor, weights: torch.Tensor = None):
        x = x.view(-1, )
        y = y.view(-1, )
        if weights is None:
            weights = torch.ones(x.shape).to(x.device)
        weights = weights.view(-1, )
        weights /= weights.sum()
        return x, y, weights

    @staticmethod
    def torch_ccc_weighted(x: torch.Tensor, y: torch.Tensor, weights: torch.Tensor = None):
        x, y, weights = ToolFactor.format_inputs(x, y, weights)
        mean_x = torch.sum(x * weights)
        mean_y = torch.sum(y * weights)
        variance_x = torch.sum(weights * torch.pow((x - mean_x), 2))
        variance_y = torch.sum(weights * torch.pow((y - mean_y), 2))
        ccc = (torch.sum(weights * x * y) - mean_x * mean_y) / (variance_x + variance_y + (mean_y - mean_x) ** 2)
        return ccc

    @staticmethod
    def torch_ic_weighted(x: torch.Tensor, y: torch.Tensor, weights: torch.Tensor = None):

        x, y, weights = ToolFactor.format_inputs(x, y, weights)
        mean_x = torch.sum(x * weights)
        mean_y = torch.sum(y * weights)
        variance_x = torch.sum(weights * torch.pow((x - mean_x), 2))
        variance_y = torch.sum(weights * torch.pow((y - mean_y), 2))
        ic = torch.sum(weights * (x - mean_x) * (y - mean_y)) / torch.sqrt(variance_y * variance_x)
        return ic

    @staticmethod
    def torch_mse_weighted(x: torch.Tensor, y: torch.Tensor, weights: torch.Tensor = None):
        x, y, weights = ToolFactor.format_inputs(x, y, weights)
        return torch.sum(weights * torch.pow(x - y, 2))

    @staticmethod
    def torch_residual(factor, risk_factor, weights=None, return_param=False):
        if risk_factor is None:
            return factor
        if factor.ndim == 1:
            factor = factor.view(-1, 1)
        else:
            assert factor.ndim == 2
        if weights is None:
            _n = risk_factor.shape[0]
            weights = torch.ones((_n, _n)).to(factor.device)
        xw = risk_factor.t().matmul(weights)
        param = (xw.matmul(risk_factor)).inverse().matmul(xw).matmul(factor)
        residual = factor - risk_factor.matmul(param)
        if return_param:
            return residual, param
        return residual

    @staticmethod
    def torch_similarity_penalty(factor, penalty_type='l2'):
        # factor shape: N * d, N is stock_num, d is feature_num
        mat = torch.triu(torch.corrcoef(factor.t()), 1)
        n = mat.shape[0]
        n_element = n * (n - 1) / 2
        if penalty_type == 'l1':
            mat = mat.abs()
            return mat.sum() / n_element
        elif penalty_type == 'l2':
            mat = mat.pow(2)
            return torch.sqrt(mat.sum() / n_element)
        elif penalty_type == 'l4':
            mat = mat.pow(4)
            return  torch.pow(mat.sum() / n_element,1.0/4)
        elif penalty_type == 'max':
            mat = torch.triu(torch.corrcoef(factor.t()), diagonal=1).abs()
            return mat.max()





    @staticmethod
    def torch_l2_similarity_penalty(x):
        f = (x.shape[0] - 1) / x.shape[0]  # 修正系数,用来修正相关系数的计算
        x_reducemean = x - torch.mean(x, axis=0)
        numerator = torch.matmul(x_reducemean.T, x_reducemean) / x.shape[
            0]  # 得到分子(x-mean(x)) * (x-mean(x))'/N, x的协方差矩阵,15*15
        var_ = x.var(axis=0).reshape(x.shape[1], 1)  # 得到var(x),x的各个分量的方差,15维
        denominator = torch.sqrt(torch.matmul(var_, var_.T)) * f  # 得到分母sqrt(var(x)' * var(x)) ,15*15的矩阵
        corrcoef = numerator / denominator  # 得到相关系数矩阵
        # l2 = torch.linalg.norm(corrcoef)                                        # 得到相关系数矩阵l2范数
        # l2_limit = torch.tensor(np.sqrt(x.shape[1]))
        #
        # return (l2 - l2_limit)/torch.tensor(x.shape[1])

        l2 = torch.linalg.norm(corrcoef).pow(2)  # 得到相关系数矩阵l2范数
        l2_limit = torch.tensor(x.shape[1])

        return torch.sqrt((l2 - l2_limit) / torch.tensor(x.shape[1] * (x.shape[1] - 1)))

    @staticmethod
    def torch_reweight(label: torch.Tensor, t_half=1.8):
        """label越大，表示希望权重越大"""
        rank = torch.argsort(label, descending=True)
        _min = 0
        _max = len(rank) - 1
        rank: torch.Tensor = (rank - _min) / (_max - _min)
        weights = torch.exp2(-rank / t_half)
        weights /= weights.sum()
        return weights

    @staticmethod
    def reweight(label:pd.Series, t_half=1.8):
        """label越大，表示希望权重越大"""
        rank = label.rank()
        _min = 0
        _max = len(rank) - 1
        rank = (rank - _min) / (_max - _min)
        weights = np.exp2(-rank / t_half)
        weights /= weights.sum()
        return weights


    @staticmethod
    #对称正交
    def symmetric_orth(fc_x):
        fc_x1 = fc_x - torch.mean(fc_x, dim=0, keepdim=True)
        fc_cov = torch.chain_matmul(fc_x1.T, fc_x1)
        fc_eig_val, fc_eig_vec = torch.linalg.eigh(fc_cov)
        fc_mat_w = torch.chain_matmul(
        fc_eig_vec,
        torch.diag(1.0 / torch.sqrt(fc_eig_val)),
        fc_eig_vec.T
        )
        fc_res = torch.matmul(fc_x1, fc_mat_w)
        return fc_res
    @staticmethod
    #斯密特正交
    def schmidt_orth(fc_y, fc_x):

        fc_y1 = (fc_y - torch.mean(fc_y, dim=0, keepdim=True))/torch.std(fc_y, dim=0, keepdim=True)
        fc_x1 = (fc_x - torch.mean(fc_x, dim=0, keepdim=True))/torch.std(fc_x, dim=0, keepdim=True)

        fc_coefficient = torch.linalg.multi_dot(
        (torch.linalg.pinv(torch.linalg.multi_dot((fc_x1.T, fc_x1))),
        fc_x1.T,
        fc_y1)
        )
        fc_res = fc_y1 - torch.matmul(fc_x1, fc_coefficient)

        return fc_res

    @staticmethod
    def toTorch(df, Dimension=1):
        df = df.astype('float32')
        dataTorch = None
        if Dimension == 1:
            dataTorch = torch.from_numpy(df.fillna(0).values)
        elif Dimension == 2:
            columnNames = list(df.columns.values)
            dataTorch = torch.zeros([df.shape[0], len(columnNames)])
            for i in range(len(columnNames)):
                dataTorch[:, i] = torch.from_numpy(df[columnNames[i]].values)
        elif Dimension == 3:
            columnNames = list(df.columns.levels[0].values)
            tsNum = df[columnNames[0]].shape[1]
            dataTorch = torch.zeros([df.shape[0], tsNum, len(columnNames)])
            for i in range(len(columnNames)):
                dataTorch[:, :, i] = torch.from_numpy(df[columnNames[i]].values)
        return dataTorch

    @staticmethod
    def similarity_penalty(factor):
        # factor shape: N * d, N is stock_num, d is feature_num
        mat = torch.corrcoef(factor.t())
        n = mat.shape[0]  # feature_num
        n_element = n * (n - 1)
        mat = mat.pow(2).sum() - n

        return torch.sqrt(mat / n_element)



