# coding: utf-8

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from package.utils import Explore
from package.utils import ARIMAPredict
from package.utils import Evalutaion
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')


def str2tuple(string):
    leaf = string.replace('\"', "").replace('\'', "").replace(' ', "").replace('(', "").replace(')', "").split(',')
    return ('&').join(leaf)


def arima_train(leaf, ts1, ts2, d):
    arima = ARIMAPredict(leaf)
    # 训练模型
    arima.train(ts1, d, select=True) # 2, 3
    # 预测
    ts_pred = ts2
    if d == 1:
        ts_pred = ts2.diff().dropna()
    predict = arima.predict(ts_pred)
    # 保存模型
    arima.save('../')

    if d == 1:
        predict = predict + ts2[:-d]

    # 可视化预测结果
    arima.plot_results(ts2[d:], predict)
    return predict

def main():
    # 加载前两周叶子元素统计信息
    leafFirst2Week = pd.read_csv('../result/leaves/leafCountTrainStatistic.csv')
    leafFirst2Week = leafFirst2Week[leafFirst2Week['diff'] == 1]
    leafFirst2Week['leaf'] = leafFirst2Week['leaf'].apply(str2tuple)
    leaves = leafFirst2Week['leaf'].tolist()

    fileFormat = '../result/leaves/leaves_KPISet_smothing/%s.csv'
    train_colName = 'true'
    suffix = '_smooth' if train_colName == 'smoothed' else ''

    freq_col = 'count'
    diff_col = 'diff' + suffix
    isWhiteNoise_col = 'isWhiteNoise' + suffix
    trainNum = 2016

    predict_colName = 'pred' + suffix

    # 记录日志
    error_path = '../logs/error_log2.csv'
    success_path = '../logs/success_log2.csv'

    if os.path.exists(error_path):
        error_log = pd.read_csv(error_path)['leaf'].tolist()
    else:
        error_log = []

    if os.path.exists(success_path):
        success_log = pd.read_csv(success_path)['leaf'].tolist()
    else:
        success_log = []

    for leaf in tqdm(leaves):
        print(leaf)
        if leaf in error_log or leaf in success_log:
            continue
        leaf_row = leafFirst2Week[leafFirst2Week['leaf'] == leaf]
        diff = leaf_row[diff_col].values[0]
        freq = leaf_row[freq_col].values[0]
        isWhiteNoise = leaf_row[isWhiteNoise_col].values[0]

        ts = pd.read_csv(fileFormat % leaf)
        ts1 = ts[train_colName][:trainNum]  # 训练数据
        ts2 = ts[train_colName][trainNum:]  # 测试数据

        # 如果频率小于400 或 差分级数大于1 或 白噪声，直接返回ts1
        if freq < 400 or diff > 1 or isWhiteNoise:
            predict = ts1
            success_log.append(leaf)
            continue
        else:
            # 其他情况下
            try:
                predict = arima_train(leaf, ts1, ts2, diff)
                success_log.append(leaf)
            except:
                predict = ts1
                error_log.append(leaf)
            success_log.append(leaf)

        ts[predict_colName] = pd.Series([0] * (trainNum + diff) + list(predict))
        ts.to_csv(fileFormat % leaf, index=False)

    error_log = pd.DataFrame(error_log, columns=['leaf'])
    error_log.to_csv(error_path, index=False)

    success_log = pd.DataFrame(success_log, columns=['leaf'])
    success_log.to_csv(success_path, index=False)

    # 评估
    mae_sum = 0
    for leaf in tqdm(leaves):
        ts = pd.read_csv(fileFormat % leaf)
        ts_true = ts[train_colName][trainNum + diff:]
        ts_predict = ts[predict_colName][trainNum + diff:]
        eva = Evalutaion()
        mae_sum += eva.evaluate(ts_true, ts_predict, 'MAE', )['MAE']

    print(mae_sum)
    print(len(leafFirst2Week))
    print('mae: %f' % (mae_sum / len(leafFirst2Week)))

if __name__ == "__main__":
    main()
