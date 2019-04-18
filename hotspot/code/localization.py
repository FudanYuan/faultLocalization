## coding: utf-8
import pandas as pd
import numpy as np
import time
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

from package.utils import KPIPoint
from package.utils import KPISet
from package.utils import Transformer
from package.HotSpot import HotSpot

def valid():
    #### 加载数据集
    # kSet_pred = KPISet({}, {})
    # kSet_pred.load('../result/metadata/KPISetValidPredict')
    # kSet_pred.test()

    # 使用前一周数据填充第二周数据
    # 读取源数据
    timestamp_start = 1535731200
    timestamp = 1536394500
    timestamp_interval = 5 * 60

    # 当前周数据
    file_path = '../2019AIOps_data_valid/'
    kSet = Transformer().transformKPIData2KPISet(file_path, timestamp, timestamp, timestamp_interval)
    # kSet.test()

    # 填充预测值
    leaf_true = []
    leaf_pred = []
    for leaf in kSet._KPIPoints[timestamp]._leaf:
        index = (timestamp - timestamp_start) / timestamp_interval
        ts = pd.read_csv('../result/leaves/result_arima/success/%s.csv' % ('&').join(leaf))
        predict = ts['pred'][index]
        kSet._KPIPoints[timestamp]._leaf[leaf][1] = predict
        leaf_true.append(kSet._KPIPoints[timestamp]._leaf[leaf][0])
        leaf_pred.append(predict)
        # print(('&').join(leaf), kSet._KPIPoints[timestamp]._leaf[leaf])

    # plt.figure(figsize=(40, 10))
    # plt.plot(leaf_true, label="true", color='green')
    # plt.plot(leaf_pred, label="predicted")
    # plt.legend()
    # plt.show()

    #### 读取异常时间戳
    outlier = pd.read_csv('../Anomalytime_data_valid.csv')
    timestamps = outlier['timestamp'].tolist()
    ps_threshold = 0.99
    ep_threshold = 0.01
    max_iter = 10

    res = {}
    res['timestamp'] = []
    res['detect_set'] = []
    sTime = time.time()
    for timestamp in tqdm(timestamps):
        ts = timestamp / 1000
        kPoint = kSet._KPIPoints[ts]
        layer_max = len(kPoint._attribute_names)
        hotSpot = HotSpot(kPoint, layer_max, ps_threshold, ep_threshold, max_iter)
        rootCauseSet = hotSpot.find_root_cause_set_revised()
        print(rootCauseSet[0][0])
        res['timestamp'].append(timestamp)
        res['detect_set'].append(list(rootCauseSet[0][0]))
        break

    eTime = time.time()
    print('runtime %fs' % (eTime - sTime))

    print(res)

    # 保存文件
    res = pd.DataFrame(res)
    res = res.sort_values(by='timestamp').reset_index(drop=True)
    res = res.merge(outlier, on='timestamp', how='left')
    res.to_csv('../result/root_cause_set_valid.csv', index=False)

    # 评估
    TP = 0
    FN = 0
    FP = 0
    for ts in res['timestamp'].tolist():
        root_cause = res[res['timestamp'] == ts]['real_set']
        root_cause_cal = res[res['timestamp'] == ts]['detect_set']
        tmp = 0
        for s1 in root_cause:
            for s2 in root_cause_cal:
                if len(s1) == len(s2) and len(set(s1).intersection(set(s2))) == len(s1):
                    tmp += 1
                    break
        TP += tmp
        FN += len(root_cause) - tmp
        FP += len(root_cause_cal) - tmp
    if TP == 0:
        TP += 1
    print(TP, FP, FN)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    FScore = (2 * Precision * Recall) / (Precision + Recall)
    print('F-score = %f' % FScore)

def valid2():
    #### 加载数据集
    # kSet_pred = KPISet({}, {})
    # kSet_pred.load('../result/metadata/KPISetValidPredict')
    # kSet_pred.test()

    # 使用前一周数据填充第二周数据
    # 读取源数据
    timestamp = 1536394500
    timestamp_interval = 5 * 60
    T = timestamp_interval * 288 * 7

    # 前一周数据
    file_path = '../2019AIOps_data/'
    kSet_train = Transformer().transformKPIData2KPISet(file_path, timestamp - T, timestamp - T, timestamp_interval)
    # kSet_train.test()

    # 当前周数据
    file_path = '../2019AIOps_data_valid/'
    kSet = Transformer().transformKPIData2KPISet(file_path, timestamp, timestamp, timestamp_interval)
    # kSet.test()

    # 填充预测值
    leaf_true = []
    leaf_pred = []
    for leaf in kSet._KPIPoints[timestamp]._leaf:
        predict = 0
        if leaf in kSet_train._KPIPoints[timestamp-T]._leaf:
            predict = kSet_train._KPIPoints[timestamp-T]._leaf[leaf][0]
        kSet._KPIPoints[timestamp]._leaf[leaf][1] = predict
        leaf_true.append(kSet._KPIPoints[timestamp]._leaf[leaf][0])
        leaf_pred.append(predict)
        # print(('&').join(leaf), kSet._KPIPoints[timestamp]._leaf[leaf])

    # plt.figure(figsize=(40, 10))
    # plt.plot(leaf_true, label="true", color='green')
    # plt.plot(leaf_pred, label="predicted")
    # plt.legend()
    # plt.show()

    #### 读取异常时间戳
    outlier = pd.read_csv('../Anomalytime_data_valid.csv')
    timestamps = outlier['timestamp'].tolist()
    ps_threshold = 0.99
    ep_threshold = 0.01
    max_iter = 10

    res = {}
    res['timestamp'] = []
    res['detect_set'] = []
    sTime = time.time()
    for timestamp in tqdm(timestamps):
        ts = timestamp / 1000
        kPoint = kSet._KPIPoints[ts]
        layer_max = len(kPoint._attribute_names)
        hotSpot = HotSpot(kPoint, layer_max, ps_threshold, ep_threshold, max_iter)
        rootCauseSet = hotSpot.find_root_cause_set_revised()
        print(rootCauseSet[0][0])
        res['timestamp'].append(timestamp)
        res['detect_set'].append(list(rootCauseSet[0][0]))
        break

    eTime = time.time()
    print('runtime %fs' % (eTime - sTime))

    print(res)

    # 保存文件
    res = pd.DataFrame(res)
    res = res.sort_values(by='timestamp').reset_index(drop=True)
    res = res.merge(outlier, on='timestamp', how='left')
    res.to_csv('../result/root_cause_set_valid.csv', index=False)

    # 评估
    TP = 0
    FN = 0
    FP = 0
    for ts in res['timestamp'].tolist():
        root_cause = res[res['timestamp'] == ts]['real_set']
        root_cause_cal = res[res['timestamp'] == ts]['detect_set']
        tmp = 0
        for s1 in root_cause:
            for s2 in root_cause_cal:
                if len(s1) == len(s2) and len(set(s1).intersection(set(s2))) == len(s1):
                    tmp += 1
                    break
        TP += tmp
        FN += len(root_cause) - tmp
        FP += len(root_cause_cal) - tmp
    if TP == 0:
        TP += 1
    print(TP, FP, FN)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    FScore = (2 * Precision * Recall) / (Precision + Recall)
    print('F-score = %f' % FScore)

def test():
    #### 加载数据集
    kSet_pred = KPISet({}, {})
    kSet_pred.load('../result/metadata/KPISetTestPredict2')
    # kSet_pred.test()
    #### 读取异常时间戳
    outlier = pd.read_csv('../Anomalytime_data_test1.csv')
    outlier = outlier['timestamp'].tolist()
    ps_threshold = 0.98
    ep_threshold = 0.01
    max_iter = 10

    res = {}
    res['timestamp'] = []
    res['set'] = []
    sTime = time.time()
    for timestamp in tqdm(outlier):
        ts = timestamp / 1000
        kPoint = kSet_pred._KPIPoints[ts]
        layer_max = len(kPoint._attribute_names)
        hotSpot = HotSpot(kPoint, layer_max, ps_threshold, ep_threshold, max_iter)
        rootCauseSet = hotSpot.find_root_cause_set_revised()
        res['timestamp'].append(timestamp)
        sets = []
        for ele in rootCauseSet[0][0]:
            sets.append("&".join(ele))
        res['set'].append(';'.join(sets))
        break
    eTime = time.time()
    print('runtime %fs' % (eTime - sTime))
    res = pd.DataFrame(res)
    res.to_csv('../result/submit%s.csv' % time.strftime("%Y%m%d%H%M%S", time.localtime(eTime)), index=False)

if __name__ == "__main__":
    valid()
    # valid2()
    # test()