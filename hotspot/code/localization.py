## coding: utf-8
import pandas as pd
import numpy as np
import time
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

from package.utils import KPIPoint
from package.utils import KPISet
from package.HotSpot import HotSpot

def valid():
    #### 加载数据集
    kSet_pred = KPISet({}, {})
    kSet_pred.load('../result/metadata/KPISetValidPredict')
    # kSet_pred.test()
    #### 读取异常时间戳
    outlier = pd.read_csv('../Anomalytime_data_valid.csv')
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
    eTime = time.time()
    print('runtime %fs' % (eTime - sTime))
    res = pd.DataFrame(res)
    res.to_csv('../result/root_cause_set_valid%s.csv' % time.strftime("%Y%m%d%H%M%S", time.localtime(eTime)), index=False)


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
    eTime = time.time()
    print('runtime %fs' % (eTime - sTime))
    res = pd.DataFrame(res)
    res.to_csv('../result/submit%s.csv' % time.strftime("%Y%m%d%H%M%S", time.localtime(eTime)), index=False)

if __name__ == "__main__":
    # valid()
    test()