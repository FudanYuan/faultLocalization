import numpy as np
import pandas as pd
from tqdm import tqdm
from package.utils import KPIPoint
from package.utils import KPISet

def transformKPIData2KPIPoint(filePath, timestamp):
    data = pd.read_csv(filePath + str(timestamp) + '.csv', header=None, names=['i', 'e', 'c', 'p', 'l', 'KPI'])
    data = data.drop(data[data.values=='unknown'].index,axis=0).reset_index(drop=True)
    data = data[data['KPI'] !=0].reset_index(drop=True)
    # 获取属性值
    attribute_list = {}
    attrs = list(data.columns)[:-1]
    for attr in attrs:
        attribute_list[attr] = sorted(data[attr].unique().tolist())
    # 获取叶子元素
    # leaf = dict(zip(zip(data['i'], data['e'], data['c'], data['p'], data['l']), data['KPI'])) # 主键相同时会覆盖
    leaf = {}
    for i in range(len(data)):
        element = tuple(data.loc[i][:-1])
        # if element in leaf:
        #    print('error')
        #    print(i, element, leaf[element])
        if element not in leaf:
            leaf[element] = [0, 0]
        leaf[element] = [leaf[element][0] + data.loc[i][-1], 0]
    return attribute_list, leaf

def transformKPIData2KPISet(filePath, timestamp_start, timestamp_end, timestamp_interval):
    kPoints = {}
    attr_list_all = {}
    for ts in tqdm(range(timestamp_strat, timestamp_end + timestamp_interval, timestamp_interval)):
        ts_file = ts * 1000
        attribute_list, leaf = transformKPIData2KPIPoint(filePath, ts_file)
        kPoints[ts] = KPIPoint(attribute_list, ts, leaf)
        for attr in attribute_list:
            if attr not in attr_list_all:
                attr_list_all[attr] = []
            attr_list_all[attr] = sorted(list(set(attr_list_all[attr]).union(set(attribute_list[attr]))))
    kSet = KPISet(attr_list_all, kPoints)
    return kSet

timestamp_strat = 1535731200
timestamp_end = 1536940500 #1535731200 + 5 * 60#
timestamp_interval = 5 * 60
file_path = '../2019AIOps_data/'
kSet = transformKPIData2KPISet(file_path, timestamp_strat, timestamp_end, timestamp_interval)
# kSet.test()
for attr in kSet._attribute_list:
    print(attr, len(kSet._attribute_list[attr]))
kSet.save('../result/metadata/KPISet')

timestamp_strat = 1536940800
timestamp_end = 1538150100 #1536940800 + 5 * 60#
timestamp_interval = 5 * 60
file_path = '../2019AIOps_data_test1/'
kSet_test = transformKPIData2KPISet(file_path, timestamp_strat, timestamp_end, timestamp_interval)
# kSet_test.test()
for attr in kSet_test._attribute_list:
    print(attr, len(kSet_test._attribute_list[attr]))
kSet_test.save('../result/metadata/KPISetTest')