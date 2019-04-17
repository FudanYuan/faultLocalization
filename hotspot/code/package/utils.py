# coding: utf-8
import itertools
import pickle
import numpy as np
import os
from sklearn.externals import joblib

'''
笛卡尔积
'''
class cartesian(object):
    def __init__(self):
        self._data_list = []
        self._res_list = []

    def add_data(self, data=[]):  # 添加生成笛卡尔积的数据列表
        self._data_list.append(data)

    def build(self):  # 计算笛卡尔积
        for item in itertools.product(*self._data_list):
            self._res_list.append(item)


'''
the data structure of KPIPoint
'''
class KPIPoint:
    def __init__(self, attribute_list, timestamp, leaf):
        # the attribute name list
        self._attribute_names = []
        # the attribute list in the 1st layer, .i.e the 1-d cuboids.
        self._attribute_list = attribute_list
        #  the timestamp of the PVSet
        self._timestamp = timestamp
        # the leaf value of the PV,
        # for example {"a1b2c3":[10,13]}
        # means the true PV of the leaf
        # element {a1, b2, c3} is 10
        # while the future value is 13
        self._leaf = leaf
        self._layers = {}
        # the total KPI value at this timestamp
        self._amount = [0, 0]
        self.get_attribute_names()

    # calculate the amount
    def calAmount(self):
        for key in self._leaf:
            self._amount[0] += self._leaf[key][0]
            self._amount[1] += self._leaf[key][1]
        return self._amount

    # get the attribute names from the _attribute_list
    def get_attribute_names(self):
        attrs = []
        for attr in self._attribute_list:
            attrs.append(attr)
        self._attribute_names = attrs

    # get the elements set of idth layer
    # layer: the ith layer
    def get_elements_set_by_layer(self, layer):
        if self._attribute_names == []:
            self.get_attribute_names()
        # get the idth element combinations set
        res = {}
        coms = itertools.combinations(range(len(self._attribute_names)), layer)
        leaves = self._leaf
        for com in coms:
            attr_com = tuple(np.array(self._attribute_names)[np.array(com)])
            res[attr_com] = {}
            for leaf in leaves:
                key = tuple(np.array(leaf)[np.array(com)])
                value = leaves[leaf]
                if (key,) not in res[attr_com]:
                    res[attr_com][(key,)] = value
                else:
                    res[attr_com][(key,)] = np.sum([res[attr_com][(key,)], value], axis=0).tolist()
        return res

    # get the elements set of idth layer
    # layer: the ith layer
    def get_elements_set_by_layer_with_prune(self, layer, parentSet):
        if len(parentSet) == 0:
            return self.get_elements_set_by_layer(layer)
        if self._attribute_names == []:
            self.get_attribute_names()
        # get the idth element combinations set
        res = {}
        coms = itertools.combinations(range(len(self._attribute_names)), layer)
        leaves = self._leaf
        remain_counter = 0
        for com in coms:
            attr_com = tuple(np.array(self._attribute_names)[np.array(com)])
            res[attr_com] = {}
            for leaf in leaves:
                flag = False
                for s in parentSet:
                    tmp = list(set(s).intersection(set(leaf)))
                    if len(tmp) != 0:
                        flag = True
                        break

                # prune
                if not flag:
                    # print('prune element', leaf)
                    continue

                key = tuple(np.array(leaf)[np.array(com)])
                value = leaves[leaf]

                if (key,) not in res[attr_com]:
                    res[attr_com][(key,)] = value
                else:
                    res[attr_com][(key,)] = np.sum([res[attr_com][(key,)], value], axis=0).tolist()
                remain_counter += 1
        return res

    # get the elements set of idth layer
    # layer: the ith layer
    def get_elements_set_by_layer2(self, layer):
        if self._attribute_names == []:
            self.get_attribute_names()
        # get the idth element combinations set
        res = {}
        coms = itertools.combinations(self._attribute_names, layer)
        for com in coms:
            res[com] = {}
            elements = self.get_elements_in_cuboid(com)
            for ele in elements:
                _, value = self.get_descendant_elements_ele(ele)
                if value == [0, 0]:  # if there exists no ele in the leaf elements, continue
                    continue
                res[com][(ele,)] = value
        return res

    # get the elements set of idth layer
    # layer: the ith layer
    def get_elements_set_by_layer_with_prune2(self, layer, parentSet):
        if len(parentSet) == 0:
            return self.get_elements_set_by_layer2(layer)
        if self._attribute_names == []:
            self.get_attribute_names()
        # get the idth element combinations set
        res = {}
        coms = itertools.combinations(self._attribute_names, layer)
        total_counter = 0
        remain_counter = 0
        for com in coms:
            print('com: ', com)
            res[com] = {}
            elements = self.get_elements_in_cuboid(com)
            for ele in elements:
                total_counter += 1
                # prune
                flag = False
                for s in parentSet:
                    tmp = list(set(s).intersection(set(ele)))
                    if len(tmp) != 0:
                        flag = True
                        break

                if not flag:
                    # print('prune element', ele)
                    continue

                _, value = self.get_descendant_elements_ele(ele)
                if value == [0, 0]:  # if there exists no ele in the leaf elements, continue
                    continue
                res[com][(ele,)] = value
                remain_counter += 1
        prune_rate = (1 - remain_counter / total_counter) * 100
        print('prune rate: %f%%' % prune_rate)
        return res

    # get all the combinations of given cuboid
    # if there are n elements in the cuboid,
    # then there are 2^n-1 combinations.
    def get_elements_coms_by_layer(self, layer):
        res = {}
        coms_elements_by_cuboid = {}
        coms = itertools.combinations(self._attribute_names, layer)
        for com in coms:
            res[com] = {}
            coms_elements = self.get_elements_coms_in_cuboid(com)
            coms_elements_by_cuboid[com] = coms_elements
            for ele_com in coms_elements:
                _, value = self.get_descendant_elements_coms(ele_com)
                res[com][ele_com] = value
        return res

    # get the descendant element
    # coms: combinations of elements
    def get_descendant_elements_coms(self, coms):
        elements = {}
        value = {}
        for leaf in self._leaf:
            for ele in coms:
                if (ele,) not in value:
                    value[(ele,)] = [0, 0]
                tmp = [0, 0]
                inter = list(set(ele).intersection(set(leaf)))
                if len(inter) == len(ele):
                    elements[leaf] = self._leaf[leaf]
                    tmp[0] += elements[leaf][0]
                    tmp[1] += elements[leaf][1]
                value[(ele,)][0] += tmp[0]
                value[(ele,)][1] += tmp[1]
        return elements, value

    # get the descendant element
    # p: parent element sigle
    def get_descendant_elements_ele(self, ele):
        elements = {}
        value = [0, 0]
        for leaf in self._leaf:
            inter = list(set(ele).intersection(set(leaf)))
            if len(inter) == len(ele):
                elements[leaf] = self._leaf[leaf]
                value[0] += elements[leaf][0]
                value[1] += elements[leaf][1]
        return elements, value

    # get the elements of given cuboid
    def get_elements_in_cuboid(self, cuboid):
        cart = cartesian()
        for ele in cuboid:
            attr_list = []
            for attr in self._attribute_list[ele]:
                attr_list.append(attr)
            cart.add_data(attr_list)
        cart.build()
        return cart._res_list

    # get the elements combinations of given cuboid
    def get_elements_coms_in_cuboid(self, cuboid):
        coms = {}
        elements = self.get_elements_in_cuboid(cuboid)
        for i in range(1, len(elements) + 1):
            for ele in itertools.combinations(elements, i):
                _, value = self.get_descendant_elements_coms(ele)
                coms[ele] = value
        return coms

    def test(self):
        print('timestamp is %d' % self._timestamp)
        print('leaf is ', self._leaf)
        print('attribute names list is ', self._attribute_names)


'''
the data structure of KPISet
'''
class KPISet:
    def __init__(self, attribute_list, KPIPoints):
        # the attribute name list
        self._attribute_names = []
        # the attribute list in the 1st layer, .i.e the 1-d cuboids.
        self._attribute_list = attribute_list
        # the list KPI points
        self._KPIPoints = KPIPoints
        # init the attribute name list
        self.get_attribute_names()

    # get the attribute names from the _attribute_list
    def get_attribute_names(self):
        attrs = []
        for attr in self._attribute_list:
            attrs.append(attr)
        self._attribute_names = attrs

    # get the attribute combinations of idth layer
    # t: timestamp
    # id: the idth layer
    def get_elements_set_by_layer(self, t, id):
        if self._attribute_names == []:
            self.get_attribute_names()
        return self._KPIPoints[t].get_elements_set_by_layer(id)

    # get the descendant element
    # t: timestamp
    # coms: parent element coms
    def get_descendant_elements_coms(self, t, coms):
        return self._KPIPoints[t].get_descendant_elements_coms(coms)

    # get the descendant element
    # t: timestamp
    # ele: parent element ele
    def get_descendant_elements_ele(self, t, ele):
        return self._KPIPoints[t].get_descendant_elements_ele(ele)

    # get the time series of a element
    def get_ts_ele(self, t1, t2, delta, ele):
        if len(ele) == len(self._attribute_names):
            return self.get_ts_leaf(t1, t2, delta, ele)
        else:
            return self.get_ts_not_leaf(t1, t2, delta, ele)

    # get the time serises of a leaf element
    def get_ts_leaf(self, t1, t2, delta, leaf):
        ts_true = []
        ts_pred = []
        for t in range(t1, t2 + delta, delta):
            if t not in self._KPIPoints:
                # print('error', '%d not exists' % t)
                break
            if leaf not in self._KPIPoints[t]._leaf:
                ts_true.append(0)
                ts_pred.append(0)
            else:
                ts_true.append(self._KPIPoints[t]._leaf[leaf][0])
                ts_pred.append(self._KPIPoints[t]._leaf[leaf][1])
        ts = {}
        ts['true'] = ts_true
        ts['pred'] = ts_pred
        return ts

    # get the time serises of a not-leaf element
    def get_ts_not_leaf(self, t1, t2, delta, ele):
        ts_true = []
        ts_pred = []
        for t in range(t1, t2 + delta, delta):
            if t not in self._KPIPoints:
                # print('error', '%d not exists' % t)
                break
            _, value = self.get_descendant_elements_ele(t, ele)
            ts_true.append(value[0])
            ts_pred.append(value[1])
        ts = {}
        ts['true'] = ts_true
        ts['pred'] = ts_pred
        return ts

    # save to file
    def save(self, file):
        with open(file + "_attrbute_list", "wb") as f:
            pickle.dump(self._attribute_list, f)
        with open(file + "_KPIPoints", "wb") as f:
            pickle.dump(self._KPIPoints, f)

    # load from file
    def load(self, file):
        with open(file + "_attrbute_list", "rb") as f:
            self._attribute_list = pickle.load(f)
        with open(file + "_KPIPoints", "rb") as f:
            self._KPIPoints = pickle.load(f)

    # test
    def test(self):
        print('attrbute list is %s, the attribute names are %s' % (self._attribute_list, self._attribute_names))
        for ts in self._KPIPoints:
            self._KPIPoints[ts].test()


'''
Test 
'''
class KPITest:
    def KPIPointTest(self):
        kPoint = KPIPoint({'a': ['a1', 'a2'], 'b': ['b1', 'b2', 'b3']},
                          1000, {('a1', 'b1'): [10, 0],
                                 ('a1', 'b2'): [10, 0],
                                 ('a1', 'b3'): [20, 0],
                                 ('a2', 'b1'): [30, 0],
                                 ('a2', 'b2'): [40, 0]})
        print('KPIPoint Test')
        kPoint.test()
        print('amount: ', kPoint.calAmount())
        print('(a1)\'s descendant elements are ', kPoint.get_descendant_elements_ele(('a1',)))
        print('combination ((\'a1\',),)\'s descendant are', kPoint.get_descendant_elements_coms((('a1',),)))
        print('cuboid (\'a\',) elements: ', kPoint.get_elements_in_cuboid(('a',)))
        print('cuboid (\'b\',) elements: ', kPoint.get_elements_in_cuboid(('b',)))
        print('cuboid (\'a\',\'b\') elements: ', kPoint.get_elements_in_cuboid(('a', 'b')))
        print('cuboid (\'a\',) elements combinations are: ', kPoint.get_elements_coms_in_cuboid(('a',)))
        print('cuboid (\'b\',) elements combinations are: ', kPoint.get_elements_coms_in_cuboid(('b',)))
        print('cuboid (\'a\',\'b\') elements combinations are: ', kPoint.get_elements_coms_in_cuboid(('a', 'b')))
        print('attrs in layer %d' % 1, kPoint.get_elements_set_by_layer(1))
        print('attrs in layer %d' % 2, kPoint.get_elements_set_by_layer(2))
        print('cuboid #1 all combinations: ', kPoint.get_elements_coms_by_layer(1))
        print('cuboid #2 all combinations: ', kPoint.get_elements_coms_by_layer(2))
        print('\n')

    def KPISetTest(self):
        kPoint1 = KPIPoint({'a': ['a1', 'a2'], 'b': ['b1', 'b2', 'b3']},
                           1000, {('a1', 'b1'): [10, 0],
                                  ('a1', 'b2'): [10, 0],
                                  ('a1', 'b3'): [20, 0],
                                  ('a2', 'b1'): [30, 0],
                                  ('a2', 'b2'): [40, 0]})
        kPoint2 = KPIPoint({'a': ['a1', 'a2'], 'b': ['b1', 'b2', 'b3']},
                           1001, {('a1', 'b1'): [20, 0],
                                  ('a1', 'b2'): [30, 0],
                                  ('a2', 'b1'): [40, 0],
                                  ('a2', 'b2'): [50, 0]})
        kSet = KPISet({'a': ['a1', 'a2'], 'b': ['b1', 'b2', 'b3']}, {1000: kPoint1, 1001: kPoint2})
        print('KPISet Test')
        kSet.test()
        print('(a1)\'s descendant elements are ', kSet.get_descendant_elements_coms(1000, (('a1',),)))
        print('timestamp: %d, attrs in layer %d' % (1000, 1), kSet.get_elements_set_by_layer(1000, 1))
        print('timestamp: %d, attrs in layer %d' % (1000, 2), kSet.get_elements_set_by_layer(1000, 2))
        print('time series of leaf (\'a1\', \'b1\')', kSet.get_ts_leaf(1000, 1011, 1, ('a1', 'b1')))
        print('time series of not leaf (\'a1\')', kSet.get_ts_not_leaf(1000, 1011, 1, ['a1']))
        kSet.save('../../result/metadata/kSet')

        kSet2 = KPISet({}, {})
        kSet2.load('../../result/metadata/kSet')
        kSet2.get_elements_set_by_layer(1000, 1)
        kSet2.get_elements_set_by_layer(1000, 2)
        kSet2.get_descendant_elements_coms(1001, ('b1', 'b3',))
        kSet2.get_ts_leaf(1000, 1011, 1, ('a1', 'b2'))
        kSet2.get_ts_not_leaf(1000, 1011, 1, ['a1'])
        print('\n')


"""
Data Transformer
"""
from tqdm import tqdm
import pandas as pd
class Transformer:
    def transformKPIData2KPIPoint(self, filePath, timestamp):
        data = pd.read_csv(filePath + str(timestamp) + '.csv', header=None, names=['i', 'e', 'c', 'p', 'l', 'KPI'])
        data = data.drop(data[data.values == 'unknown'].index, axis=0).reset_index(drop=True)
        data = data[data['KPI'] != 0].reset_index(drop=True)
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
            if element not in leaf:
                leaf[element] = [0, 0]
            leaf[element] = [leaf[element][0] + data.loc[i][-1], 0]
        return attribute_list, leaf

    def transformKPIData2KPISet(self, filePath, timestamp_start, timestamp_end, timestamp_interval):
        kPoints = {}
        attr_list_all = {}
        for ts in tqdm(range(timestamp_start, timestamp_end + timestamp_interval, timestamp_interval)):
            ts_file = ts * 1000
            attribute_list, leaf = self.transformKPIData2KPIPoint(filePath, ts_file)
            kPoints[ts] = KPIPoint(attribute_list, ts, leaf)
            for attr in attribute_list:
                if attr not in attr_list_all:
                    attr_list_all[attr] = []
                attr_list_all[attr] = sorted(list(set(attr_list_all[attr]).union(set(attribute_list[attr]))))
        kSet = KPISet(attr_list_all, kPoints)
        return kSet


"""
Data Explore
"""
class Explore:
    # 滚动统计
    def rolling_statistics(self, timeseries, timewindows=10):
        # Determing rolling statistics
        rolmean = timeseries.rolling(timewindows).mean()
        rolstd = timeseries.rolling(timewindows).std()
        allmean = [np.mean(timeseries)] * len(timeseries)
        allstd = [np.std(timeseries)] * len(timeseries)

        # Plot rolling statistics:
        plt.figure(figsize=(40, 10))
        orig = plt.plot(timeseries, color='blue', label='Original')

        mean = plt.plot(rolmean, color='red', label='Rolling Mean')
        std = plt.plot(rolstd, color='black', label='Rolling Std')
        mean_total = plt.plot(allmean, color='green', label='Total Mean')
        std_total = plt.plot(allstd, color='yellow', label='Total Std')
        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation')
        plt.show(block=False)

    '''
    # 返回值含义
    * adf_test的返回值
    * Test statistic：代表检验统计量
    * p-value：代表p值检验的概率
    * Lags used：使用的滞后k，autolag=AIC时会自动选择滞后
    * Number of Observations Used：样本数量
    * Critical Value(5%) : 显著性水平为5%的临界值。

    # 判断标准
    * 假设是存在单位根，即不平稳；
    * 显著性水平，1%：严格拒绝原假设；5%：拒绝原假设，10%类推。
    * 看P值和显著性水平a的大小，p值越小，小于显著性水平的话，就拒绝原假设，认为序列是平稳的；大于的话，不能拒绝，认为是不平稳的
    * 看检验统计量和临界值，检验统计量小于临界值的话，就拒绝原假设，认为序列是平稳的；大于的话，不能拒绝，认为是不平稳的
    '''
    # 平稳性检测
    def stationarity_test(self, timeseries, verbose=False):
        from statsmodels.tsa.stattools import adfuller as ADF
        diff = 0
        adf = ADF(timeseries)
        while adf[1] > 0.05:
            diff = diff + 1
            adf = ADF(timeseries.diff(diff).dropna())
        if verbose:
            print(u'原始序列经过%s阶差分后归于平稳，p值为%f' % (diff, adf[1]))
        return diff

    '''
    acorr_ljungbox(x, lags=None, boxpierce=False)函数检验无自相关
    lags为延迟期数，如果为整数，则是包含在内的延迟期数，如果是一个列表或数组，那么所有时滞都包含在列表中最大的时滞中
    boxpierce为True时表示除开返回LB统计量还会返回Box和Pierce的Q统计量
    返回值：
    lbvalue:测试的统计量
    pvalue:基于卡方分布的p统计量
    bpvalue:((optionsal), float or array) – test statistic for Box-Pierce test
    bppvalue:((optional), float or array) – p-value based for Box-Pierce test on chi-square distribution
    '''
    # 白噪声检测
    def whitenoise_test(self, timeseries, diff=1, verbose=False):
        ret = False
        from statsmodels.stats.diagnostic import acorr_ljungbox
        [[lb], [p]] = acorr_ljungbox(timeseries, lags=1)
        if p < 0.05:
            if verbose:
                print('原始序列为非白噪声序列，对应的p值为：%s' % p)
            pass
        else:
            ret = True
            if verbose:
                print('原始该序列为白噪声序列，对应的p值为：%s' % p)
            pass

        # 如果差分阶数为0，则返回结果
        if diff < 1:
            return ret

        [[lb], [p]] = acorr_ljungbox(timeseries.diff(diff).dropna(), lags=1)
        if p < 0.05:
            if verbose:
                print('%d阶差分序列为非白噪声序列，对应的p值为：%s' % (diff, p))
            pass
        else:
            ret = True
            if verbose:
                print('%d阶差分序列为白噪声序列，对应的p值为：%s' % (diff, p))

        return ret

    # 自相关系数与偏自相关系数
    def plot_acfandpacf(self, timeseries, diff=0, lags=100):
        from statsmodels.graphics.tsaplots import plot_acf
        from statsmodels.graphics.tsaplots import plot_pacf
        if diff > 0:
            timeseries = timeseries.diff(diff).dropna()
        fig = plt.figure(figsize=(40, 10))
        ax1 = fig.add_subplot(211)
        plot_acf(timeseries, lags=lags, ax=ax1)
        ax2 = fig.add_subplot(212)
        plot_pacf(timeseries, lags=lags, ax=ax2)
        plt.show()

    def statistic(self, filePath, leaves, type, length):
        stat = {}
        stat['leaf'] = leaves.tolist()
        stat['diff'] = []
        stat['isWhiteNoise'] = []
        stat['max'] = []
        stat['min'] = []
        stat['mean'] = []
        stat['std'] = []

        for leaf in tqdm(leaves.tolist()):
            l = '&'.join(leaf)
            ts = pd.read_csv(filePath + '/%s.csv' % l)[type]
            # 获取统计信息
            stat['max'].append(np.max(ts))
            stat['min'].append(np.min(ts))
            stat['mean'].append(np.mean(ts))
            stat['std'].append(np.std(ts))

            # 平稳性检测
            d = self.stationarity_test(ts[:length])
            stat['diff'].append(d)

            # 白噪声检验
            isWN = self.whitenoise_test(ts[:length], d)

            # 输出不是高斯白噪声的叶子元素
            if not isWN:
                stat['isWhiteNoise'].append(0)
            else:
                stat['isWhiteNoise'].append(1)

        stat = pd.DataFrame(stat)
        return stat

"""
predict
"""
class ARIMAPredict():
    def __init__(self, leaf):
        self.leaf = leaf
        self.model_result = None

    # ARIMA训练
    def train(self, ts, d, p=None, q=None, select=False):
        timeseries = list(ts / 1.0)
        if select:
            # 在statsmodels包里还有更直接的函数：
            import statsmodels.tsa.stattools as st
            order = st.arma_order_select_ic(timeseries, max_ar=5, max_ma=5, ic=['aic', 'bic', 'hqic'])
            p, q = order.bic_min_order
            print(order.bic_min_order, p, q)

        # 如果不自动选择p、q，则必须设置p和q，否则报错
        if not select and (pd.isnull(p) or pd.isnull(q)):
            raise ValueError("The p or q must be configured!")

        order = (p, d, q)
        from statsmodels.tsa.arima_model import ARIMA
        model = ARIMA(timeseries, order=order)
        self.model_result = model.fit(disp=-1)
        return order

    # ARIMA预测
    def predict(self, ts, number=None, start=None, end=None):
        self.model_result.model.endog = np.asarray(ts)
        return self.model_result.predict(start, end)

    # 保存模型
    def save(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)
        joblib.dump(self.model_result, path + '/model_' + self.leaf + '.pkl')

    # 加载模型
    def load(self, path):
        self.model_result = joblib.load(path + '/model_' + self.leaf + '.pkl')

    # 绘预测结果图
    def plot_results(self, ts_true, ts_pred):
        plt.figure(figsize=(40, 10))
        plt.plot(ts_true, label="true", color='green')
        plt.plot(pd.Series(ts_pred, index=ts_true.index), label="predicted")
        plt.legend()
        plt.show()


"""
评估
"""
import matplotlib.pyplot as plt
class Evalutaion:
    # 评估模型
    def evaluate(self, y_test, y_pred, score='', verbose=False):
        # 评估模型
        from sklearn import metrics
        mse = metrics.mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        mae = metrics.mean_absolute_error(y_test, y_pred)
        mdae = metrics.median_absolute_error(y_test, y_pred)
        acc = metrics.explained_variance_score(y_test,
                                               y_pred)  # 1 - np.sum(np.abs(np.array(((y_test - y_pred) / y_test))) / len(y_test))
        r2 = metrics.r2_score(y_test, y_pred)

        score_ret = {}
        score_ret['MSE'] = mse
        score_ret['RMSE'] = rmse
        score_ret['MAE'] = mae
        score_ret['MDAE'] = mae
        score_ret['ACC'] = acc
        score_ret['R2'] = r2

        if score == '':
            if verbose:
                # 用scikit-learn计算MSE
                print("MSE:", mse)
                # 用scikit-learn计算RMSE
                print("RMSE:", mse)
                # 用scikit-learn计算MAE
                print("MAE:", mae)
                # 用scikit-learn计算MDAE
                print("MDAE:", mdae)
                # 正常精确度
                print("ACC:", acc)
                # R2
                print("R2:", r2)
            return score_ret
        else:
            if verbose:
                print(score, score_ret[score])
            return {score: score_ret[score]}

    # 可视化模型结果
    def visualize(self, y_test, y_pred):
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred)
        ax.plot([np.array(y_test).min(), np.array(y_test).max()], [np.array(y_pred).min(), np.array(y_pred).max()],
                'k--', lw=4)
        ax.set_xlabel('Measured')
        ax.set_ylabel('Predicted')
        plt.show()