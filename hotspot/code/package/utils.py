# coding: utf-8
import itertools
import pickle
import numpy as np

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
                if value == [0, 0]: # if there exists no ele in the leaf elements, continue
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
                if value == [0, 0]: # if there exists no ele in the leaf elements, continue
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