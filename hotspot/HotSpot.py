# coding: utf-8
from package.utils import KPIPoint
from package.utils import KPISet
from package.utils import KPITest
import pandas as pd
import numpy as np


class HotSpot():
    def __init__(self, KPIPoint, L, pt, m):
        # All the KPI values of elements when the total KPI is found anomalous
        self._KPIPoint = KPIPoint
        # maximum ID of Layer
        self._max_layer_id = L
        # potential score threshold
        self._ps_threshold = pt
        # maximum number of Iteration
        self._max_iteration = m
        # the actual leaf value vertor
        self._actaul_vector = []
        # the feature leaf value vertor
        self._feature_vector = []
        self.cal_vector()

    # find the root cause set
    def find_root_cause_set(self):
        RSets = []
        for l in range(1, self._max_layer_id + 1):
            # Parallel Execution in each cuboid
            elements_set = self._KPIPoint.get_elements_set_by_layer(l)

            # Calculate Potential Scores ps(ek) of each element ek
            # Sort ek in a descending order of ps(ek)
            ps_set = {}
            for ele in elements_set:
                ps = self.cal_potential_scores(ele, elements_set[ele])
                ps_set[ele] = ps

            temp = sorted(ps_set.items(), key=lambda x: x[1], reverse=True)
            ps_set_sorted = {}
            for i in range(len(temp)):
                ps_set_sorted[temp[i][0]] = temp[i][1]
            print('sorted ps set: ', ps_set_sorted)
            print('\n')
            continue

            # i is the number of iteration now, and be initialed 0
            i = 0
            while True:
                # Choose a set use UCB algorithm
                cand_set = []
                if i >= self._max_iteration:
                    break
                if ps_sorted[cand_set] >= self._ps_threshold:
                    RSets = cand_set
                    return RSets
                i = i + 1
                # Obtain BSetl,j
                # Prune ec in layer l+1 whose father e f are not in BSetl,j
                # if All the ec in layer l+1 are pruned then break
        # Choose RSet form BSetl,j with the largest ps
        # ps(RSet) = Max{ps(BSetl,j)}
        return RSets

    # calculate the potential score of element
    def cal_potential_scores(self, elements_set, value):
        # print('elements_set: ', elements_set, '\n')
        # print(value)
        deduced_leaf = {}
        for leaf in self._KPIPoint._leaf:
            deduced_leaf[leaf] = self._KPIPoint._leaf[leaf][1]
        visited = {}
        for element in elements_set:
            # print('element', element)
            deduced_value = 0
            # if element is in LEAF
            if len(element) == len(self._KPIPoint._attribute_names):
                print('element is in LEAF')
                deduced_leaf[element] = self._KPIPoint._leaf[element][0]
            else:
                leaves, _ = self._KPIPoint.get_descendant_elements_ele(element)
                # print('leaves involved', leaves)
                for leaf in leaves:
                    if leaf not in visited:
                        visited[leaf] = 0
                    visited[leaf] = visited[leaf] + 1
                    # print('leaf: ', leaf)
                    inter = list(set(element).intersection(set(leaf)))
                    if len(inter) == 0:
                        # print('inter %d', len(inter))
                        deduced_value = self._KPIPoint._leaf[leaf][1]
                    else:
                        f_c = value[1]
                        if f_c == 0:
                            print('error, f(c) == 0')
                            continue
                        f_l = self._KPIPoint._leaf[leaf][1]
                        a_c = value[0]
                        deduced_value = f_l - (f_c - a_c) * (f_l / f_c)
                    # print('deduced_value %f' % deduced_value)
                    deduced_leaf[leaf] = deduced_value

        # update deduced vector
        deduced_vector = []
        for leaf in self._KPIPoint._leaf:
            deduced_value = deduced_leaf[leaf]
            deduced_vector.append(deduced_value)
        print('visited: ', visited)
        dis_a_v = self.cal_euclidean_distance(self._actaul_vector, deduced_vector)
        dis_a_f = self.cal_euclidean_distance(self._actaul_vector, self._feature_vector)
        rate = dis_a_v / dis_a_f
        ps = np.max([1 - rate, 0])
        print('ps of ', elements_set, ' is', ps)
        return ps

    # calculate potential scores Revised
    def cal_potential_scores_revised(self, ele, value):
        return

    # calculate the actual and predict vector
    def cal_vector(self):
        self._actaul_vector = []
        self._feature_vector = []
        for leaf in self._KPIPoint._leaf:
            self._actaul_vector.append(self._KPIPoint._leaf[leaf][0])
            self._feature_vector.append(self._KPIPoint._leaf[leaf][1])
        print(self._actaul_vector,
              self._feature_vector)

    # calculate the Euclidean distance
    def cal_euclidean_distance(self, v1, v2):
        vec1 = np.array(v1)
        vec2 = np.array(v2)
        return np.linalg.norm(vec1 - vec2)