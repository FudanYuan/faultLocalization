# coding: utf-8
from package.utils import KPIPoint
from package.utils import KPISet
from package.utils import KPITest
import pandas as pd
import numpy as np
from math import *
import random
import time


class HotSpot():

    def __init__(self, KPIPoint=None, L=1, pt=0.99, ept=0.01, m=100):
        # All the KPI values of elements when the total KPI is found anomalous
        self._KPIPoint = KPIPoint
        # maximum ID of Layer
        self._max_layer_id = L
        # potential score threshold
        self._ps_threshold = pt
        # explanatory power threshold
        self._ep_threshold = ept
        # maximum number of Iteration
        self._max_iteration = m
        # the actual leaf value vertor
        self._actaul_vector = []
        # the feature leaf value vertor
        self._feature_vector = []
        self.cal_vector()

    # find the root cause set
    def find_root_cause_set(self):
        RSets = None
        BSets = {}
        candidateSet = []
        sTime = time.time()
        for l in range(1, self._max_layer_id + 1):
            # Parallel Execution in each cuboid
            """
            # 暴力搜索
            elements_set = self._KPIPoint.get_elements_coms_by_layer(l)
            """
            # MCTS搜索
            # prune strategy
            # print('get all elements set...')
            elements_set = self._KPIPoint.get_elements_set_by_layer_with_prune(l, candidateSet)
            # elements_set = self._KPIPoint.get_elements_set_by_layer(l)
            # print('layer #%d all elements: ' % l, elements_set)

            for cuboid in elements_set:
                # Calculate Potential Scores ps(ek) of each element ek
                # Sort ek in a descending order of ps(ek)
                ps_set = {}
                for ele in elements_set[cuboid]:
                    # potential score
                    ps = self.cal_potential_scores(ele, elements_set[cuboid])
                    if ps == 0:
                        continue
                    ps_set[ele] = ps

                # if ps_set is none
                if len(ps_set) == 0:
                    continue

                ps_set_sorted = dict(sorted(ps_set.items(), key=lambda x: x[1], reverse=True))
                # print('sorted ps set: ', ps_set_sorted)
                # print('element length: ', len(ps_set_sorted))
                state = self.State(ps_set_sorted)
                bestSet, bestPS = self.MCTS(state, self._max_iteration, True)
                parentList = []
                for e in bestSet:
                    parentList.append(e)
                candidateSet = parentList
                BSets[bestSet] = bestPS
                # print('best set', bestSet)
                # print('\n')
                # break

                # Obtain BSetl,j
                # Prune e_c in layer l+1 whose father e_f are not in BSetl,j
                # if All the e_c in layer l+1 are pruned then break
        # print('best set is : ', BSets)
        # print('candidate set are :', candidateSet)

        # Choose RSet form BSetl,j with the largest ps
        # ps(RSet) = Max{ps(BSetl,j)}
        RSets = sorted(BSets.items(), key=lambda c: c[1], reverse=True)[:3]
        eTime = time.time()
        # print('runtime %fs' % (eTime - sTime))
        return RSets

    # find the root cause set Revised
    def find_root_cause_set_revised(self):
        RSets = None
        BSets = {}
        candidateSet = []
        # sTime = time.time()
        value_cuboid = np.sum(list(map(lambda c: c[1], list(self._KPIPoint._leaf.items()))), axis=0)
        for l in range(1, self._max_layer_id + 1):
            # Parallel Execution in each cuboid
            """
            # 暴力搜索
            elements_set = self._KPIPoint.get_elements_coms_by_layer(l)
            """
            # MCTS搜索
            # prune strategy
            # print('get all elements set...')
            elements_set = self._KPIPoint.get_elements_set_by_layer_with_prune(l, candidateSet)
            # elements_set = self._KPIPoint.get_elements_set_by_layer(l)
            # print('layer #%d all elements: ' % l, elements_set)

            for cuboid in elements_set:
                # Calculate Potential Scores ps(ek) of each element ek
                # Sort ek in a descending order of ps(ek)
                ps_set = {}
                for ele in elements_set[cuboid]:
                    # potential score
                    ps = self.cal_potential_scores(ele, elements_set[cuboid], True)
                    if ps == 0:
                        continue
                    # explanatory power
                    value_subcuboid = elements_set[cuboid][ele]
                    ep = (value_subcuboid[0] - value_subcuboid[1]) / (value_cuboid[0] - value_cuboid[1])
                    if ep < self._ep_threshold:
                        continue
                    ps_set[ele] = ps

                # if ps_set is none
                if len(ps_set) == 0:
                    continue

                ps_set_sorted = dict(sorted(ps_set.items(), key=lambda x: x[1], reverse=True))
                # print('sorted ps set: ', ps_set_sorted)
                # print('element length: ', len(ps_set_sorted))
                state = self.State(ps_set_sorted)
                bestSet, bestPS = self.MCTS(state, self._max_iteration, True)
                parentList = []
                for e in bestSet:
                    parentList.append(e)
                candidateSet = parentList
                BSets[bestSet] = bestPS
                # print('best set', bestSet)
                # print('\n')
                # break

                # Obtain BSetl,j
                # Prune e_c in layer l+1 whose father e_f are not in BSetl,j
                # if All the e_c in layer l+1 are pruned then break
        # print('best set is : ', BSets)
        # print('candidate set are :', candidateSet)

        # Choose RSet form BSetl,j with the largest ps
        # ps(RSet) = Max{ps(BSetl,j)}
        RSets = sorted(BSets.items(), key=lambda c: c[1], reverse=True)[:3]
        # eTime = time.time()
        # print('runtime %fs' % (eTime - sTime))
        return RSets

    # calculate the potential score of element
    def cal_potential_scores(self, elements_set, value, revised=False):
        # print('elements_set: ', elements_set)
        # print(value)
        deduced_leaf = {}
        for leaf in self._KPIPoint._leaf:
            deduced_leaf[leaf] = self._KPIPoint._leaf[leaf][1]
        for element in elements_set:
            deduced_value = 0
            # if element is in LEAF
            if len(element) == len(self._KPIPoint._attribute_names):
                if revised:  # if version is revised
                    deduced_leaf[element] = self._KPIPoint._leaf[element][0]
                else:
                    deduced_leaf[element] = self._KPIPoint._leaf[element][0]
            else:
                leaves, _ = self._KPIPoint.get_descendant_elements_ele(element)
                # print('leaves involved', leaves)
                for leaf in leaves:
                    inter = list(set(element).intersection(set(leaf)))
                    if len(inter) == 0:
                        # print('inter %d', len(inter))
                        deduced_value = self._KPIPoint._leaf[leaf][1]
                    else:
                        f_c = value[(element,)][1]
                        if f_c == 0:
                            # print('f(c) == 0')
                            continue
                        f_l = self._KPIPoint._leaf[leaf][1]
                        a_c = value[(element,)][0]
                        deduced_value = f_l - (f_c - a_c) * (f_l / f_c)
                    deduced_leaf[leaf] = deduced_value

        # update deduced vector
        deduced_vector = []
        for leaf in self._KPIPoint._leaf:
            deduced_value = deduced_leaf[leaf]
            deduced_vector.append(deduced_value)
        dis_a_v = self.cal_euclidean_distance(self._actaul_vector, deduced_vector)
        dis_a_f = self.cal_euclidean_distance(self._actaul_vector, self._feature_vector)
        rate = dis_a_v / dis_a_f
        ps = np.max([1 - rate, 0])
        return ps

    # calculate the actual and predict vector
    def cal_vector(self):
        self._actaul_vector = []
        self._feature_vector = []
        for leaf in self._KPIPoint._leaf:
            self._actaul_vector.append(self._KPIPoint._leaf[leaf][0])
            self._feature_vector.append(self._KPIPoint._leaf[leaf][1])
        # print(self._actaul_vector, self._feature_vector)

    # calculate the Euclidean distance
    def cal_euclidean_distance(self, v1, v2):
        vec1 = np.array(v1)
        vec2 = np.array(v2)
        return np.linalg.norm(vec1 - vec2)

    class State:
        """ A state of current tree
        """

        def __init__(self, elements=None):
            # all the elements to select
            self._elements = elements
            # record the visited elements
            self._visited_elements = []
            # record the visited times of edge(s,a)
            self._edge_visited = {}

        # create a deep clone of this state.
        def Clone(self):
            state = HotSpot.State(self._elements)
            state._edge_visited = self._edge_visited
            return state

        # visit an element
        def visit_element(self, ele):
            self._visited_elements.append(ele)

        # visit an to_ele from from_ele
        def update_edge_visits(self, from_ele, to_ele):
            if from_ele is None:
                from_ele = ()
            if (from_ele, to_ele) not in self._edge_visited:
                self._edge_visited[(from_ele, to_ele)] = 0
            self._edge_visited[(from_ele, to_ele)] += 1

        # get the unvisited elements
        def get_unvisited_elements(self):
            unvisited_elements = {}
            for element in self._elements:
                if element not in self._visited_elements:
                    unvisited_elements[element] = self._elements[element]
            return unvisited_elements

        # get the element with max potential score.
        def get_ele_with_max_ps(self):
            return sorted(self._elements.items(), key=lambda x: x[1])[-1]

    class Node:
        """ A node in the tree.
        """

        def __init__(self, elements=None, parent=None, state=None):
            self._elementsSet = elements  # the elements that got us to this node - "None" for the root node
            self._parent_node = parent  # "None" for the root node
            self._child_nodes = []
            self._score = state.get_ele_with_max_ps()[1]
            self._visits = 0
            self._state = state  # all nodes share one state
            self._unvisited_elements = state.get_unvisited_elements()  # future child nodes

        def select_child(self):
            """ Use the UCB formula to select a child node.
            """
            # if N(s,a) = 0, then assign a probability of taking unvisited actions to be
            # R = (1 − Q(s,amax)), where amax = argmaxa∈A(s)∩N(s,a)=0 Q(s,a).
            # print("parent: ", self._elementsSet, "child: ", self._child_nodes)
            # print(self._state._edge_visited)
            node = sorted(self._child_nodes,
                          key=lambda c: c._score +
                                        sqrt(2 * log(c._visits)
                                             / self._state._edge_visited[(self._elementsSet, c._elementsSet)]))[-1]
            if len(self._unvisited_elements) == 0:
                return node

            prob = random.random()
            element = sorted(self._unvisited_elements.items(), key=lambda x: x[1])[-1]
            if prob > 1 - element[1]:
                # print('explore')
                self._state.visit_element(element[0])
                node = self.add_child(element[0], self._state)
            return node

        def add_child(self, e, s):
            """ Remove m from _unvisited_elements and add a new child node for this elements.
                Return the added child node
                * We choose e∗ to have the largest ps(S) value of the remaining elements rather than choosing e∗ randomly.
            """
            # print(self._elementsSet, ' add child: ', e)
            # print('state unvisited elements', s.get_unvisited_elements())
            if self._elementsSet is None:
                self._elementsSet = ()
            elementsSet = self._elementsSet + e
            n = HotSpot.Node(elements=elementsSet, parent=self, state=s)

            if e in self._unvisited_elements:
                del self._unvisited_elements[e]
            self._child_nodes.append(n)
            return n

        def update(self, result):
            """ update this node. We update the Q of a father only when the child’s Q is greater than the father’s.
            """
            self._visits += 1
            if self._score < result:
                self._score = result

            from_elementsSet = ()
            if self._parent_node != None:
                from_elementsSet = self._parent_node._elementsSet

            if len(self._elementsSet) == 0:
                return
            to_elementsSet = self._elementsSet
            self._state.update_edge_visits(from_elementsSet, to_elementsSet)

        # get the element with max potential score.
        def get_unvisited_ele_with_max_ps(self):
            return sorted(self._unvisited_elements.items(), key=lambda x: x[1])[-1]

        def __repr__(self):
            return "[elementsSet:" + str(self._elementsSet) + " Score/Visits:" + str(self._score) \
                   + "/" + str(self._visits) + " Unvisited:" + str(self._unvisited_elements) + "]"

        def TreeToString(self, indent):
            s = self.IndentString(indent) + str(self)
            for c in self._child_nodes:
                s += c.TreeToString(indent + 1)
            return s

        def IndentString(self, indent):
            s = "\n"
            for i in range(1, indent + 1):
                s += "| "
            return s

        def ChildrenToString(self):
            s = ""
            for c in self._child_nodes:
                s += str(c) + "\n"
            return s

    def MCTS(self, root_state, itermax, verbose=False):
        """ Conduct a MCTS search for itermax iterations starting from root_state.
            Return the best set from the root_state.
        """

        retSet = {}
        rootnode = self.Node(state=root_state)
        bestPS = 0
        bestSet = None
        for i in range(itermax):
            node = rootnode
            state = root_state.Clone()

            ps = node._score
            # print('ps origin', ps)

            # Select
            while node._child_nodes != []:
                node = node.select_child()
                for element in node._elementsSet:
                    state.visit_element((element,))
            # print('Select Node')

            # Expand
            if len(node._unvisited_elements) != 0:  # if we can expand (i.e. state/node is non-terminal)
                element_choose = node.get_unvisited_ele_with_max_ps()[0]
                state.visit_element(element_choose)
                node = node.add_child(element_choose, state)  # add child and descend tree
            # print('Expand Node')

            # Evaluation
            elementsSet = node._elementsSet
            if elementsSet in state._elements:
                ps = state._elements[elementsSet]
            else:
                _, value = self._KPIPoint.get_descendant_elements_coms(elementsSet)
                ps = self.cal_potential_scores(elementsSet, value)
            # print('Evaluation', ps)

            if ps > bestPS:
                bestPS = ps
                bestSet = elementsSet

            # Early stop
            if ps >= self._ps_threshold:
                # print('Early stop')
                elementsSet = node._elementsSet
                return elementsSet, ps

            # Backpropagate
            while node != None:  # backpropagate from the expanded node and work back to the root node
                node.update(ps)  # state is terminal. update node with potential score
                node = node._parent_node
            # print('Backpropagate')

            # Output some information about the tree - can be omitted
            if (verbose):
                # print(rootnode.TreeToString(0))
                pass
            else:
                # print(rootnode.ChildrenToString())
                pass

            if len(state.get_unvisited_elements()) == 0:
                # print('all elements visited')
                break

            # print('\n#%d edge visited' % i, state._edge_visited, node, '\n')

        # retNode = sorted(rootnode._child_nodes, key=lambda c: c._score)[-1]
        # ps = retNode._score
        return bestSet, bestPS


if __name__ == "__main__":
    attr_map = {'a': ['a1', 'a2'], 'b': ['b1', 'b2', 'b3']}
    # kPoint = KPIPoint(attr_map,
    #                   1000, {('a1', 'b1'): [14, 20],
    #                          ('a1', 'b2'): [9, 15],
    #                          # ('a1', 'b3'): [10, 10],
    #                          # ('a2', 'b1'): [7, 10],
    #                          ('a2', 'b2'): [15, 25],
    #                          ('a2', 'b3'): [20, 20],
    #                          })

    # kPoint = KPIPoint(attr_map,
    #                   1000, {('a1', 'b1'): [50000, 0.1],
    #                          ('a1', 'b2'): [100000, 0.2],
    #                          ('a1', 'b3'): [5050, 5000],
    #                          # ('a2', 'b1'): [7, 10],
    #                          # ('a2', 'b2'): [15, 25],
    #                          ('a2', 'b3'): [1040, 1000],
    #                          })

    kPoint = KPIPoint(attr_map,
                      1000, {('a1', 'b1'): [5, 20],
                             ('a1', 'b2'): [15, 15],
                             ('a1', 'b3'): [10, 10],
                             ('a2', 'b1'): [10, 10],
                             ('a2', 'b2'): [13, 25],
                             ('a2', 'b3'): [20, 20],
                             })
    hotSpot = HotSpot(kPoint, 2, 0.99, 0.01, 10)
    rootCauseSet = hotSpot.find_root_cause_set()
    print(rootCauseSet)

    rootCauseSet = hotSpot.find_root_cause_set_revised()
    print(rootCauseSet)
