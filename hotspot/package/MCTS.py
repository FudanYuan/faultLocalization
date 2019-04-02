# coding: utf-8
from package.HotSpot import *
from math import *
import random

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
        # record the visited times of state s, N(s)
        self._node_visited = {}

    # create a deep clone of this state.
    def Clone(self):
        return State(self.elements)

    # visit an to_ele from from_ele
    def visit_element(self, from_ele, to_ele):
        self._visited_elements.append(to_ele)
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
        return sorted(unvisited_elements.items(), key = lambda x : x[1])[-1][0]


class Node:
    """ A node in the tree.
    """
    def __init__(self, elements=None, parent=None, state=None):
        self._elementsSet = elements  # the elements that got us to this node - "None" for the root node
        self._parent_node = parent  # "None" for the root node
        self._child_nodes = []
        self._score = 0
        self._visits = 0
        self._state = state         # all nodes share one state
        self._unvisited_elements = state.get_unvisited_elements()  # future child nodes

    def select_child(self):
        """ Use the UCB formula to select a child node.
        """
        # if N(s,a) = 0, then assign a probability of taking unvisited actions to be
        # R = (1 − Q(s,amax)), where amax = argmaxa∈A(s)∩N(s,a)=0 Q(s,a).
        node = sorted(self._child_nodes,
                      key=lambda c: c._score +
                                    sqrt(2 * log(c._visits)
                                         / self._state._edge_visited[(self._elementsSet, c._elementsSet[-1])]))[-1]
        if self._unvisited_elements == []:
            return node

        prob = random.random()
        element = sorted(unvisited_elements.items(), key=lambda x: x[1])[-1]
        if prob > 1 - element[element.keys()[0]]:
            elementsSet = self._elementsSet
            self._state.visit_element(self._elementsSet, element)
            node = self.add_child(element, self._state)
        return node

    def add_child(self, e, s):
        """ Remove m from _unvisited_elements and add a new child node for this elements.
            Return the added child node
            * We choose e∗ to have the largest ps(S) value of the remaining elements rather than choosing e∗ randomly.
        """
        elementsSet = self._elementsSet + e
        n = Node(elements=elementsSet, parent=self, state=s)
        self._unvisited_elements.remove(e)
        self._child_nodes.append(n)
        return n

    def update(self, result):
        """ update this node. We update the Q of a father only when the child’s Q is greater than the father’s.
        """
        self.visits += 1
        if self.score > result:
            self.score = result

    def __repr__(self):
        return "[M:" + str(self._elementsSet) + " W/V:" + str(self.score) + "/" + str(self.visits) + " U:" + str(
            self._unvisited_elements) + "]"

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


def MCTS(root_state, itermax, verbose=False):
    """ Conduct a MCTS search for itermax iterations starting from root_state.
        Return the best set from the root_state.
    """

    rootnode = Node(state=root_state)

    for i in range(itermax):
        node = rootnode
        state = root_state.Clone()

        # Select
        while node._unvisited_elements == [] and node._child_nodes != []:  # node is fully expanded and non-terminal
            node = node.select_child()
            state.visit_element((), node._elementsSet)

        # Expand
        if node._unvisited_elements != []:  # if we can expand (i.e. state/node is non-terminal)
            element_choose = node.get_unvisited_ele_with_max_ps()
            state.visit_element(node._elementsSet, element_choose)
            node = node.add_child(element_choose, state)  # add child and descend tree

        # Evaluation


        # Backpropagate
        # while node != None:  # backpropagate from the expanded node and work back to the root node
            # node.update(state.GetResult(.playerJustMoved))  # state is terminal. update node with result from POV of node.playerJustMoved
            # node = node._parent_node

    # Output some information about the tree - can be omitted
    if (verbose):
        print(rootnode.TreeToString(0))
    else:
        print(rootnode.ChildrenToString())

    return sorted(rootnode._child_nodes, key=lambda c: c.visits)[-1]._elementsSet  # return the elements that was most visited