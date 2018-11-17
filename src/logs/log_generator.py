from src.basic_entities.graph import DGraph
from src.main.config import RESULTS_PATH
import numpy as np

def generate_single_split_model(split_bias_probability=0.0):
    '''

    :param split_bias_probability: how far from random (i.e., 50/50) should the split be); must be <= 0.5
    :return: the model:
    1->(a, 0.5+bias)->2
    1->(a, 0.5-bias)->3
    2->(b, 1)->4
    3->(c, 1)->4
    '''
    if split_bias_probability > 0.5:
        raise AssertionError('split bais must be smaller then 0.5')

    g = DGraph()
    g.add_node(1, 'n1')
    g.add_node(2, 'n2')
    g.add_node(3, 'n3')
    g.add_node(4, 'n4')
    g.add_edge(1, 2, 0.5 + split_bias_probability)
    g.add_edge_attribute(1, 2, 'label', 'a')
    g.add_edge(1, 3, 0.5 - split_bias_probability)
    g.add_edge_attribute(1, 3, 'label', 'a')
    g.add_edge(2, 4, 1)
    g.add_edge_attribute(2, 4, 'label', 'b')
    g.add_edge(3, 4, 1)
    g.add_edge_attribute(3, 4, 'label', 'c')
    return g


def choose_element_at_random(elements, probabilities = None):
    '''
    :param elements: : a list of elemnts
    :param probabilities: a list of probabilities per elements (must sum to 1); assumed uniformed of non is provided
    :return: element from the list
    '''
    if sum(probabilities) != 1:
        raise AssertionError('probabilities must some to 1.0')
    N = len(elements)
    if not probabilities:
        probabilities = []
        for i in range(N):
            probabilities.append(1.0 / N)
    ind = np.random.choice(N, p=probabilities, size=1)
    return elements[ind[0]]

def generate_trace(g, edge_label_attribute='label', node_label_attribute=None, edge_prob_attribute='weight'):
    '''
    :param g: a directed graph,
    Assumption 1: must have at least one sink state
    Assumption 2: sink states are interpreted as terminal states
    Assumption 3: it is assumed that any state in g can reach a terminal state
    :return: a trace
    '''
    init_nodes = [n[0] for n in g.get_initial_nodes()]
    terminal_nodes = [n[0] for n  in g.get_sink_nodes()]
    if len(init_nodes) == 0:
        raise AssertionError('To generate a trace, at least one node must be provided!')

    current_node = init_nodes[0]
    if len(init_nodes) > 1:
        # if multiple initial nodes exists, choose one at random: TODO: use node weight when possible
        current_node = choose_element_at_random(init_nodes)

    trace = []
    while current_node not in terminal_nodes:
        # if required, add nodes label
        if node_label_attribute:
            trace.append(g.node_attr(current_node, node_label_attribute))

        # select next edge randomaly by weight
        edges = g.out_edges(current_node)
        weights = [g.get_edge_data(e)[edge_prob_attribute] for e in edges]
        e = choose_element_at_random(list(edges), weights)

        # if required, add edge label
        if edge_label_attribute:
            trace.append(g.get_edge_data(e)[edge_label_attribute])
        current_node = e[1]

    return trace

def get_test_log(bias = 0.0, size =10, add_dummy_initial= True, add_dummy_terminal= True):
    '''
        return logs from the model:
        1->(a, 0.5+bias)->2
        1->(a, 0.5-bias)->3
        2->(b, 1)->4
        3->(c, 1)->4
        :param split_bias_probability: how far from random (i.e., 50/50) should the split be); must be <= 0.5
        :return:
        '''
    g = generate_single_split_model(bias)
    traces = []
    for i in range(size):
        t = generate_trace(g)
        if add_dummy_initial:
            t.insert(0, 'I')
        if add_dummy_terminal:
            t.append('T')
        traces.append(tuple(t))
    return traces

if __name__ == '__main__':

    g = generate_single_split_model()
    g.write_dot(RESULTS_PATH + '/exmaple_1.dot', True)
    ks_dict = {}
    for i in range(100):
        t = tuple(generate_trace(g))
        ks_dict[t] = ks_dict.get(t, 0) + 1
    print (ks_dict)