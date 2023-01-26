import networkx as nx
import numpy as np


class random_graph_MT(object):
    def __init__(self, N, alpha=1.0, min_degree=None,
                 max_degree=None, gamma=2.5):
        self.alpha = alpha
        self.N = N
        self.min_degree = min_degree
        self.max_degree = max_degree
        self.gamma = gamma

    def node_model(self, ii):
        # build a node connectivity probability
        dnode = np.arange(1, self.N - ii)
        ps = np.exp(-self.alpha * dnode)
        ps = ps / np.sum(ps)
        con_node = dnode + ii

        # determine the degree of this node
        k_max = self.N - ii
        # k_min = 1

        # if we provide a fractional degree, we compute what we need based on
        # the allowed number of connections
        min_degree = min(self.min_degree, k_max)
        max_degree = min(self.max_degree, k_max) - 1
        # min_degree = min(min_degree, max_degree)
        if self.max_degree < 1:
            max_degree = min(k_max, int(self.N * self.max_degree + 0.5)) - 1
        if self.min_degree < 1:
            min_degree = min(k_max, int(self.N * self.min_degree + 0.5)) - 1
            min_degree = max(1, min_degree)

        min_degree = max(min_degree, 1)
        max_degree = max(max_degree, 1)
        min_degree = min(min_degree, max_degree)

        degrees = np.arange(min_degree, max_degree + 1)

        pd = (1.0 * degrees) ** (-self.gamma)
        pd = pd / np.sum(pd)
        # now do the sampling of nodes, without replacement
        # first determin the actual degree
        this_degree = degrees[0]
        if len(degrees) > 1:
            this_degree = np.random.choice(a=degrees, size=1, p=pd)[0]
        # now we need to draw this_degree nodes
        connected_nodes = np.random.choice(a=con_node, size=this_degree,
                                           p=ps, replace=False)
        sel = np.argsort(connected_nodes)
        return connected_nodes[sel]

    def random_graph(self):
        G = nx.DiGraph()
        for ii in range(self.N - 1):
            cons = self.node_model(ii)
            for c in cons:
                G.add_edge(ii + 1, c + 1)
        return G


def random_LL_graph(N, alpha, gamma, min_degree=None, max_degree=None, return_graph=False):
    if max_degree is None:
        max_degree = 1.0
    obj = random_graph_MT(N, alpha=alpha, gamma=gamma, min_degree=min_degree, max_degree=max_degree)
    G = obj.random_graph()
    if return_graph:
        return G

    g_random = nx.linalg.graphmatrix.adjacency_matrix(G).todense()
    g_random = np.triu(g_random)

    rows = np.sum(g_random, axis=0)
    columns = np.sum(g_random, axis=1)

    tots = rows + columns
    start_points = np.where((rows == 0) * (tots > 0))[0]
    end_points = np.where((columns == 0) * (tots > 0))[0]
    return g_random, start_points, end_points


def assign_size(G, powers, p_power=None):
    if p_power is None:
        p_power = []
    if powers is None:
        tmp = []
        for ii, node in enumerate(G.nodes()):
            G.nodes[node]["size"] = 0
            tmp.append(1)
    else:
        if len(p_power) == 0:
            p_power = np.ones(len(powers))
            p_power = p_power / np.sum(p_power)
        for ii, node in enumerate(G.nodes()):
            this_power = np.random.choice(powers, 1, p=p_power, replace=True)
            G.nodes[node]["size"] = this_power[0]

    return G


def tst():
    G = random_LL_graph(alpha=10.25, gamma=0.0, N=5, return_graph=True)
    powers = np.array([0, -1, -2, -3])
    p_power = np.array([2.0, 1.0, 1.0, 1.0])
    p_power = p_power / np.sum(p_power)
    assign_size(G, powers, p_power)


if __name__ == "__main__":
    tst()
