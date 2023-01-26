# TODO: Add docstrings

import networkx as nx
import numpy as np
from dlsia.core.networks import graph_utils


def random_DAG_scale_free(n, p):
    G_random = nx.barabasi_albert_graph(n, max(1, int(n * p)))
    g_random = nx.linalg.graphmatrix.adjacency_matrix(G_random).todense()
    g_random = np.triu(g_random)

    rows = np.sum(g_random, axis=0)
    columns = np.sum(g_random, axis=1)

    tots = rows + columns

    start_points = np.where((rows == 0) * (tots > 0))[0]
    end_points = np.where((columns == 0) * (tots > 0))[0]
    return g_random, start_points, end_points


def random_DAG(n, p):
    G_random = nx.generators.random_graphs.fast_gnp_random_graph(n, p)
    g_random = nx.linalg.graphmatrix.adjacency_matrix(G_random).todense()
    g_random = np.triu(g_random)

    rows = np.sum(g_random, axis=0)
    columns = np.sum(g_random, axis=1)

    tots = rows + columns

    start_points = np.where((rows == 0) * (tots > 0))[0]
    end_points = np.where((columns == 0) * (tots > 0))[0]
    return g_random, start_points, end_points


def sparsity(g):
    N = g.shape[0]
    tot = (N * N - N) / 2
    obs = np.sum(g)
    return obs / tot


def draw_random_dag(n, p, tol=0.01, max_count=100):
    eps = 2.0 / ((n * n - n) / 2)
    if p < eps:
        p = eps
    if tol < eps / 2.0:
        tol = eps / 2.0

    ok = False
    count = 0
    g = None
    while not ok:
        g, starts, stops = random_DAG(n, p)
        obs_sparsity = sparsity(g)
        if obs_sparsity > 0:
            if np.abs(obs_sparsity - p) < tol:
                ok = True
        count += 1
        if count > max_count:
            print("Having trouble getting desired sparsity, will continue")
            ok = True

    return g


class RandomMultiSourceMultiSinkGraph(object):
    """
    Builds a random graph with specified sparsity parameters.
    Can specify the source and sink parameters, and control
    the distribution of length of skip connections between hidden
    layers.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 layers,
                 dilations,
                 LL_alpha=0.25,
                 LL_gamma=1.5,
                 LL_min_degree=1,
                 LL_max_degree=None,
                 IL_target=0.15,
                 LO_target=0.15,
                 IO_target=False):
        """

        :param in_channels: Input channels
        :param out_channels: Output channels
        :param layers: hidden layers
        :param dilations: Dilations choices
        :param LL_alpha: Controls distribution of radius of hidden layer
                         submatrix
        :param LL_gamma: Controls average degree of hidden layer submatrix
        :param LL_max_degree: set the maximum degree of hidden layer nodes
        :param IL_target: Sparsity of input to hidden layer submatrix
        :param LO_target: Sparsity of hidden layer to output submatrix
        :param IO_target: Boolean choice of input to output feedthrough

        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers = layers
        self.dilation_choices = dilations

        self.LL_alpha = LL_alpha
        self.LL_gamma = LL_gamma
        self.LL_max_degree = LL_max_degree
        self.LL_min_degree = LL_min_degree
        self.IL_target = IL_target
        self.IO_target = IO_target
        self.LO_target = LO_target

        # self.these_dilation = self.draw_dilations()

    def draw_dilations(self):
        """
        Draws dilations
        :return: Return a random dilation vector
        """

        tmp = np.random.choice(self.dilation_choices,
                               self.layers,
                               replace=True
                               )
        source_dilations = self.in_channels * [0]
        sink_dilations = self.out_channels * [1]
        tmp = np.hstack([source_dilations, tmp, sink_dilations])
        return tmp

    def build_matrix(self, return_numpy=True):
        """
        Build full random matrix given specifications
        :param return_numpy: return a numpy array
        :type return_numpy: bool
        :return: the graph
        """

        IL = np.zeros((self.in_channels, self.layers))  #
        IO = np.zeros((self.in_channels, self.out_channels))
        LO = np.zeros((self.layers, self.out_channels))

        #########################
        # LL MATRIX CONSTRUCTION
        #########################
        if self.LL_gamma is None:
            LL = np.triu(np.ones((self.layers, self.layers)))
            for ii in range(self.layers):
                LL[ii, ii] = 0
        else:
            LL, _, _ = graph_utils.random_LL_graph(self.layers,
                                                   self.LL_alpha,
                                                   self.LL_gamma,
                                                   self.LL_min_degree,
                                                   self.LL_max_degree
                                                   )

        ##################
        # GRAPH ANALYSIS #
        ##################

        # we now need to identify the 'must have' starting and ending points
        column_sums = np.sum(LL, axis=0)
        row_sums = np.sum(LL, axis=1)
        cols = column_sums == 0
        rows = row_sums != 0
        must_have_inputs = np.where(cols * rows)[0]
        must_have_outputs = np.where(~cols * ~rows)[0]
        in_between_layers = np.where(~cols * rows)[0]
        IL_skip = np.hstack([in_between_layers, must_have_outputs])
        LO_skip = np.hstack([must_have_inputs, in_between_layers])

        #########################
        # IL MATRIX CONSTRUCTION
        #########################

        # CHOICE: ALL INPUT CHANNELS CONNECT TO ALL MANDATORY INPUTS
        IL[:, must_have_inputs] = 1

        # Now we pick at random other connections from I to L
        # Again, I don't want to favor one input channel over another
        # So I just connect all of them
        N_picks = int(len(IL_skip) * self.IL_target + 0.5)
        choices = IL_skip
        these_layers = np.random.choice(choices, N_picks, False)
        IL[:, these_layers] = 1

        #########################
        # LO MATRIX CONSTRUCTION
        #########################

        # Now we need to identify the layer to output layer
        # All Mandatory outputs are linked to all output channels
        LO[must_have_outputs, :] = 1

        # Randomize skips as before
        N_picks = int(len(LO_skip) * self.LO_target + 0.5)
        choices = LO_skip
        these_layers = np.random.choice(choices, N_picks, False)
        LO[these_layers, :] = 1

        #########################
        # IO MATRIX CONSTRUCTION
        #########################

        # As before, I don't want to favor one channel over another
        # this is equivalent to either connecting input channels
        # to output or not.
        if self.IO_target:
            IO = IO + 1

        full_matrix = np.zeros((self.in_channels + self.layers + self.out_channels,
                                self.in_channels + self.layers + self.out_channels))

        full_matrix[0:self.in_channels, self.in_channels:self.in_channels + self.layers] = IL

        full_matrix[0:self.in_channels, self.in_channels + self.layers:] = IO

        full_matrix[
        self.in_channels:self.in_channels + self.layers, self.in_channels:self.in_channels + self.layers] = LL

        full_matrix[self.in_channels:self.in_channels + self.layers, self.in_channels + self.layers:] = LO

        # Something is amiss here... return_numpy = False returns G, which is
        # densely connected
        if not return_numpy:
            G = nx.from_numpy_array(full_matrix, create_using=nx.DiGraph())
            return G

        return full_matrix


def tst():
    print("OK")


if __name__ == "__main__":
    tst()
