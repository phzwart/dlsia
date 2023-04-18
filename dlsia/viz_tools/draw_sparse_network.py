import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap


def build_sizing_colormaps(these_sizes, these_nodes, stride=2):
    """
    Build a colormap for nodes, taking into account sizing information

    parameters
    ----------
    these_sizes: A list of possible sizing operator (exponents to the stride)
    these_nodes: For each node, the possible sizing operator
    stride: The stride

    Returns
    -------
    a colormap and a set of patches for a matplotlib legend
    """

    n_items = len(these_sizes)
    mycmap = cm.get_cmap('rainbow', n_items)
    v = np.arange(n_items)
    new_map = mycmap(v)
    newcmp = ListedColormap(new_map, n_items)
    cmap = []
    for this_node in these_nodes:
        cmap.append(newcmp(this_node))

    these_patches = []
    for ii in range(n_items):
        lab = these_sizes[ii]
        if lab < 0:
            slab = int(stride ** (-lab))
            slab = "Downsampled by 1/%i " % slab
        elif lab > 0:
            slab = int(stride ** lab)
            slab = "Upsampled by %i" % slab
        else:
            slab = "Input Size"

        this_patch = mpatches.Patch(color=newcmp(ii), label=slab)
        these_patches.append(this_patch)

    return cmap, these_patches


def build_custom_colormap(mapper):
    """
    Build a colormap for a dilation or channel adjacency matrix

    parameters
    ----------
    mapper: a dictionairy that relates dilation or channel to their rank-order (1st, 2nd etc)

    Returns
    -------
    a colormap and a set of patches for a matplotlib legend
    """

    n_items = len(mapper.keys())
    mycmap = cm.get_cmap('rainbow', n_items - 1)
    v = np.arange(n_items - 1)
    new_map = mycmap(v)
    new_map = np.vstack([np.array([1, 1, 1, 1]), new_map])
    newcmp = ListedColormap(new_map, n_items)
    these_patches = []
    for kk in mapper:
        ii = mapper[kk]
        if ii > 0:
            this_patch = mpatches.Patch(color=newcmp(ii), label='%i' % kk)
            these_patches.append(this_patch)
    return newcmp, these_patches


def draw_network(SMSobj, fsize=3):
    """
    Draw the network

    parameters
    ----------
    SMSObj: A SMS Network object
    fsize: image window size

    Returns
    -------
    A network, dilation and channel graph

    """

    plt.rc('font', size=12)
    plt.rc('axes', titlesize=12)
    plt.rc('axes', labelsize=12)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('legend', fontsize=14)
    plt.rc('figure', titlesize=16)

    G = SMSobj.network_graph

    these_sizes = []
    all_sizes = []
    for node in G.nodes:
        this_size = G.nodes[node]["size"]
        all_sizes.append(this_size)
        if this_size not in these_sizes:
            these_sizes.append(this_size)
    these_sizes = np.sort(these_sizes)[::-1]
    for ii in range(len(all_sizes)):
        tmp = all_sizes[ii]
        loc = np.where(these_sizes == tmp)[0][0]
        all_sizes[ii] = loc
    size_map, size_legend = build_sizing_colormaps(these_sizes, all_sizes)

    pos = nx.circular_layout(G)
    network_fig = plt.figure(figsize=(fsize, fsize))
    nx.draw(G, pos=pos, with_labels=True, node_color=size_map)
    plt.legend(handles=size_legend, loc='center')
    plt.title("Network Graph - Layer Dimension")

    gmat = nx.adjacency_matrix(G).todense()
    dilation_mat = np.zeros(gmat.shape, dtype=int)
    channel_mat = np.zeros(gmat.shape, dtype=int)

    edges = G.edges()
    these_dils = []
    all_dils = []
    these_channels = []
    all_channels = []
    for edge in edges:
        dil = int(G.get_edge_data(edge[0], edge[1])["dilation"])
        chan = int(G.get_edge_data(edge[0], edge[1])["channels"])
        dilation_mat[edge[0], edge[1]] = dil
        channel_mat[edge[0], edge[1]] = chan
        # book keeping
        all_dils.append(dil)
        all_channels.append(chan)
        if dil not in these_dils:
            these_dils.append(dil)
        if chan not in these_channels:
            these_channels.append(chan)

    dil_mapper = {0: 0}
    these_dils = np.sort(these_dils)
    for ii in range(len(these_dils)):
        dil_mapper[these_dils[ii]] = int(ii + 1)

    chan_mapper = {0: 0}
    these_channels = np.sort(these_channels)
    for ii in range(len(these_channels)):
        chan_mapper[these_channels[ii]] = int(ii + 1)

    for edge in edges:
        ii = edge[0]
        jj = edge[1]
        dilation_mat[ii, jj] = dil_mapper[dilation_mat[ii, jj]]
        channel_mat[ii, jj] = chan_mapper[channel_mat[ii, jj]]

    N_nodes = gmat.shape[0]
    step = 4
    if N_nodes < 25:
        step = 3
    if N_nodes < 20:
        step = 2
    if N_nodes < 15:
        step = 1

    ticks = np.arange(1, N_nodes + 1, step).round(0)

    dil_cmp, patches = build_custom_colormap(dil_mapper)
    dil_fig = plt.figure(figsize=(fsize + 1, fsize + 1))
    plt.imshow(dilation_mat,
               cmap=dil_cmp,
               extent=(1 - 0.5, N_nodes + 0.5, N_nodes + 0.5, 1 - 0.5),
               interpolation='none')
    plt.ylabel("From Node")
    plt.xlabel("To Node")
    plt.title("Adjency Matrix - Dilations")
    plt.xticks(ticks)
    plt.yticks(ticks[::-1])
    plt.legend(handles=patches, loc='lower left')

    chan_cmp, patches = build_custom_colormap(chan_mapper)
    chan_fig = plt.figure(figsize=(fsize + 1, fsize + 1))
    plt.imshow(channel_mat,
               cmap=chan_cmp,
               extent=(1 - 0.5, N_nodes + 0.5, N_nodes + 0.5, 1 - 0.5),
               interpolation='none'
               )
    plt.ylabel("From Node")
    plt.xlabel("To Node")
    plt.title("Adjency Matrix - Channels")
    plt.xticks(ticks)
    plt.yticks(ticks[::-1])
    plt.legend(handles=patches, loc='lower left')

    return network_fig, dil_fig, chan_fig
