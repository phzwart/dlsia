import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_rgb_and_labels(img, labels, plot_params):
    """
    Make a side-by-side plot of an RGB image and a set of labels

    :param img: The rgb image
    :param labels: the heatmap with labels
    :param plot_params: some mplot parameters in a dict
    :return: a graph object
    """

    fig = make_subplots(rows=1, cols=2)

    fig.add_trace(
        go.Heatmap(px.imshow(labels).data[0]),
        row=1, col=2
    )

    fig.add_trace(
        go.Image(px.imshow(img).data[0]),
        row=1, col=1
    )

    fig.update_yaxes(autorange="reversed", row=1, col=2)
    fig.update_layout(yaxis=dict(scaleanchor='x', constrain=None))
    fig.update_layout(title_text=plot_params["title"])

    # TODO: this is ugly, and basically I don't fully understand how to make a
    # general solution so this is it for now-ish
    # fig["layout"]["yaxis2"]["domain"] = [0.05, 0.95]
    fig.update_layout(
        autosize=False,
        width=900,
        height=450)
    return fig


def plot_heatmap_and_labels(img, labels, plot_params):
    """
    Make a side-by-side plot of an image and a set of labels

    :param img: The rgb image
    :param labels: the heatmap with labels
    :param plot_params: some mplot parameters in a dict
    :return: a graph object
    """

    fig = make_subplots(rows=1, cols=2)

    mu = np.mean(img)
    std = np.std(img)
    zmin = mu - 5 * std
    zmax = mu + 5 * std
    fig.add_trace(
        go.Heatmap(px.imshow(labels).data[0], zmin=zmin, zmax=zmax),
        row=1, col=2
    )

    fig.add_trace(
        go.Heatmap(px.imshow(img).data[0]),
        row=1, col=1
    )

    # fig.update_yaxes(autorange="reversed", row=1, col=2)
    fig.update_layout(yaxis=dict(scaleanchor='x', constrain=None))
    fig.update_layout(title_text=plot_params["title"])
    fig.update_layout(
        autosize=False,
        width=700,
        height=350)
    return fig


def plot_training_results_segmentation(results_dict):
    training_loss = results_dict['Training loss']
    validation_loss = results_dict['Validation loss']
    training_f1 = results_dict['F1 training macro']
    validation_f1 = results_dict['F1 validation macro']
    epoch = np.array(range(len(training_loss)))
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Loss", "F1"])
    fig.add_trace(go.Scatter(x=epoch,
                             y=training_loss,
                             mode='lines+markers',
                             name="Training",
                             line=dict(color='red')), col=1, row=1)
    fig.add_trace(go.Scatter(x=epoch,
                             y=validation_loss,
                             mode='lines+markers',
                             name="Validation",
                             line=dict(color='blue')), col=1, row=1)

    fig.update_yaxes(type="log", col=1, row=1)
    fig.add_trace(go.Scatter(x=epoch,
                             y=training_f1,
                             mode='lines+markers', showlegend=False,
                             line=dict(color='red')), col=2, row=1)
    fig.add_trace(go.Scatter(x=epoch,
                             y=validation_f1,
                             mode='lines+markers', showlegend=False,
                             line=dict(color='blue')), col=2, row=1)

    fig.update_yaxes(type="log", col=1, row=1)

    fig.update_layout(autosize=True)
    return fig


def plot_training_results_regression(results_dict):
    training_loss = results_dict['Training loss']
    validation_loss = results_dict['Validation loss']
    training_cc = results_dict['CC training']
    validation_cc = results_dict['CC validation']
    epoch = np.array(range(len(training_loss)))
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Loss", "CC"])
    fig.add_trace(go.Scatter(x=epoch,
                             y=training_loss,
                             mode='lines+markers',
                             name="Training",
                             line=dict(color='red')), col=1, row=1)
    fig.add_trace(go.Scatter(x=epoch,
                             y=validation_loss,
                             mode='lines+markers',
                             name="Validation",
                             line=dict(color='blue')), col=1, row=1)

    fig.update_yaxes(type="log", col=1, row=1)
    fig.add_trace(go.Scatter(x=epoch,
                             y=training_cc,
                             mode='lines+markers', showlegend=False,
                             line=dict(color='red')), col=2, row=1)
    fig.add_trace(go.Scatter(x=epoch,
                             y=validation_cc,
                             mode='lines+markers', showlegend=False,
                             line=dict(color='blue')), col=2, row=1)

    fig.update_yaxes(type="log", col=1, row=1)

    fig.update_layout(autosize=True)
    return fig


def plot_shapes_data_numpy(shapes_dict):
    """
    This function takes the returned dict of numpy array image bundles from
    test_data/two_d/random_shapes/build_shape_set_numpy and plots the ground
    truth, noisy, and mask for each of the four shape classes

    :param shapes_dict: the shapes data dictionary containing 'GroundTruth,'
                        'Noisy,' 'ClassImage,' and 'Labels' numpy arrays
    """

    gt = shapes_dict['GroundTruth']
    noisy = shapes_dict['Noisy']
    mask = shapes_dict['ClassImage']
    label = shapes_dict['Label']

    # get first rectangle
    rect_id = np.where(label == 1)
    rect_id = rect_id[0][0]

    # get first circle
    circ_id = np.where(label == 2)
    circ_id = circ_id[0][0]

    # get first triangle
    tri_id = np.where(label == 3)
    tri_id = tri_id[0][0]

    # get first annulus
    annu_id = np.where(label == 4)
    annu_id = annu_id[0][0]

    shape_ids = [rect_id, circ_id, tri_id, annu_id]

    # clim_max = np.ceil(np.max(mask))
    clim_max = 4

    col_wid = 0.03
    row_hei = 2.0
    fig = make_subplots(rows=4, cols=3,
                        subplot_titles=['Ground Truth',
                                        'Noise Added',
                                        'Binary Mask'],
                        column_widths=[col_wid, col_wid, col_wid],
                        row_heights=[row_hei, row_hei, row_hei, row_hei],
                        vertical_spacing=.04,
                        horizontal_spacing=.05)
    for j, shape_id in enumerate(shape_ids):
        fig.add_trace(
            px.imshow(gt[shape_id, :, :]).data[0],
            row=j + 1, col=1)
        fig.add_trace(
            px.imshow(noisy[shape_id, :, :]).data[0],
            row=j + 1, col=2)
        fig.add_trace(
            px.imshow(mask[shape_id, :, :]).data[0],
            row=j + 1, col=3)
    fig.update_coloraxes(cmin=0)
    fig.update_coloraxes(cmax=clim_max)
    fig['layout']['yaxis']['title'] = 'Class 1: Rectangle'
    fig['layout']['yaxis4']['title'] = 'Class 2: Circles'
    fig['layout']['yaxis7']['title'] = 'Class 3: Triangles'
    fig['layout']['yaxis10']['title'] = 'Class 4: Annuli'
    fig.update_layout(width=700, height=700)
    # fig.show()
    return fig
