import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


def instance_segment_binary_map_via_watershed(guessed_labels, min_distance=20, fill_holes=False):
    """
    Instance segmentation via a watershedded map.
    Class labels expected: 1 for background, 2 for foreground.

    :param guessed_labels: input labels
    :param min_distance: minimum distance between object centers
    :param fill_holes: boolean variable that controls hole filling
    :return: instance segmented map.
    """
    bin_map = guessed_labels - 1.0
    bin_map = bin_map.astype(int)

    if fill_holes:
        bin_map = ndi.binary_fill_holes(bin_map)
    # di stance transform
    distance = ndi.distance_transform_edt(bin_map)
    coords = peak_local_max(distance,
                            min_distance=int(min_distance),
                            footprint=np.ones((3, 3)),
                            labels=bin_map)
    print(coords)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    print(markers.shape, distance.shape)
    new_labels = watershed(-distance, markers, mask=bin_map)
    return new_labels


def instance_segment_binary_map_via_watershed_from_external_peak_map(guessed_labels,
                                                                     external_peak_map,
                                                                     absolute_threshold=0.1,
                                                                     smooth_distance_map=3,
                                                                     min_distance=10):
    """
    Instance segmentation via a watershedded map.
    Class labels expected: 1 for background, 2 for foreground.

    :param external_peak_map: an external peak map for watershed
    :param smooth_distance_map: dool that toggle smoothing of map before watershedding
    :param min_distance: minimum distance between peaks
    :param absolute_threshold: threshold for peak picking
    :param guessed_labels: input labels
    :return: instance segmented map.
    """
    bin_map = guessed_labels - 1.0
    bin_map = bin_map.astype(int)

    distance = ndi.distance_transform_edt(bin_map)

    if smooth_distance_map < 1:
        distance = bin_map * (external_peak_map + 1.0)
    else:
        for ii in range(smooth_distance_map):
            distance = ndi.median_filter(distance, size=3)

    coords = peak_local_max(external_peak_map,
                            threshold_abs=absolute_threshold,
                            min_distance=min_distance,
                            labels=bin_map)

    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    new_labels = watershed(-bin_map, markers, mask=bin_map)
    return new_labels


def edge_mask(guessed_labels):
    sel = guessed_labels > 1
    bin_map = np.zeros(guessed_labels.shape)
    bin_map[sel] = 1
    mapa = ndi.binary_dilation(bin_map, iterations=3).astype(int)
    mapb = ndi.binary_erosion(bin_map, iterations=3).astype(int)
    edge_map = mapa - mapb
    return edge_map
