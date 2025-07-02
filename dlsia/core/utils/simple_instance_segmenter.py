import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy.spatial.distance import cdist


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
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
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


def merge_close_instances(instance_labels, min_distance, max_iterations=100):
    """
    Iteratively merge instance segmented objects that are closer than min_distance.
    Continues until no more merges are possible or max_iterations is reached.
    
    :param instance_labels: input instance segmentation labels (0 for background, >0 for instances)
    :param min_distance: minimum distance threshold for merging instances
    :param max_iterations: maximum number of iterations to prevent infinite loops
    :return: merged instance labels
    """
    if instance_labels is None or instance_labels.size == 0:
        return instance_labels
    
    # Create a copy to avoid modifying the original
    merged_labels = instance_labels.copy()
    
    for iteration in range(max_iterations):
        # Get unique instance IDs (excluding background 0)
        unique_instances = np.unique(merged_labels)
        unique_instances = unique_instances[unique_instances > 0]
        
        if len(unique_instances) <= 1:
            break  # No more instances to merge
        
        # Calculate centroids for each instance
        centroids = []
        instance_ids = []
        
        for instance_id in unique_instances:
            # Find coordinates of current instance
            coords = np.where(merged_labels == instance_id)
            if len(coords[0]) > 0:
                # Calculate centroid
                centroid = np.array([np.mean(coords[0]), np.mean(coords[1])])
                centroids.append(centroid)
                instance_ids.append(instance_id)
        
        if len(centroids) <= 1:
            break
        
        # Convert to numpy array for distance calculation
        centroids = np.array(centroids)
        
        # Calculate pairwise distances between all centroids
        distances = cdist(centroids, centroids)
        
        # Set diagonal to infinity to avoid self-comparison
        np.fill_diagonal(distances, np.inf)
        
        # Find pairs of instances that are closer than min_distance
        merge_pairs = []
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                if distances[i, j] < min_distance:
                    merge_pairs.append((instance_ids[i], instance_ids[j]))
        
        if not merge_pairs:
            break  # No more merges possible
        
        # Perform merges
        for instance_id_1, instance_id_2 in merge_pairs:
            # Merge instance_id_2 into instance_id_1
            merged_labels[merged_labels == instance_id_2] = instance_id_1
        
        # Relabel to ensure consecutive numbering
        merged_labels = relabel_instances(merged_labels)
    
    return merged_labels


def relabel_instances(instance_labels):
    """
    Relabel instances to ensure consecutive numbering starting from 1.
    
    :param instance_labels: input instance labels
    :return: relabeled instance labels
    """
    if instance_labels is None or instance_labels.size == 0:
        return instance_labels
    
    # Get unique labels (excluding background 0)
    unique_labels = np.unique(instance_labels)
    unique_labels = unique_labels[unique_labels > 0]
    
    # Create mapping from old labels to new consecutive labels
    label_mapping = {0: 0}  # Keep background as 0
    for new_id, old_id in enumerate(unique_labels, start=1):
        label_mapping[old_id] = new_id
    
    # Apply mapping
    relabeled = np.zeros_like(instance_labels)
    for old_id, new_id in label_mapping.items():
        relabeled[instance_labels == old_id] = new_id
    
    return relabeled


def get_object_bounding_boxes(instance_labels, edge_extension=5):
    """
    Generate bounding boxes around objects with user-defined edge extension.
    Yields object label and slicer coordinates for each object.
    
    :param instance_labels: input instance segmentation labels (0 for background, >0 for instances)
    :param edge_extension: number of pixels to extend the bounding box in each direction
    :yield: tuple of (object_label, x_slice, y_slice) for each object
    """
    if instance_labels is None or instance_labels.size == 0:
        return
    
    # Get unique instance IDs (excluding background 0)
    unique_instances = np.unique(instance_labels)
    unique_instances = unique_instances[unique_instances > 0]
    
    for instance_id in unique_instances:
        # Find coordinates of current instance
        coords = np.where(instance_labels == instance_id)
        
        if len(coords[0]) == 0:
            continue
        
        # Calculate bounding box
        y_min, y_max = np.min(coords[0]), np.max(coords[0])
        x_min, x_max = np.min(coords[1]), np.max(coords[1])
        
        # Apply edge extension
        y_min_extended = max(0, y_min - edge_extension)
        y_max_extended = min(instance_labels.shape[0] - 1, y_max + edge_extension)
        x_min_extended = max(0, x_min - edge_extension)
        x_max_extended = min(instance_labels.shape[1] - 1, x_max + edge_extension)
        
        # Create slicer objects
        y_slice = slice(y_min_extended, y_max_extended + 1)
        x_slice = slice(x_min_extended, x_max_extended + 1)
        
        yield instance_id, x_slice, y_slice


def get_object_bounding_boxes_with_info(instance_labels, edge_extension=5):
    """
    Generate bounding boxes around objects with user-defined edge extension.
    Returns a list of dictionaries with object information and slicer coordinates.
    
    :param instance_labels: input instance segmentation labels (0 for background, >0 for instances)
    :param edge_extension: number of pixels to extend the bounding box in each direction
    :return: list of dictionaries containing object information
    """
    if instance_labels is None or instance_labels.size == 0:
        return []
    
    # Get unique instance IDs (excluding background 0)
    unique_instances = np.unique(instance_labels)
    unique_instances = unique_instances[unique_instances > 0]
    
    object_info_list = []
    
    for instance_id in unique_instances:
        # Find coordinates of current instance
        coords = np.where(instance_labels == instance_id)
        
        if len(coords[0]) == 0:
            continue
        
        # Calculate bounding box
        y_min, y_max = np.min(coords[0]), np.max(coords[0])
        x_min, x_max = np.min(coords[1]), np.max(coords[1])
        
        # Apply edge extension
        y_min_extended = max(0, y_min - edge_extension)
        y_max_extended = min(instance_labels.shape[0] - 1, y_max + edge_extension)
        x_min_extended = max(0, x_min - edge_extension)
        x_max_extended = min(instance_labels.shape[1] - 1, x_max + edge_extension)
        
        # Create slicer objects
        y_slice = slice(y_min_extended, y_max_extended + 1)
        x_slice = slice(x_min_extended, x_max_extended + 1)
        
        # Calculate object properties
        object_area = len(coords[0])
        bbox_width = x_max_extended - x_min_extended + 1
        bbox_height = y_max_extended - y_min_extended + 1
        
        object_info = {
            'label': instance_id,
            'x_slice': x_slice,
            'y_slice': y_slice,
            'original_bbox': (x_min, y_min, x_max, y_max),
            'extended_bbox': (x_min_extended, y_min_extended, x_max_extended, y_max_extended),
            'area': object_area,
            'bbox_width': bbox_width,
            'bbox_height': bbox_height,
            'centroid': (np.mean(coords[1]), np.mean(coords[0]))  # (x, y) format
        }
        
        object_info_list.append(object_info)
    
    return object_info_list


def extract_object_region(image, instance_labels, object_label, edge_extension=5):
    """
    Extract a specific object region from an image using bounding box with edge extension.
    
    :param image: input image (can be any 2D array)
    :param instance_labels: instance segmentation labels
    :param object_label: specific object label to extract
    :param edge_extension: number of pixels to extend the bounding box
    :return: tuple of (object_region, x_slice, y_slice) or None if object not found
    """
    for label, x_slice, y_slice in get_object_bounding_boxes(instance_labels, edge_extension):
        if label == object_label:
            object_region = image[y_slice, x_slice]
            return object_region, x_slice, y_slice
    
    return None


def extract_object_region_fixed_size(image, instance_labels, object_label, output_size=(64, 64), edge_extension=5, fill_value=0):
    """
    Extract a specific object region from an image with a fixed output size.
    The object is centered within the output array with optional edge extension.
    
    :param image: input image (can be any 2D or 3D array)
    :param instance_labels: instance segmentation labels
    :param object_label: specific object label to extract
    :param output_size: tuple of (width, height) for the output region
    :param edge_extension: number of pixels to extend the bounding box
    :param fill_value: value to fill areas outside the image boundaries
    :return: tuple of (object_region, x_offset, y_offset) or None if object not found
    """
    if len(output_size) != 2:
        raise ValueError("output_size must be a tuple of (width, height)")
    
    output_width, output_height = output_size
    
    # Handle multi-channel images
    if len(image.shape) == 3:
        # Multi-channel image (e.g., RGB)
        channels = image.shape[2]
        output_shape = (output_height, output_width, channels)
    else:
        # Single-channel image
        output_shape = (output_height, output_width)
    
    # Find the object's bounding box
    for label, x_slice, y_slice in get_object_bounding_boxes(instance_labels, edge_extension):
        if label == object_label:
            # Calculate the center of the object's bounding box
            obj_center_x = (x_slice.start + x_slice.stop - 1) // 2
            obj_center_y = (y_slice.start + y_slice.stop - 1) // 2
            
            # Calculate the output region bounds
            x_start = obj_center_x - output_width // 2
            x_end = x_start + output_width
            y_start = obj_center_y - output_height // 2
            y_end = y_start + output_height
            
            # Create output array filled with fill_value
            output_region = np.full(output_shape, fill_value, dtype=image.dtype)
            
            # Calculate the valid region within the image bounds
            img_x_start = max(0, x_start)
            img_x_end = min(image.shape[1], x_end)
            img_y_start = max(0, y_start)
            img_y_end = min(image.shape[0], y_end)
            
            # Calculate the corresponding region in the output array
            out_x_start = img_x_start - x_start
            out_x_end = out_x_start + (img_x_end - img_x_start)
            out_y_start = img_y_start - y_start
            out_y_end = out_y_start + (img_y_end - img_y_start)
            
            # Copy the valid region from the image to the output
            if img_x_end > img_x_start and img_y_end > img_y_start:
                output_region[out_y_start:out_y_end, out_x_start:out_x_end] = \
                    image[img_y_start:img_y_end, img_x_start:img_x_end]
            
            # Calculate offsets for reference
            x_offset = x_start
            y_offset = y_start
            
            return output_region, x_offset, y_offset
    
    return None


def get_object_bounding_boxes_fixed_size(instance_labels, output_size=(64, 64), edge_extension=5):
    """
    Generate fixed-size bounding boxes around objects with user-defined edge extension.
    Yields object label, fixed-size region, and offset information for each object.
    
    :param instance_labels: input instance segmentation labels (0 for background, >0 for instances)
    :param output_size: tuple of (width, height) for the output region
    :param edge_extension: number of pixels to extend the bounding box in each direction
    :yield: tuple of (object_label, x_offset, y_offset, output_width, output_height) for each object
    """
    if instance_labels is None or instance_labels.size == 0:
        return
    
    if len(output_size) != 2:
        raise ValueError("output_size must be a tuple of (width, height)")
    
    output_width, output_height = output_size
    
    # Get unique instance IDs (excluding background 0)
    unique_instances = np.unique(instance_labels)
    unique_instances = unique_instances[unique_instances > 0]
    
    for instance_id in unique_instances:
        # Find coordinates of current instance
        coords = np.where(instance_labels == instance_id)
        
        if len(coords[0]) == 0:
            continue
        
        # Calculate bounding box
        y_min, y_max = np.min(coords[0]), np.max(coords[0])
        x_min, x_max = np.min(coords[1]), np.max(coords[1])
        
        # Apply edge extension
        y_min_extended = max(0, y_min - edge_extension)
        y_max_extended = min(instance_labels.shape[0] - 1, y_max + edge_extension)
        x_min_extended = max(0, x_min - edge_extension)
        x_max_extended = min(instance_labels.shape[1] - 1, x_max + edge_extension)
        
        # Calculate the center of the extended bounding box
        obj_center_x = (x_min_extended + x_max_extended) // 2
        obj_center_y = (y_min_extended + y_max_extended) // 2
        
        # Calculate the output region bounds
        x_offset = obj_center_x - output_width // 2
        y_offset = obj_center_y - output_height // 2
        
        yield instance_id, x_offset, y_offset, output_width, output_height


def extract_all_objects_fixed_size(image, instance_labels, output_size=(64, 64), edge_extension=5, fill_value=0):
    """
    Extract all objects from an image with fixed-size regions.
    
    :param image: input image (can be any 2D or 3D array)
    :param instance_labels: instance segmentation labels
    :param output_size: tuple of (width, height) for the output region
    :param edge_extension: number of pixels to extend the bounding box
    :param fill_value: value to fill areas outside the image boundaries
    :return: list of tuples (object_label, object_region, x_offset, y_offset)
    """
    if len(output_size) != 2:
        raise ValueError("output_size must be a tuple of (width, height)")
    
    output_width, output_height = output_size
    
    # Handle multi-channel images
    if len(image.shape) == 3:
        # Multi-channel image (e.g., RGB)
        channels = image.shape[2]
        output_shape = (output_height, output_width, channels)
    else:
        # Single-channel image
        output_shape = (output_height, output_width)
    
    extracted_objects = []
    
    for obj_label, x_offset, y_offset, out_w, out_h in get_object_bounding_boxes_fixed_size(
        instance_labels, output_size, edge_extension):
        
        # Create output array filled with fill_value
        output_region = np.full(output_shape, fill_value, dtype=image.dtype)
        
        # Calculate the valid region within the image bounds
        img_x_start = max(0, x_offset)
        img_x_end = min(image.shape[1], x_offset + output_width)
        img_y_start = max(0, y_offset)
        img_y_end = min(image.shape[0], y_offset + output_height)
        
        # Calculate the corresponding region in the output array
        out_x_start = img_x_start - x_offset
        out_x_end = out_x_start + (img_x_end - img_x_start)
        out_y_start = img_y_start - y_offset
        out_y_end = out_y_start + (img_y_end - img_y_start)
        
        # Copy the valid region from the image to the output
        if img_x_end > img_x_start and img_y_end > img_y_start:
            output_region[out_y_start:out_y_end, out_x_start:out_x_end] = \
                image[img_y_start:img_y_end, img_x_start:img_x_end]
        
        extracted_objects.append((obj_label, output_region, x_offset, y_offset))
    
    return extracted_objects
