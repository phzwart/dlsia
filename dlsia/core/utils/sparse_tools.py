import torch

def save_sparse_numpy_tensor(numpy_array, file_path):
    """
    Converts a 3D NumPy array into a sparse tensor and saves it.

    Parameters:
    numpy_array (numpy.ndarray): A 3D NumPy array.
    file_path (str): Path where the sparse tensor will be saved.
    """
    # Convert the NumPy array to a PyTorch tensor
    tensor = torch.tensor(numpy_array)

    # Find the indices where the tensor is not zero
    idx = torch.nonzero(tensor, as_tuple=False)

    # Gather the non-zero values
    values = tensor[idx[:, 0], idx[:, 1], idx[:, 2]]

    # Create a sparse tensor
    sparse_tensor = torch.sparse_coo_tensor(idx.t(), values, tensor.size())

    # Save the sparse tensor
    torch.save(sparse_tensor, file_path)