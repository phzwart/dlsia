import einops
import numpy as np
import torch
import tqdm
import zarr
from qlty import cleanup
from qlty import qlty2D


# TODO: should perhaps finds its way to qlty
def step_overlap_suggestions(N, W, target):
    suggestions = []
    steps = []
    for ii in range(1, W + 1):
        if (N - W) % ii == 0:
            suggestions.append(ii)
            steps.append(ii)
    steps = np.array(steps)
    overlap = 1.0 - np.array(suggestions) / W
    sel = overlap > target
    steps = steps[sel]
    return steps[-1]


def N_steps(N, W, step):
    delta = (N - (W - step)) // step
    return delta


def choose_window_and_step(N, min_window, max_window, overlap):
    boxes = []
    windows = []
    steps = []
    for W in range(min_window, max_window + 1):
        step = step_overlap_suggestions(N, W, overlap)
        D = N_steps(N, W, step)
        boxes.append(D)
        windows.append(W)
        steps.append(step)
    windows = np.array(windows)
    boxes = np.array(boxes)
    steps = np.array(steps)

    this_one = np.argmin(boxes)
    return windows[this_one], steps[this_one]


def slice_equalizer(wz, wy, wx):
    smallest = min((wz, wy, wx))
    return slice(0, smallest, 1)


def slice_detector(labels, threshold=50, missing_label=-1):
    Z, Y, X = labels.shape
    these_z = []
    these_y = []
    these_x = []

    for ii in range(Z):
        tmp = labels[ii, ...]
        sel = tmp != missing_label
        there = np.sum(sel)
        if there > threshold:
            these_z.append(ii)

    for ii in range(Y):
        tmp = labels[:, ii, :]
        sel = tmp != missing_label
        there = np.sum(sel)
        if there > threshold:
            these_y.append(ii)

    for ii in range(X):
        tmp = labels[:, :, ii]
        sel = tmp != missing_label
        there = np.sum(sel)
        if there > threshold:
            these_x.append(ii)

    return np.array(these_z), np.array(these_y), np.array(these_x),


class OrthoQuilt(object):
    def __init__(self,
                 Z,
                 Y,
                 X,
                 min_window,
                 max_window,
                 min_overlap=0.10,
                 border=None):

        self.Z = Z
        self.Y = Y
        self.X = X
        self.border = border
        if self.border is not None:
            self.border = (border, border)

        self.wz, self.sz = choose_window_and_step(self.Z, min_window, max_window, min_overlap)
        self.wy, self.sy = choose_window_and_step(self.Y, min_window, max_window, min_overlap)
        self.wx, self.sx = choose_window_and_step(self.X, min_window, max_window, min_overlap)

        self.window = (self.wz, self.wy, self.wx)
        self.step = (self.sz, self.sy, self.sx)

        self.qz = qlty2D.NCYXQuilt(Y=self.Y,
                                   X=self.X,
                                   window=(self.wy, self.wx),
                                   step=(self.sy, self.sx),
                                   border=self.border,
                                   border_weight=1.0)

        self.qy = qlty2D.NCYXQuilt(Y=self.Z,
                                   X=self.X,
                                   window=(self.wz, self.wx),
                                   step=(self.sz, self.sx),
                                   border=self.border,
                                   border_weight=1.0)

        self.qx = qlty2D.NCYXQuilt(Y=self.Z,
                                   X=self.Y,
                                   window=(self.wz, self.wy),
                                   step=(self.sz, self.sy),
                                   border=self.border,
                                   border_weight=1.0
                                   )

        self.equi_slicer = slice_equalizer(self.wz, self.wy, self.wx)

        self.cache = None
        self.network = None
        self.device = None

    def process_and_clean_pairs(self,
                                data,
                                labels,
                                missing_label=-1,
                                slice_detection=True,
                                slice_threshold=100
                                ):
        print(data.shape, labels.shape)
        data_chops = []
        lbls_chops = []

        if slice_detection:
            these_Z, these_Y, these_X = slice_detector(labels, slice_threshold)
        else:
            these_Z = np.arange(self.Z)
            these_Y = np.arange(self.Y)
            these_X = np.arange(self.X)

        for ii in these_Z:
            data_slice = torch.Tensor(data[:, ii, :, :]).unsqueeze(0)
            lbl_slice = torch.tensor(labels[ii, :, :]).unsqueeze(0)
            img, lbl = self.qz.unstitch_data_pair(data_slice, lbl_slice)
            img, lbl, _ = cleanup.weed_sparse_classification_training_pairs_2D(
                img, lbl, missing_label=missing_label, border_tensor=self.qz.border_tensor())

            img = img[:, :, self.equi_slicer, self.equi_slicer]
            lbl = lbl[:, self.equi_slicer, self.equi_slicer]
            data_chops.append(img)
            lbls_chops.append(lbl)

        for ii in these_Y:
            data_slice = torch.Tensor(data[:, :, ii, :]).unsqueeze(0)
            lbl_slice = torch.tensor(labels[:, ii, :]).unsqueeze(0)
            img, lbl = self.qy.unstitch_data_pair(data_slice, lbl_slice)
            img, lbl, _ = cleanup.weed_sparse_classification_training_pairs_2D(
                img, lbl, missing_label=missing_label, border_tensor=self.qy.border_tensor())
            img = img[:, :, self.equi_slicer, self.equi_slicer]
            lbl = lbl[:, self.equi_slicer, self.equi_slicer]
            data_chops.append(img)
            lbls_chops.append(lbl)

        for ii in these_X:
            data_slice = torch.Tensor(data[:, :, :, ii]).unsqueeze(0)
            lbl_slice = torch.tensor(labels[:, :, ii]).unsqueeze(0)
            img, lbl = self.qx.unstitch_data_pair(data_slice, lbl_slice)
            img, lbl, _ = cleanup.weed_sparse_classification_training_pairs_2D(
                img, lbl, missing_label=missing_label, border_tensor=self.qx.border_tensor())
            img = img[:, :, self.equi_slicer, self.equi_slicer]
            lbl = lbl[:, self.equi_slicer, self.equi_slicer]
            data_chops.append(img)
            lbls_chops.append(lbl)

        data_chops = torch.concat(data_chops, dim=0)
        lbls_chops = torch.concat(lbls_chops, dim=0)
        return data_chops, lbls_chops

    def setup_ortho_cache(self, my_cache_name, channels, chunks=None):
        store = zarr.DirectoryStore(my_cache_name)
        shape = (3, channels, self.Z, self.Y, self.X)
        dtype = 'float32'
        if chunks is None:
            chunks = (1, channels, self.sz, self.sy, self.sx)
        self.cache = zarr.create(shape=shape,
                                 dtype=dtype,
                                 store=store,
                                 chunks=chunks,
                                 overwrite=True
                                 )

    def setup_network(self, network, device):
        self.network = network.to(device)
        self.device = device

    def slicer_x(self, CZYX_zarr, this_slice, batch_size):
        img = torch.tensor(CZYX_zarr[:, :, :, this_slice])
        img = einops.rearrange(img, "Cin Z Y X -> X Cin Z Y")
        q_imgs = self.qx.unstitch(img)
        with torch.no_grad():
            torch.cuda.empty_cache()
            n = len(q_imgs)  # Total number of images
            n_batches = (n + batch_size - 1) // batch_size  # Calculate the number of batches
            results = []  # List to store the individual result tensors
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n)
                mini_batch = q_imgs[start_idx:end_idx]
                mini_batch_result = self.network.to(self.device)(mini_batch.to(self.device)).cpu()
                results.append(mini_batch_result)
            q_results = torch.cat(results, dim=0)
            q_results = torch.nn.Softmax(1)(q_results)
        results, _ = self.qx.stitch(q_results)
        results = einops.rearrange(results, "X Cout Z Y -> Cout Z Y X")

        self.cache[2, :, :, :, this_slice] = results.numpy()

    def slicer_y(self, CZYX_zarr, this_slice, batch_size):
        img = torch.tensor(CZYX_zarr[:, :, this_slice, :])
        img = einops.rearrange(img, "Cin Z Y X -> Y Cin Z X")
        q_imgs = self.qy.unstitch(img)
        with torch.no_grad():
            torch.cuda.empty_cache()
            n = len(q_imgs)  # Total number of images
            n_batches = (n + batch_size - 1) // batch_size  # Calculate the number of batches
            results = []  # List to store the individual result tensors
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n)
                mini_batch = q_imgs[start_idx:end_idx]
                mini_batch_result = self.network.to(self.device)(mini_batch.to(self.device)).cpu()
                results.append(mini_batch_result)
            q_results = torch.cat(results, dim=0)
            q_results = torch.nn.Softmax(1)(q_results)
        results, _ = self.qy.stitch(q_results)
        results = einops.rearrange(results, "Y Cout Z X -> Cout Z Y X")
        self.cache[1, :, :, this_slice, :] = results.numpy()

    def slicer_z(self, CZYX_zarr, this_slice, batch_size):
        img = torch.tensor(CZYX_zarr[:, this_slice, :, :])
        img = einops.rearrange(img, "Cin Z Y X -> Z Cin Y X")
        q_imgs = self.qz.unstitch(img)
        with torch.no_grad():
            torch.cuda.empty_cache()
            n = len(q_imgs)  # Total number of images
            n_batches = (n + batch_size - 1) // batch_size  # Calculate the number of batches
            results = []  # List to store the individual result tensors
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n)
                mini_batch = q_imgs[start_idx:end_idx]
                mini_batch_result = self.network.to(self.device)(mini_batch.to(self.device)).cpu()
                results.append(mini_batch_result)
            q_results = torch.cat(results, dim=0)
            q_results = torch.nn.Softmax(1)(q_results)
        results, _ = self.qz.stitch(q_results)
        results = einops.rearrange(results, "Z Cout Y X -> Cout Z Y X")
        self.cache[0, :, this_slice, :, :] = results.numpy()

    def inference(self,
                  zarr_data,
                  chunk_size_z=10,
                  chunk_size_y=10,
                  chunk_size_x=10,
                  batch_size=16
                  ):

        NZ = zarr_data.shape[-3]
        NY = zarr_data.shape[-2]
        NX = zarr_data.shape[-1]

        assert NZ == self.Z
        assert NY == self.Y
        assert NX == self.X

        # Iterate through chunks in Z dimension
        for ii in tqdm.tqdm(range(0, NZ, chunk_size_z)):
            this_slice = slice(ii, min(ii + chunk_size_z, NZ))
            self.slicer_z(zarr_data, this_slice, batch_size)

        # Iterate through chunks in Y dimension
        for ii in tqdm.tqdm(range(0, NY, chunk_size_y)):
            this_slice = slice(ii, min(ii + chunk_size_y, NY))
            self.slicer_y(zarr_data, this_slice, batch_size)

        # Iterate through chunks in X dimension
        for ii in tqdm.tqdm(range(0, NX, chunk_size_x)):
            this_slice = slice(ii, min(ii + chunk_size_x, NX))
            self.slicer_x(zarr_data, this_slice, batch_size)

    def inference_ensemble(self,
                           networks,
                           zarr_data,
                           cache_base,
                           chunk_size_z=10,
                           chunk_size_y=10,
                           chunk_size_x=10,
                           batch_size=16,
                           device='cuda:0'
                           ):
        names = []
        for ii in range(len(networks)):
            this_name = cache_base + "_ensemble_%i.zarr" % ii
            names.append(this_name)
            self.setup_ortho_cache(this_name, networks[ii].out_channels)
            self.setup_network(networks[ii].cpu(), device=device)
            self.inference(zarr_data, chunk_size_z, chunk_size_y, chunk_size_x, batch_size)
