import h5py
import numpy as np
import os
from torch.utils import data
from torchvision import transforms

# solves OMP: Error #15: Initializing libiomp5.dylib, but found libomp.dylib already initialized.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class Hdf5Dataset2D(data.Dataset):
    """
    A pytorch compatible dataset object with input and output based on a single
    hdf5 file. To be used in tandem with the dlsia 2d data generators
    """

    def __init__(self, filename, x_label, y_label, transform=None, max_size=None):
        """
        Load data for pytorch from a hdf5 file, needs a stack of 2D images.

        Following are list of x and y labels:
            -trax_GT: ground truth
            -trax_mask: binary array indicating 1 if peak and 0 if background
            -trax_noisy: Gaussian or unifom noise added to image
            -trax_noisy_norm: noisy image transformed or normalized (typically
                              scaled to interval [-1,1]
            -trax_classes: two-channel binary images classifying foreground
                           and background

        :param filename: the hdf5 filename
        :param x_label: the path / label for the x datas (input)
        :param y_label: the path / label for the y data (output)
        :param transform: a pytroch transform operator. If none it's just
                         'to tensor'
        :param max_size: if specified to some number, only the top max_size
                         entries will be used by cheating the len function
        """

        self.filename = filename
        self.x_label = x_label
        self.y_label = y_label
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])

        f = h5py.File(self.filename, "r")
        x_obj = f[x_label]
        self.shape = x_obj.shape
        assert (len(self.shape)) == 3
        self.len = self.shape[0]
        if max_size is not None:
            assert max_size < self.shape[0]
            self.len = max_size
        f.close()

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        f = h5py.File(self.filename, "r")
        x_obj = f[self.x_label]
        y_obj = f[self.y_label]
        these_x = np.array(x_obj[index, :, :][()])
        these_y = np.array(y_obj[index, :, :][()])

        # Must swap dimensions if loading 2 channel classes
        if self.y_label == 'trax_classes':
            these_y = np.einsum('kij->ijk', these_y)

        f.close()

        return self.transform(these_x), self.transform(these_y)


class Hdf5Dataset2DClasses(data.Dataset):
    """
    A pytorch compatible dataset object with input and output based on a single
    hdf5 file. To be used in tandem with the noisy_2d scripts
    """

    def __init__(self, filename, x_label, y_label, transform=None, max_size=None):
        """
        Load data for pytorch from a hdf5 file, needs a stack of 2D images and
        a stack of class labels

        :param filename: the hdf5 filename
        :param x_label: the path / label for the x datas (input)
        :param y_label: the path / label for the y data (output) This should be
                        a (M,K,n_xy,n_xy) array
        :param transform: a pytorch transform operator. If none it's just
                         'to tensor'
        :param max_size: if specified to some number, only the top max_size
                         entries will be used by cheating the len function
        """

        self.filename = filename
        self.x_label = x_label
        self.y_label = y_label
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])

        f = h5py.File(self.filename, "r")
        x_obj = f[x_label]
        self.shape = x_obj.shape
        assert (len(self.shape)) == 3
        self.len = self.shape[0]
        if max_size is not None:
            assert max_size < self.shape[0]
            self.len = max_size
        f.close()

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        f = h5py.File(self.filename, "r")
        x_obj = f[self.x_label]
        y_obj = f[self.y_label]
        these_x = np.array(x_obj[index, :, :][()])
        these_y = np.array(y_obj[index, :, :, :][()])

        # Must swap dimensions if loading 2 channel classes
        these_y = np.einsum('kij->ijk', these_y)

        f.close()

        return self.transform(these_x), self.transform(these_y)


class Hdf5Dataset2Dtime(data.Dataset):
    """
    A pytorch compatible dataset object with input and output based on a single
    hdf5 file. To be used with the 2D + time simulation code.
    """

    def __init__(self, filename, x_label, y_label, transform=None,
                 max_size=None, time_point=None):
        """
        Load data for pytorch from a hdf5 file, needs a stack of 2D+time images.

        :param filename: the hdf5 filename
        :param x_label: the path / label for the x datas (input)
        :param y_label: the path / label for the y data (output)
        :param transform: a pytorch transform operator. If none it's just
                         'to tensor'
        :param max_size: if specified to some number, only the top max_size
                         entries will be used by cheating the len function
        :param time_point: if None, all time points are used, else a single
                           index or slice can be given
        """

        self.filename = filename
        self.x_label = x_label
        self.y_label = y_label
        self.transform = transform
        self.time_point = time_point

        if self.transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])

        f = h5py.File(self.filename, "r")
        x_obj = f[x_label]
        self.shape = x_obj.shape
        assert (len(self.shape)) == 4

        if self.time_point is None:
            self.time_point = slice(0, self.shape[2], 1)

        self.len = self.shape[0]
        if max_size is not None:
            assert max_size < self.shape[0]
            self.len = max_size
        f.close()

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        f = h5py.File(self.filename, "r")
        x_obj = f[self.x_label]
        y_obj = f[self.y_label]

        these_x = np.array(x_obj[index, self.time_point, :, :][()])
        these_y = np.array(y_obj[index, self.time_point, :, :][()])

        f.close()

        # permute dimensions so tensor transform returns dimensions [num_channels, x, y]
        if type(self.time_point).__name__ == 'slice':
            these_x = np.transpose(these_x, (1, 2, 0))
            these_y = np.transpose(these_y, (1, 2, 0))
        # elif (len(self. time_point) != 1):
        # these_x = np.transpose(these_x, (1, 2, 0))
        # these_y = np.transpose(these_y, (1, 2, 0))

        return self.transform(these_x), self.transform(these_y)


def tst():
    f_train = "train_data_2d.hdf5"
    g_train = "train_data_2d_time.hdf5"
    assert os.path.isfile(f_train)
    assert os.path.isfile(g_train)

    f_obj = Hdf5Dataset2D(filename=f_train, x_label="trax_obs", y_label="trax_GT")
    g_obj = Hdf5Dataset2Dtime(filename=g_train, x_label="trax_obs", y_label="trax_GT")
    assert len(f_obj) == len(g_obj)


if __name__ == "__main__":
    tst()
