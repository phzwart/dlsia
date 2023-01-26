import h5py
import numpy as np
import tqdm
from dlsia.test_data.three_d import noisy_gauss_3d


# TODO fill out docstrings? EJR (11/29/22


def build_data_standard_sets_3d(n_imgs=100,
                                n_peaks=1,
                                n_xyz=32,
                                snr=0.5,
                                mask_radius=2.0):
    engine_b = noisy_gauss_3d.DataMaker3D(n_peaks=n_peaks,
                                          n_xyz=n_xyz)

    # build a training_set
    # n_imgs = 10000

    f = h5py.File('train_data_3d.hdf5', 'w')
    dset_imgs = f.create_dataset("trax_GT", (n_imgs, n_xyz, n_xyz, n_xyz), dtype='f')
    dset_msks = f.create_dataset("trax_mask", (n_imgs, n_xyz, n_xyz, n_xyz), dtype='f')
    dset_obs = f.create_dataset("trax_obs", (n_imgs, n_xyz, n_xyz, n_xyz), dtype='f')

    # Lets fill up this file in chunks, making sure we don't fill our memory
    chunk = 100
    for ii in tqdm.tqdm(range(int(n_imgs // chunk))):
        b_imgs, b_msks, b_n_imgs = engine_b.generate_data_with_gaussian_noise(m_images=chunk,
                                                                              snr=snr,
                                                                              mask_radius=mask_radius)
        dset_imgs[ii * chunk:(ii + 1) * chunk, :, :, :] = b_imgs
        dset_msks[ii * chunk:(ii + 1) * chunk, :, :, :] = b_msks
        dset_obs[ii * chunk:(ii + 1) * chunk, :, :, :] = b_n_imgs

    # close the h5 file
    f.close()

    # build a test set
    n_imgs = np.floor(n_imgs * .1)
    if n_imgs < 100:
        n_imgs = 100
    # n_imgs = 1000

    f = h5py.File('test_data_3d.hdf5', 'w')
    dset_imgs = f.create_dataset("trax_GT", (n_imgs, n_xyz, n_xyz, n_xyz), dtype='f')
    dset_msks = f.create_dataset("trax_mask", (n_imgs, n_xyz, n_xyz, n_xyz), dtype='f')
    dset_obs = f.create_dataset("trax_obs", (n_imgs, n_xyz, n_xyz, n_xyz), dtype='f')

    # Lets fill up this file in chunks, making sure we don't fill our memory
    chunk = 100

    for ii in tqdm.tqdm(range(int(n_imgs // chunk))):
        b_imgs, b_msks, b_n_imgs = engine_b.generate_data_with_gaussian_noise(m_images=chunk,
                                                                              snr=snr,
                                                                              mask_radius=mask_radius)
        dset_imgs[ii * chunk:(ii + 1) * chunk, :, :, :] = b_imgs
        dset_msks[ii * chunk:(ii + 1) * chunk, :, :, :] = b_msks
        dset_obs[ii * chunk:(ii + 1) * chunk, :, :, :] = b_n_imgs

    # close the h5 file
    f.close()

    # build a validation set
    if n_imgs < 100:
        n_imgs = 100

    f = h5py.File('validate_data_3d.hdf5', 'w')
    dset_imgs = f.create_dataset("trax_GT", (n_imgs, n_xyz, n_xyz, n_xyz), dtype='f')
    dset_msks = f.create_dataset("trax_mask", (n_imgs, n_xyz, n_xyz, n_xyz), dtype='f')
    dset_obs = f.create_dataset("trax_obs", (n_imgs, n_xyz, n_xyz, n_xyz), dtype='f')

    # Lets fill up this file in chunks, making sure we don't fill our memory
    chunk = 100

    for ii in tqdm.tqdm(range(int(n_imgs // chunk))):
        b_imgs, b_msks, b_n_imgs = engine_b.generate_data_with_gaussian_noise(m_images=chunk,
                                                                              snr=snr,
                                                                              mask_radius=mask_radius)
        dset_imgs[ii * chunk:(ii + 1) * chunk, :, :, :] = b_imgs
        dset_msks[ii * chunk:(ii + 1) * chunk, :, :, :] = b_msks
        dset_obs[ii * chunk:(ii + 1) * chunk, :, :, :] = b_n_imgs

    # close the h5 file
    f.close()


def build_data_standard_sets_3d_sliced(n_imgs=100,
                                       n_peaks=1,
                                       n_xyz=32,
                                       snr=0.5):
    engine_b = noisy_gauss_3d.DataMaker3D(n_peaks=n_peaks,
                                          n_xyz=n_xyz)

    f = h5py.File('test_data_3d_sliced.hdf5', 'w')
    dset_imgs = f.create_dataset("trax_GT", (n_imgs * n_xyz, n_xyz, n_xyz), dtype='f')
    dset_msks = f.create_dataset("trax_mask", (n_imgs * n_xyz, n_xyz, n_xyz), dtype='f')
    dset_obs = f.create_dataset("trax_obs", (n_imgs * n_xyz, n_xyz, n_xyz), dtype='f')

    # Lets fill up this file in chunks, making sure we don't fill our memory
    chunk = 100
    for ii in tqdm.tqdm(range(n_imgs // chunk)):
        b_imgs, b_msks, b_n_imgs = engine_b.generate_data_with_gaussian_noise(m_images=chunk,
                                                                              snr=snr,
                                                                              mask_radius=2.0)
        newshape = b_imgs.shape
        dset_imgs[ii * chunk * n_xyz:(ii + 1) * chunk * n_xyz, :, :] = b_imgs.reshape(
            newshape[0] * newshape[1], newshape[2], newshape[3]
        )

        dset_msks[ii * chunk * n_xyz:(ii + 1) * chunk * n_xyz, :, :] = b_msks.reshape(
            newshape[0] * newshape[1], newshape[2], newshape[3]
        )

        dset_obs[ii * chunk * n_xyz:(ii + 1) * chunk * n_xyz, :, :] = b_n_imgs.reshape(
            newshape[0] * newshape[1], newshape[2], newshape[3]
        )

    # close the h5 file
    f.close()

    # build a training_set
    n_imgs = 10000

    f = h5py.File('train_data_3d_sliced.hdf5', 'w')
    dset_imgs = f.create_dataset("trax_GT", (n_imgs * n_xyz, n_xyz, n_xyz), dtype='f')
    dset_msks = f.create_dataset("trax_mask", (n_imgs * n_xyz, n_xyz, n_xyz), dtype='f')
    dset_obs = f.create_dataset("trax_obs", (n_imgs * n_xyz, n_xyz, n_xyz), dtype='f')

    # Lets fill up this file in chunks, making sure we don't fill our memory
    chunk = 100
    for ii in tqdm.tqdm(range(n_imgs // chunk)):
        b_imgs, b_msks, b_n_imgs = engine_b.generate_data_with_gaussian_noise(m_images=chunk,
                                                                              snr=snr,
                                                                              mask_radius=2.0)
        newshape = b_imgs.shape
        newshape = (newshape[0] * newshape[1], newshape[2], newshape[3])
        dset_imgs[ii * chunk * n_xyz:(ii + 1) * chunk * n_xyz, :, :] = b_imgs.reshape(
            newshape
        )

        dset_msks[ii * chunk * n_xyz:(ii + 1) * chunk * n_xyz, :, :] = b_msks.reshape(
            newshape
        )

        dset_obs[ii * chunk * n_xyz:(ii + 1) * chunk * n_xyz, :, :] = b_n_imgs.reshape(
            newshape
        )
    # close the h5 file
    f.close()


if __name__ == "__main__":
    build_data_standard_sets_3d(n_peaks=1,
                                n_xyz=32,
                                snr=0.5)
    build_data_standard_sets_3d_sliced(n_peaks=1,
                                       n_xyz=32,
                                       snr=1.0)
