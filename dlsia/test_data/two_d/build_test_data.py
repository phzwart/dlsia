import h5py
import numpy as np
import os
import tqdm
from dlsia.test_data.two_d import noisy_gauss_2d
from dlsia.test_data.two_d import noisy_gauss_2d_time


def build_data_standard_sets_2d(n_imgs=1000,
                                n_peaks=3,
                                n_xy=64,
                                snr=1.0,
                                mask_radius=2.0, chunk=10):
    engine_b = noisy_gauss_2d.DataMaker(n_peaks=n_peaks,
                                        n_xy=n_xy)

    # Build Training Set
    ####################

    f = h5py.File('train_data_2d.hdf5', 'w')

    dset_imgs = f.create_dataset("trax_GT", (n_imgs, n_xy, n_xy), dtype='f')
    dset_msks = f.create_dataset("trax_mask", (n_imgs, n_xy, n_xy), dtype='f')
    dset_clss = f.create_dataset("trax_classes", (n_imgs, 2, n_xy, n_xy), dtype='f')
    dset_obs = f.create_dataset("trax_obs", (n_imgs, n_xy, n_xy), dtype='f')
    dset_obs_norm = f.create_dataset("trax_obs_norm", (n_imgs, n_xy, n_xy), dtype='f')

    # Lets fill up this file in chunks, making sure we don't fill our memory

    for ii in tqdm.tqdm(range(int(n_imgs // chunk))):
        b_imgs, b_msks, b_n_imgs, b_n_imgs_norm, b_c_class = engine_b.generate_data_with_normal_noise(
            n_images=chunk, snr=snr, mask_radius=mask_radius)

        dset_imgs[ii * chunk:(ii + 1) * chunk, :, :] = b_imgs
        dset_msks[ii * chunk:(ii + 1) * chunk, :, :] = b_msks
        dset_obs[ii * chunk:(ii + 1) * chunk, :, :] = b_n_imgs
        dset_obs_norm[ii * chunk:(ii + 1) * chunk, :, :] = b_n_imgs_norm
        dset_clss[ii * chunk:(ii + 1) * chunk, :, :, :] = b_c_class

    # close the h5 file
    f.close()

    # Build Validation Set
    ######################
    n_imgs = max(100, np.floor(n_imgs * .2))
    # n_xy = 64

    f = h5py.File('validate_data_2d.hdf5', 'w')

    dset_imgs = f.create_dataset("trax_GT", (n_imgs, n_xy, n_xy), dtype='f')
    dset_msks = f.create_dataset("trax_mask", (n_imgs, n_xy, n_xy), dtype='f')
    dset_clss = f.create_dataset("trax_classes", (n_imgs, 2, n_xy, n_xy), dtype='f')
    dset_obs = f.create_dataset("trax_obs", (n_imgs, n_xy, n_xy), dtype='f')
    dset_obs_norm = f.create_dataset("trax_obs_norm", (n_imgs, n_xy, n_xy), dtype='f')

    # Lets fill up this file in chunks, making sure we don't fill our memory

    for ii in tqdm.tqdm(range(int(n_imgs // chunk))):
        b_imgs, b_msks, b_n_imgs, b_n_imgs_norm, b_c_class = engine_b.generate_data_with_normal_noise(
            n_images=chunk, snr=snr, mask_radius=mask_radius)
        dset_imgs[ii * chunk:(ii + 1) * chunk, :, :] = b_imgs
        dset_msks[ii * chunk:(ii + 1) * chunk, :, :] = b_msks
        dset_obs[ii * chunk:(ii + 1) * chunk, :, :] = b_n_imgs
        dset_obs_norm[ii * chunk:(ii + 1) * chunk, :, :] = b_n_imgs_norm
        dset_clss[ii * chunk:(ii + 1) * chunk, :, :, :] = b_c_class

    # close the h5 file
    f.close()

    # Build Testing Set
    ###################

    # n_xy = 64

    f = h5py.File('test_data_2d.hdf5', 'w')

    dset_imgs = f.create_dataset("trax_GT", (n_imgs, n_xy, n_xy), dtype='f')
    dset_msks = f.create_dataset("trax_mask", (n_imgs, n_xy, n_xy), dtype='f')
    dset_clss = f.create_dataset("trax_classes", (n_imgs, 2, n_xy, n_xy), dtype='f')
    dset_obs = f.create_dataset("trax_obs", (n_imgs, n_xy, n_xy), dtype='f')
    dset_obs_norm = f.create_dataset("trax_obs_norm", (n_imgs, n_xy, n_xy), dtype='f')

    # Lets fill up this file in chunks, making sure we don't fill our memory
    chunk = 100
    for ii in tqdm.tqdm(range(int(n_imgs // chunk))):
        b_imgs, b_msks, b_n_imgs, b_n_imgs_norm, b_c_class = engine_b.generate_data_with_normal_noise(
            n_images=chunk, snr=snr, mask_radius=mask_radius)

        dset_imgs[ii * chunk:(ii + 1) * chunk, :, :] = b_imgs
        dset_msks[ii * chunk:(ii + 1) * chunk, :, :] = b_msks
        dset_obs[ii * chunk:(ii + 1) * chunk, :, :] = b_n_imgs
        dset_obs_norm[ii * chunk:(ii + 1) * chunk, :, :] = b_n_imgs_norm
        dset_clss[ii * chunk:(ii + 1) * chunk, :, :, :] = b_c_class

    # close the h5 file
    f.close()


def build_data_mixed_level_sets_2d(n_imgs=1000,
                                   n_peaks=3,
                                   n_xy=32,
                                   mask_radius=1.0, chunk=10):
    engine_b = noisy_gauss_2d.MixedNoiseDataMaker(n_peaks=n_peaks,
                                                  n_xy=n_xy)

    noise_limits = np.array(engine_b.snr_brackets)
    print(noise_limits)

    # Build Training Set
    ####################

    f = h5py.File('train_data_2d.hdf5', 'w')

    dset_imgs = f.create_dataset("trax_GT", (n_imgs, n_xy, n_xy), dtype='f')
    dset_msks = f.create_dataset("trax_mask", (n_imgs, n_xy, n_xy), dtype='i')
    dset_obs = f.create_dataset("trax_obs", (n_imgs, n_xy, n_xy), dtype='f')
    dset_obs_norm = f.create_dataset("trax_obs_norm", (n_imgs, n_xy, n_xy), dtype='f')

    dset_levels = f.create_dataset("psnr_brackets", noise_limits.shape, dtype='f')
    dset_levels[:, :] = noise_limits[:, :]

    # Lets fill up this file in chunks

    for ii in tqdm.tqdm(range(int(n_imgs // chunk))):
        b_imgs, b_msks, b_n_imgs, b_n_imgs_norm = engine_b.generate_data_with_normal_noise(m_images=chunk,
                                                                                           mask_radius=mask_radius)

        dset_imgs[ii * chunk:(ii + 1) * chunk, :, :] = b_imgs
        dset_msks[ii * chunk:(ii + 1) * chunk, :, :] = b_msks
        dset_obs[ii * chunk:(ii + 1) * chunk, :, :] = b_n_imgs
        dset_obs_norm[ii * chunk:(ii + 1) * chunk, :, :] = b_n_imgs_norm

    # close the h5 file
    f.close()

    # Build Validation Set
    ######################
    n_imgs = max(100, np.floor(n_imgs * .2))

    f = h5py.File('validate_data_2d.hdf5', 'w')

    dset_imgs = f.create_dataset("trax_GT", (n_imgs, n_xy, n_xy), dtype='f')
    dset_msks = f.create_dataset("trax_mask", (n_imgs, n_xy, n_xy), dtype='f')
    dset_obs = f.create_dataset("trax_obs", (n_imgs, n_xy, n_xy), dtype='f')
    dset_obs_norm = f.create_dataset("trax_obs_norm", (n_imgs, n_xy, n_xy), dtype='f')

    # Not used
    # dset_levels = f.create_dataset("psnr_brackets", noise_limits.shape, dtype='f')
    # dset_levels = noise_limits

    # Lets fill up this file in chunks, making sure we don't fill our memory
    for ii in tqdm.tqdm(range(int(n_imgs // chunk))):
        b_imgs, b_msks, b_n_imgs, b_n_imgs_norm = engine_b.generate_data_with_normal_noise(m_images=chunk,
                                                                                           mask_radius=mask_radius)
        # print('min/max of noisy: ', np.min(b_n_imgs), np.max(b_n_imgs))
        dset_imgs[ii * chunk:(ii + 1) * chunk, :, :] = b_imgs
        dset_msks[ii * chunk:(ii + 1) * chunk, :, :] = b_msks
        dset_obs[ii * chunk:(ii + 1) * chunk, :, :] = b_n_imgs
        dset_obs_norm[ii * chunk:(ii + 1) * chunk, :, :] = b_n_imgs_norm

    # close the h5 file
    f.close()

    # Build Testing Set
    ####################
    f = h5py.File('test_data_2d.hdf5', 'w')

    dset_imgs = f.create_dataset("trax_GT", (n_imgs, n_xy, n_xy), dtype='f')
    dset_msks = f.create_dataset("trax_mask", (n_imgs, n_xy, n_xy), dtype='f')
    dset_obs = f.create_dataset("trax_obs", (n_imgs, n_xy, n_xy), dtype='f')
    dset_obs_norm = f.create_dataset("trax_obs_norm", (n_imgs, n_xy, n_xy), dtype='f')

    # Not seemed to be used
    # dset_levels = f.create_dataset("psnr_brackets", noise_limits.shape, dtype='f')
    # dset_levels = noise_limits

    # Lets fill up this file in chunks, making sure we don't fill our memory
    chunk = 100
    for ii in tqdm.tqdm(range(int(n_imgs // chunk))):
        b_imgs, b_msks, b_n_imgs, b_n_imgs_norm = engine_b.generate_data_with_normal_noise(m_images=chunk,
                                                                                           mask_radius=mask_radius)
        dset_imgs[ii * chunk:(ii + 1) * chunk, :, :] = b_imgs
        dset_msks[ii * chunk:(ii + 1) * chunk, :, :] = b_msks
        dset_obs[ii * chunk:(ii + 1) * chunk, :, :] = b_n_imgs
        dset_obs_norm[ii * chunk:(ii + 1) * chunk, :, :] = b_n_imgs_norm

    # close the h5 file
    f.close()


def build_data_standard_sets_2d_time(n_imgs=1000,
                                     k_time_points=8,
                                     n_peaks=3,
                                     sigma=0.02,
                                     trend=0.01,
                                     dxy=0.01,
                                     cc=0.6,
                                     n_xy=64,
                                     normalize=True):
    engine_b = noisy_gauss_2d_time.DataMaker(n_peaks,
                                             sigma=sigma,
                                             trend=trend,
                                             dxy=dxy,
                                             cc=cc,
                                             n_xy=n_xy)

    # Build Training Set
    ####################

    if normalize:
        f = h5py.File('train_data_2d_time_norm.hdf5', 'w')
    else:
        f = h5py.File('train_data_2d_time.hdf5', 'w')

    dset_imgs = f.create_dataset("trax_GT", (n_imgs, k_time_points, n_xy, n_xy), dtype='f')
    dset_msks = f.create_dataset("trax_mask", (n_imgs, k_time_points, n_xy, n_xy), dtype='f')
    dset_obs = f.create_dataset("trax_obs", (n_imgs, k_time_points, n_xy, n_xy), dtype='f')

    # Lets fill up this file in chunks, making sure we don't fill our memory
    chunk = 100
    for ii in tqdm.tqdm(range(n_imgs // chunk)):
        b_imgs, b_msks, b_n_imgs = engine_b.generate_data_with_uniform_noise(m_images=chunk,
                                                                             k_time_steps=k_time_points,
                                                                             noise_level=1,
                                                                             mask_radius=2.0)
        dset_imgs[ii * chunk:(ii + 1) * chunk, :, :, :] = b_imgs
        dset_msks[ii * chunk:(ii + 1) * chunk, :, :, :] = b_msks
        dset_obs[ii * chunk:(ii + 1) * chunk, :, :, :] = b_n_imgs

    # close the h5 file
    f.close()

    # Build Testing Set
    ###################
    n_imgs = int(np.floor(n_imgs * .1))

    if normalize:
        f = h5py.File('test_data_2d_time_norm.hdf5', 'w')
    else:
        f = h5py.File('test_data_2d_time.hdf5', 'w')

    dset_imgs = f.create_dataset("trax_GT", (n_imgs, k_time_points, n_xy, n_xy), dtype='f')
    dset_msks = f.create_dataset("trax_mask", (n_imgs, k_time_points, n_xy, n_xy), dtype='f')
    dset_obs = f.create_dataset("trax_obs", (n_imgs, k_time_points, n_xy, n_xy), dtype='f')

    # Lets fill up this file in chunks, making sure we don't fill our memory
    chunk = 100
    for ii in tqdm.tqdm(range(n_imgs // chunk)):
        b_imgs, b_msks, b_n_imgs = engine_b.generate_data_with_uniform_noise(m_images=chunk,
                                                                             k_time_steps=k_time_points,
                                                                             noise_level=1,
                                                                             mask_radius=2.0)
        dset_imgs[ii * chunk:(ii + 1) * chunk, :, :, :] = b_imgs
        dset_msks[ii * chunk:(ii + 1) * chunk, :, :, :] = b_msks
        dset_obs[ii * chunk:(ii + 1) * chunk, :, :, :] = b_n_imgs

    # close the h5 file
    f.close()


if __name__ == "__main__":
    are_you_sure = None
    while are_you_sure not in ["Y", "N"]:
        are_you_sure = input("Are you sure you want to build Test data Y/N: ")
    print("go", are_you_sure)
    if are_you_sure == "Y":
        if os.path.isfile("train_data_2d.hdf5"):
            os.remove("train_data_2d.hdf5")
        if os.path.isfile("test_data_2d.hdf5"):
            os.remove("test_data_2d.hdf5")
        build_data_standard_sets_2d()

        if os.path.isfile("train_data_2d_time.hdf5"):
            os.remove("train_data_2d_time.hdf5")
        if os.path.isfile("test_data_2d_time.hdf5"):
            os.remove("test_data_2d_time.hdf5")
        build_data_standard_sets_2d_time()

    else:
        print("MAKE UP YOUR MIND")
