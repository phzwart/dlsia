import einops
import h5py
import numpy as np
import skimage.transform
import tqdm


class Shape(object):
    """
    Base class for shapes.
    """

    def __init__(self, n_xy=64):
        """
        Base class for building shapes.

        :param n_xy: the size of the canvas we paint on
        """
        self.n_xy = n_xy
        self.mean = np.array([n_xy / 2.0, n_xy / 2.0])
        self.canvas = np.zeros((self.n_xy, self.n_xy))
        x = np.arange(0, n_xy)
        self.x, self.y = np.meshgrid(x, x)

    def get_random_rotation(self):
        """
        Rotates the canvas in a random fashion

        :return: return image in random rotation
        """
        phi = np.random.uniform(0, 360, 1)[0]
        tmp = skimage.transform.rotate(self.canvas, phi)
        return tmp


class Circle(Shape):
    def __init__(self, radius=0.25, n_xy=64):
        """
        Build a circle with radius

        :param radius: the radius
        :param n_xy: canvas size
        """
        super().__init__(n_xy=n_xy)
        self.radius = radius * n_xy

        r = (self.x - self.mean[0]) ** 2.0 + (self.y - self.mean[1]) ** 2.0
        r = np.sqrt(r)
        sel = r < self.radius
        self.canvas[sel] = 1.0


class Rectangle(Shape):
    def __init__(self, height=0.20, width=0.05, n_xy=64):
        """
        Build a rectangle

        :param height: height
        :param width: width
        :param n_xy: canvas size
        """
        super().__init__(n_xy=n_xy)
        self.height = height * n_xy
        self.width = width * n_xy

        dx = np.abs(self.x - self.mean[0])
        dy = np.abs(self.y - self.mean[1])

        sel = (dx < self.height) & (dy < self.width)
        self.canvas[sel] = 1.0


class Triangle(Rectangle):
    def __init__(self, height=0.25, n_xy=64):
        """
        Build an equilateral right triangle

        :param height: the height
        :param n_xy: canvas size
        """
        super().__init__(height=height, width=height, n_xy=n_xy)
        self.canvas = np.triu(self.canvas)


class Donut(Shape):
    def __init__(self, radius=.5, width=.25, n_xy=64):
        """
        Build a donut

        :param radius: radius of circle
        :param width: ring width
        :param n_xy: canvas size
        """
        super().__init__(n_xy=n_xy)
        self.radius = radius
        self.width = width * radius
        outer = Circle(radius, n_xy)
        inner = Circle(radius - width, n_xy)
        self.canvas = outer.canvas - inner.canvas


def random_rectangle(width=(0.10, 0.20), height=(0.30, 0.40), n_xy=64):
    """
    Build a rectangle of random size

    :param width: width range
    :param height: height range
    :param n_xy: canvas size
    :return: a random rectangle
    """
    this_width = np.random.uniform() * (width[1] - width[0]) + width[0]
    this_height = np.random.uniform() * (height[1] - height[0]) + height[0]
    obj = Rectangle(this_width, this_height, n_xy)
    return obj.get_random_rotation()


def random_triangle(width=(0.10, 0.35), n_xy=64):
    """
    Build a random right equilateral triangle

    :param width: width range
    :param n_xy: canvas size
    :return: a random triangle
    """
    this_width = np.random.uniform() * (width[1] - width[0]) + width[0]
    obj = Triangle(this_width, n_xy)
    return obj.get_random_rotation()


def random_circle(radius=(0.10, 0.40), n_xy=64):
    """
    Build a random circle

    :param radius: radius range
    :param n_xy: canvas size
    :return: a random circle
    """
    this_radius = np.random.uniform() * (radius[1] - radius[0]) + radius[0]
    obj = Circle(this_radius, n_xy)
    return obj.get_random_rotation()


def random_donut(radius=None, width=None, n_xy=64):
    """
    Build a random donut

    :param radius: radius range
    :param width: fractional width range, relatrive to radius
    :param n_xy: canvas size
    :return: a random donut
    """
    if radius is None:
        radius = [0.10, 0.40]
    if width is None:
        width = [0.20, 0.70]
    this_radius = np.random.uniform() * (radius[1] - radius[0]) + radius[0]
    this_width = np.random.uniform() * (width[1] - width[0]) + width[0]
    this_width = this_radius * this_width
    obj = Donut(this_radius, this_width, n_xy)
    return obj.get_random_rotation()


def get_random_object(noise_level=0.1, n_xy=64):
    """
    Build a random shape with fixed uniform noise level

    :param noise_level: uniform noise level
    :param n_xy: canvas size
    :return: a random shape
    """
    names = ["rectangle", "circle", "triangle", "donut"]
    obj_gen = [random_rectangle, random_circle, random_triangle, random_donut]
    ind = np.random.randint(0, 4)
    img = obj_gen[ind](n_xy=n_xy)
    noise = np.random.uniform(0, noise_level, img.shape)

    class_img = img * 1.0
    sel = class_img > 0.5
    class_img[sel] = 1
    class_img[~sel] = 0
    class_img = class_img * (ind + 1)
    norma = img + noise - np.min(img + noise)
    div = np.max(norma)
    if np.max(norma) < 1e-12:
        div = 1.0
    norma = norma / div
    return img, img + noise, class_img, norma, names[ind]


def build_random_shape_set_numpy(n_imgs, noise_level=0.1, n_xy=64):
    """
    Build a numpy array with random shapes.
    Returned is a dictionairy with results


    Parameters
    ----------
    n_imgs : number of images
    noise_level : noise level
    n_xy : canvas size

    Returns
    -------
    GroundTruth, Noisy, ClassImage,Label
    """
    gt_all = []
    obs_all = []
    class_all = []

    for _ in range(n_imgs):
        gt, obs, clss, _, _ = get_random_object(noise_level, n_xy)
        gt_all.append(gt)
        obs_all.append(obs)
        class_all.append(clss)
    gt_all = einops.rearrange(gt_all, "N Y X -> N Y X")
    obs_all = einops.rearrange(obs_all, "N Y X -> N Y X")
    class_all = einops.rearrange(class_all, "N Y X -> N Y X")
    label_all = np.max(class_all, axis=(-1, -2))

    # get rectangle
    rect_id = np.where(label_all == 1)
    if not np.any(rect_id[0]):
        raise TypeError('Class 1 is missing. Please generate a larger set')

    # get circle
    circ_id = np.where(label_all == 2)
    if not np.any(circ_id[0]):
        raise TypeError('Class 2 is missing. Please generate a larger set')

    # get triangle
    tri_id = np.where(label_all == 3)
    if not np.any(tri_id[0]):
        raise TypeError('Class 3 is missing. Please generate a larger set')

    # get annulus
    annu_id = np.where(label_all == 4)
    if not np.any(annu_id[0]):
        raise TypeError('Class 4 is missing. Please generate a larger set')

    return {"GroundTruth": gt_all, "Noisy": obs_all, "ClassImage": class_all, "Label": label_all}


def build_random_shape_set(n_train, n_test, n_validate, noise_level=0.1, n_xy=64):
    """
        Build 3 h5 files containing test data with random shapes.
        Uses standard filenames: train_shapes_2d.hdf5, etc etc

        :param n_train: number of training images
        :param n_test: number of test images
        :param n_validate: number of validation images
        :param noise_level: noise level
        :param n_xy: canvas size
        :return:
        """
    f = h5py.File('train_shapes_2d.hdf5', 'w')
    dset_imgs = f.create_dataset("shape_GT", (n_train, n_xy, n_xy), dtype='f')
    dset_obsr = f.create_dataset("shape_obs", (n_train, n_xy, n_xy), dtype='f')
    dset_clss = f.create_dataset("shape_class", (n_train, n_xy, n_xy), dtype='i')
    dset_norm = f.create_dataset("shape_norma", (n_train, n_xy, n_xy), dtype='f')
    for ii in tqdm.tqdm(range(n_train)):
        gt, obs, clss, norma, _ = get_random_object(noise_level, n_xy)
        dset_imgs[ii, :, :] = gt
        dset_obsr[ii, :, :] = obs
        dset_clss[ii, :, :] = clss
        dset_norm[ii, :, :] = norma
    f.close()

    f = h5py.File('test_shapes_2d.hdf5', 'w')
    dset_imgs = f.create_dataset("shape_GT", (n_test, n_xy, n_xy), dtype='f')
    dset_obsr = f.create_dataset("shape_obs", (n_test, n_xy, n_xy), dtype='f')
    dset_clss = f.create_dataset("shape_class", (n_test, n_xy, n_xy), dtype='i')
    dset_norm = f.create_dataset("shape_norma", (n_test, n_xy, n_xy), dtype='f')
    for ii in tqdm.tqdm(range(n_test)):
        gt, obs, clss, norma, _ = get_random_object(noise_level, n_xy)
        dset_imgs[ii, :, :] = gt
        dset_obsr[ii, :, :] = obs
        dset_clss[ii, :, :] = clss
        dset_norm[ii, :, :] = norma
    f.close()

    f = h5py.File('validate_shapes_2d.hdf5', 'w')
    dset_imgs = f.create_dataset("shape_GT", (n_validate, n_xy, n_xy), dtype='f')
    dset_obsr = f.create_dataset("shape_obs", (n_validate, n_xy, n_xy), dtype='f')
    dset_clss = f.create_dataset("shape_class", (n_validate, n_xy, n_xy), dtype='i')
    dset_norm = f.create_dataset("shape_norma", (n_validate, n_xy, n_xy), dtype='f')
    for ii in tqdm.tqdm(range(n_validate)):
        gt, obs, clss, norma, _ = get_random_object(noise_level, n_xy)
        dset_imgs[ii, :, :] = gt
        dset_obsr[ii, :, :] = obs
        dset_clss[ii, :, :] = clss
        dset_norm[ii, :, :] = norma
    f.close()
