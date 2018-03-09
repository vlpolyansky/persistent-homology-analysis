from utils import *
import idx2numpy
from scipy.ndimage import rotate
import cPickle


def normalize_data(data):
    data = np.array(data, dtype=np.float32)
    data /= np.max(data)
    data -= 0.5
    return data


def shuffle_data(images, labels):
    assert len(images) == len(labels)
    p = np.random.permutation(len(images))
    return images[p], labels[p]


def read_mnist_data(params, normalize=True):
    train_images = idx2numpy.convert_from_file(params['raw_data_folder'] + params['train_data'])
    train_labels = idx2numpy.convert_from_file(params['raw_data_folder'] + params['train_labels'])
    test_images = idx2numpy.convert_from_file(params['raw_data_folder'] + params['test_data'])
    test_labels = idx2numpy.convert_from_file(params['raw_data_folder'] + params['test_labels'])

    train_images = train_images[:, :, :, None]
    test_images = test_images[:, :, :, None]

    if 'filter_labels' in params and params['filter_labels'] is not None:
        train_images, train_labels = filter_labels(train_images, train_labels, params['filter_labels'])
        test_images, test_labels = filter_labels(test_images, test_labels, params['filter_labels'])

    if 'num_angles' in params and params['num_angles'] is not None and params['num_angles'] > 1:
        train_images, train_labels = rotate_images(train_images, train_labels, params['num_angles'])
        test_images, test_labels = rotate_images(test_images, test_labels, params['num_angles'])

    if 'blur_sigma' in params:
        sigma = params['blur_sigma']
        if sigma is not None and sigma > 0:
            train_images = blur_images(train_images, sigma)
            test_images = blur_images(test_images, sigma)

    if normalize:
        train_images = normalize_data(train_images)
        test_images = normalize_data(test_images)

    return train_images, train_labels, test_images, test_labels


def filter_labels(images, labels, to_filter):
    func = np.vectorize(lambda t: t in to_filter)
    indices = func(labels)
    return images[indices], labels[indices]


def rotate_images(images, labels, angles):
    mask = np.ones([1] + list(images.shape[1:]), dtype=images.dtype)
    square = mask.copy()
    for i in range(angles):
        angle = i * 360 / angles
        tmp = rotate(square, angle, axes=(2, 1), reshape=False)
        mask = np.minimum(mask, tmp)
    mask = np.where(mask < 0.5, 0, 1).astype(images.dtype)
    new_images = []
    new_labels = [] if labels is not None else None

    for i in range(angles):
        angle = i * 360.0 / angles
        tmp = rotate(images, angle, axes=(2, 1), reshape=False)
        tmp = tmp * mask
        new_images.append(tmp)
        if labels is not None:
            new_labels.append(np.copy(labels))

    return np.concatenate(new_images), np.concatenate(new_labels)


def blur_images(images, sigma):
    from skimage.filters import gaussian

    images_conv = []
    for image in images:
        tmp = gaussian(image, sigma=sigma, multichannel=True)
        images_conv.append(tmp)
    return np.array(images_conv)


def unpickle(filename):
    with open(filename, 'rb') as fo:
        dictionary = cPickle.load(fo)
    return dictionary


def read_cifar_data(params, normalize=True):
    train_images = []
    train_labels = []
    for i in range(1, 6):
        data = unpickle(os.path.join(params['raw_data_folder'], 'data_batch_%d' % i))
        train_images.append(data['data'])
        train_labels.append(data['labels'])
    train_images = np.concatenate(train_images)
    train_labels = np.concatenate(train_labels)
    data = unpickle(os.path.join(params['raw_data_folder'], 'test_batch'))
    test_images = data['data']
    test_labels = np.array(data['labels'])

    train_images = train_images.reshape((train_images.shape[0], 3, 32, 32)).transpose((0, 2, 3, 1))
    test_images = test_images.reshape((test_images.shape[0], 3, 32, 32)).transpose((0, 2, 3, 1))

    if 'filter_labels' in params and params['filter_labels'] is not None:
        train_images, train_labels = filter_labels(train_images, train_labels, params['filter_labels'])
        test_images, test_labels = filter_labels(test_images, test_labels, params['filter_labels'])

    if 'num_angles' in params and params['num_angles'] is not None and params['num_angles'] > 1:
        train_images, train_labels = rotate_images(train_images, train_labels, params['num_angles'])
        test_images, test_labels = rotate_images(test_images, test_labels, params['num_angles'])

    if normalize:
        train_images = normalize_data(train_images)
        test_images = normalize_data(test_images)

    return train_images, train_labels, test_images, test_labels


def read_unlabeled_data(params, normalize=True):
    frames = np.load(os.path.join(params['raw_data_folder'], 'frames.npy'))

    if 'filter_labels' in params and params['filter_labels'] is not None:
        pass  # Ignore

    if 'num_angles' in params and params['num_angles'] is not None and params['num_angles'] > 1:
        frames, _ = rotate_images(frames, None, params['num_angles'])

    labels = np.zeros((frames.shape[0]), dtype=np.int32)

    if normalize:
        frames = normalize_data(frames)

    return frames, labels, None, None


def read_vector_data(params, normalize=True):
    train_data = np.load(os.path.join(params['raw_data_folder'], 'train_data.npy'))
    train_labels = np.load(os.path.join(params['raw_data_folder'], 'train_labels.npy'))
    test_data = np.load(os.path.join(params['raw_data_folder'], 'test_data.npy'))
    test_labels = np.load(os.path.join(params['raw_data_folder'], 'test_labels.npy'))

    return train_data, train_labels, test_data, test_labels
