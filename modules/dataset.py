import tensorflow as tf
import tensorflow_datasets as tfds


_CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
_CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]


def _meshgrid_tf(x, y):
    """ workaround solution of the tf.meshgrid() issue:
        https://github.com/tensorflow/tensorflow/issues/34470"""
    grid_shape = [tf.shape(y)[0], tf.shape(x)[0]]
    grid_x = tf.broadcast_to(tf.reshape(x, [1, -1]), grid_shape)
    grid_y = tf.broadcast_to(tf.reshape(y, [-1, 1]), grid_shape)
    return grid_x, grid_y


def _cutout(img, length, pad_values):
    """coutout"""
    h, w = tf.shape(img)[0], tf.shape(img)[1]
    y = tf.random.uniform([], 0, h, dtype=tf.int32)
    x = tf.random.uniform([], 0, w, dtype=tf.int32)

    y1 = tf.clip_by_value(y - length // 2, 0, h)
    y2 = tf.clip_by_value(y + length // 2, 0, h)
    x1 = tf.clip_by_value(x - length // 2, 0, w)
    x2 = tf.clip_by_value(x + length // 2, 0, w)

    grid_x, grid_y = _meshgrid_tf(tf.range(h), tf.range(w))
    cond = tf.stack([grid_x > y1, grid_x < y2, grid_y > x1, grid_y < x2], -1)
    mask = 1 - tf.cast(tf.math.reduce_all(cond, axis=-1, keepdims=True),
                       tf.float32)
    img = mask * img + (1 - mask) * pad_values

    return img


def _transform_data_cifar10(using_normalize, using_crop, using_flip,
                            using_cutout, cutout_length):
    def transform_data(features):
        img = features['image']
        labels = features['label']
        img = tf.cast(img, tf.float32)

        pad_values = tf.reduce_mean(img, axis=[0, 1])

        # randomly crop
        if using_crop:
            img = tf.pad(img, [[4, 4], [4, 4], [0, 0]], constant_values=-1)
            img = tf.where(img == -1, pad_values, img)
            img = tf.image.random_crop(img, [32, 32, 3])

        # randomly left-right flip
        if using_flip:
            img = tf.image.random_flip_left_right(img)

        # cutout
        if using_cutout:
            img = _cutout(img, cutout_length, pad_values)

        # rescale 0. ~ 1.
        img = img / 255.

        # normalize
        if using_normalize:
            mean = tf.constant(_CIFAR_MEAN)[tf.newaxis, tf.newaxis]
            std = tf.constant(_CIFAR_STD)[tf.newaxis, tf.newaxis]
            img = (img - mean) / std

        return img, labels
    return transform_data


def load_cifar10_dataset(batch_size, split='train', using_normalize=True,
                         using_crop=True, using_flip=True, using_cutout=True,
                         cutout_length=16, shuffle=True, buffer_size=10240,
                         drop_remainder=True):
    """load dataset from tfrecord"""
    dataset = tfds.load(name="cifar10", split=split)

    if 'train' in split:
        dataset = dataset.repeat()

    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)

    dataset = dataset.map(
        _transform_data_cifar10(using_normalize, using_crop, using_flip,
                                using_cutout, cutout_length),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset
