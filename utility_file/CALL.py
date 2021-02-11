import tensorflow as tf
import numpy as np
np.random.seed(2018)


def resize(x, scale=2, to_shape=None, align_corners=True, dynamic=False,
           func=tf.image.resize_bilinear, name='resize'):
    if dynamic:
        xs = tf.cast(tf.shape(x), tf.float32)
        new_xs = [tf.cast(xs[1] * scale, tf.int32),
                  tf.cast(xs[2] * scale, tf.int32)]
    else:
        xs = x.get_shape().as_list()
        new_xs = [int(xs[1] * scale), int(xs[2] * scale)]
    with tf.variable_scope(name):
        if to_shape is None:
            x = func(x, new_xs, align_corners=align_corners)
        else:
            x = func(x, [to_shape[0], to_shape[1]],
                     align_corners=align_corners)
    return x


def contextual_attention(f, b, mask, ksize=3, stride=1, rate=1,
                         fuse_k=3, softmax_scale=10., training=True, fuse=True):
    """ Contextual attention layer implementation.

    Contextual attention is first introduced in publication:
        Generative Image Inpainting with Contextual Attention, Yu et al.

    Args:
        x: Input feature to match (foreground).
        t: Input feature for match (background).
        mask: Input mask for t, indicating patches not available.
        ksize: Kernel size for contextual attention.
        stride: Stride for extracting patches from t.
        rate: Dilation for matching.
        softmax_scale: Scaled softmax for attention.
        training: Indicating if current graph is training or inference.

    Returns:
        tf.Tensor: output

    """
    # get shapes
    raw_int_bs = b.get_shape().as_list()
    f = resize(f, to_shape=[int(raw_int_bs[1]), int(raw_int_bs[2])], func=tf.image.resize_nearest_neighbor)
    raw_fs = tf.shape(f)
    raw_int_fs = f.get_shape().as_list()
    mask = resize(mask, to_shape=[int(raw_int_bs[1]), int(raw_int_bs[2])],func=tf.image.resize_nearest_neighbor)
    # extract patches from background with stride and rate
    kernel = 2 * rate
    raw_w = tf.extract_image_patches(
        b, [1, kernel, kernel, 1], [1, rate * stride, rate * stride, 1], [1, 1, 1, 1], padding='SAME')

    l = raw_w.shape[1] * raw_w.shape[2] * raw_w.shape[3] // (raw_int_bs[3] * kernel * kernel)
    if raw_int_bs[0] == None:
        raw_w = tf.reshape(raw_w, [-1, l, kernel, kernel, raw_int_bs[3]])
    else:
        raw_w = tf.reshape(raw_w, [raw_int_bs[0], -1, kernel, kernel, raw_int_bs[3]])
    # raw_w = tf.reshape(raw_w, [raw_int_bs[0], -1, kernel, kernel, raw_int_bs[3]])
    raw_w = tf.transpose(raw_w, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    # downscaling foreground option: downscaling both foreground and
    # background for matching and use original background for reconstruction.
    f = resize(f, scale=1. / rate, func=tf.image.resize_nearest_neighbor)
    b = resize(b, to_shape=[int(raw_int_bs[1] / rate), int(raw_int_bs[2] / rate)],
               func=tf.image.resize_nearest_neighbor)  # https://github.com/tensorflow/tensorflow/issues/11651
    if mask is not None:
        mask = resize(mask, scale=1. / rate, func=tf.image.resize_nearest_neighbor)
    fs = tf.shape(f)
    int_fs = f.get_shape().as_list()
    f_groups = tf.split(f, int_fs[0] if int_fs[0] != None else 1, axis=0)
    # from t(H*W*C) to w(b*k*k*c*h*w)
    bs = tf.shape(b)
    int_bs = b.get_shape().as_list()
    w = tf.extract_image_patches(
        b, [1, ksize, ksize, 1], [1, stride, stride, 1], [1, 1, 1, 1], padding='SAME')
    k = w.shape[1] * w.shape[2] * w.shape[3] // (int_fs[3] * ksize * ksize)
    if int_fs[0] == None:
        w = tf.reshape(w, [-1, k, ksize, ksize, int_fs[3]])
    else:
        w = tf.reshape(w, [int_fs[0], -1, ksize, ksize, int_fs[3]])
    # w = tf.reshape(w, [int_fs[0], -1, ksize, ksize, int_fs[3]])
    w = tf.transpose(w, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    # process mask
    if mask is None:
        mask = tf.zeros([1, bs[1], bs[2], 1])
    m = tf.extract_image_patches(
        mask, [1, ksize, ksize, 1], [1, stride, stride, 1], [1, 1, 1, 1], padding='SAME')
    s = m.shape[1] * m.shape[2] * m.shape[3] // (1 * ksize * ksize)
    if raw_int_bs[0] == None:
        m = tf.reshape(m, [-1, s, ksize, ksize, 1])
    else:
        m = tf.reshape(m, [1, -1, ksize, ksize, 1])
    # m = tf.reshape(m, [1, -1, ksize, ksize, 1])
    m = tf.transpose(m, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    m = tf.split(m, int_bs[0] if int_bs[0] != None else 1, axis=0)
    # # m = m[0]
    # mm = tf.cast(tf.equal(tf.reduce_mean(m, axis=[0, 1, 2], keep_dims=True), 0.), tf.float32)
    w_groups = tf.split(w, int_bs[0]if int_bs[0] != None else 1, axis=0)
    raw_w_groups = tf.split(raw_w, int_bs[0]if int_bs[0] != None else 1, axis=0)
    y = []
    k = fuse_k
    scale = softmax_scale
    fuse_weight = tf.reshape(tf.eye(k), [k, k, 1, 1])
    for xi, wi, raw_wi,mi in zip(f_groups, w_groups, raw_w_groups,m):
        #mask处理
        mi_ = mi[0]
        mm = tf.cast(tf.equal(tf.reduce_mean(mi_, axis=[0, 1, 2], keep_dims=True), 0.), tf.float32)
        # conv for compare
        wi = wi[0]
        wi_normed = wi / tf.maximum(tf.sqrt(tf.reduce_sum(tf.square(wi), axis=[0, 1, 2])), 1e-4)
        yi = tf.nn.conv2d(xi, wi_normed, strides=[1, 1, 1, 1], padding="SAME")

        # conv implementation for fuse scores to encourage large patches
        if fuse:
            yi = tf.reshape(yi, [-1, fs[1] * fs[2], bs[1] * bs[2], 1])
            yi = tf.nn.conv2d(yi, fuse_weight, strides=[1, 1, 1, 1], padding='SAME')
            yi = tf.reshape(yi, [-1, fs[1], fs[2], bs[1], bs[2]])
            yi = tf.transpose(yi, [0, 2, 1, 4, 3])
            yi = tf.reshape(yi, [-1, fs[1] * fs[2], bs[1] * bs[2], 1])
            yi = tf.nn.conv2d(yi, fuse_weight, strides=[1, 1, 1, 1], padding='SAME')
            yi = tf.reshape(yi, [-1, fs[2], fs[1], bs[2], bs[1]])
            yi = tf.transpose(yi, [0, 2, 1, 4, 3])
        yi = tf.reshape(yi, [-1, fs[1], fs[2], bs[1] * bs[2]])

        # softmax to match
        yi *= mm  # mask
        yi = tf.nn.softmax(yi * scale, 3)
        yi *= mm  # mask

        # deconv for patch pasting
        # 3.1 paste center
        wi_center = raw_wi[0]
        if raw_int_fs[0] == None:
            ts = raw_fs[0:]
            yi = tf.nn.conv2d_transpose(yi, wi_center, ts,
                                        strides=[1, rate, rate, 1]) / 4.
        else:
            yi = tf.nn.conv2d_transpose(yi, wi_center, tf.concat([[1], raw_fs[1:]], axis=0),
                                        strides=[1, rate, rate, 1]) / 4.
        y.append(yi)
    y = tf.concat(y, axis=0)
    if raw_int_fs[0] != None:
        y.set_shape(raw_int_fs)

    return y