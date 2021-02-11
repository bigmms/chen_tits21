import tensorflow as tf
from sewar.full_ref import *
from PIL import Image

DIV2K_RGB_MEAN = np.array([0.4488, 0.4371, 0.4040]) * 255

def resolve_single(model, lr):
    return resolve(model, tf.expand_dims(lr, axis=0))[0]

def resolve(model, lr_batch):
    lr_batch = tf.cast(lr_batch, tf.float32)
    sr_batch = model(lr_batch)
    sr_batch = tf.clip_by_value(sr_batch, 0, 255)
    sr_batch = tf.round(sr_batch)
    sr_batch = tf.cast(sr_batch, tf.uint8)
    return sr_batch
# ---------------------------------------
#  Normalization
# ---------------------------------------
def normalize(x, rgb_mean=DIV2K_RGB_MEAN):
    return (x - rgb_mean) / 127.5

def denormalize(x, rgb_mean=DIV2K_RGB_MEAN):
    return x * 127.5 + rgb_mean

def normalize_01(x):
    """Normalizes RGB images to [0, 1]."""
    return x / 255.0

def normalize_m11(x):
    """Normalizes RGB images to [-1, 1]."""
    return x / 127.5 - 1

def denormalize_m11(x):
    """Inverse of normalize_m11."""
    return (x + 1) * 127.5

def denormalize_01(x):
    """Normalizes RGB images to [0, 255]."""
    return  x * 255.0

# ---------------------------------------
#  See https://arxiv.org/abs/1609.05158
# ---------------------------------------

def size_normalize(h, w):
    return lambda x: tf.image.resize(x, [h, w], tf.image.ResizeMethod.BICUBIC)

def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)

def draw_picture(I, L_grave, B_grave, B, Clean, GT, epoch=1, batchsize=1):
    for i in range(batchsize):
        imgI = Image.fromarray(I[i].astype(np.uint8))
        imgL_grave = Image.fromarray(L_grave[i].astype(np.uint8))
        imgB_grave = Image.fromarray(B_grave[i].astype(np.uint8))
        imgB = Image.fromarray(B[i].astype(np.uint8))
        imgClean = Image.fromarray(Clean[i].astype(np.uint8))
        imgGT = Image.fromarray(GT[i].astype(np.uint8))

        imgI.save("./result/%04d/I.png" % epoch, format='png')
        imgL_grave.save("./result/%04d/Gm.png" % epoch, format='png')
        imgB_grave.save("./result/%04d/Jm.png" % epoch, format='png')
        imgClean.save("./result/%04d/J.png" % epoch, format='png')
        imgGT.save("./result/%04d/GT.png" % epoch, format='png')