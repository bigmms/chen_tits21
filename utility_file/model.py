from __future__ import print_function, division
from keras_unet.models.custom_unet import conv2d_block, upsample_conv, upsample_simple
from utility_file.CALL import *
from keras.layers import Lambda, Conv2DTranspose, Add
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from utility_file.common import *
from segmentation_models.utils import freeze_model
from segmentation_models.backbones import get_backbone


def build_backbone(backbone_name, input_shape, isfreeze):
    backbone = get_backbone(backbone_name,
                            input_shape=input_shape,
                            input_tensor=None,
                            weights='imagenet',
                            include_top=False)
    if (isfreeze):
        freeze_model(backbone)
    features = []
    features.append(backbone.get_layer(backbone.layers[2].name).output)
    for l in range(0, 16, 4):
        features.append(backbone.get_layer(backbone.layers[5 + l].name).output)
    return Model(inputs=backbone.input, outputs=features)

def lightspot_branch(
        down_layers,
        num_classes=3,
        use_batch_norm=True,
        upsample_mode='deconv',  # 'deconv' or 'simple'
        use_dropout_on_upsampling=False,
        dropout=0.0,
        dropout_change_per_layer=0.0,
        filters=16,
        output_activation='tanh'):  # 'sigmoid' or 'softmax'

    if upsample_mode == 'deconv':
        upsample = upsample_conv
    else:
        upsample = upsample_simple

    if not use_dropout_on_upsampling:
        dropout = 0.0
        dropout_change_per_layer = 0.0
    L_prime = []
    mask = []
    x = down_layers[4]
    for conv in reversed(down_layers[0:4]):
        filters //= 2  # decreasing number of filters with each layer
        dropout -= dropout_change_per_layer
        x = upsample(filters, (2, 2), strides=(2, 2), padding='same')(x)
        x = concatenate([x, conv])
        x = conv2dblock(inputs=x, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout)

        L_grave  = Conv2D(num_classes, (1, 1), activation=output_activation)(x)
        L_grave  = Lambda(denormalize_m11)(L_grave)
        L_grave  = Lambda(normalize_01)(L_grave)
        L_prime.append(L_grave)

        L_bar = Lambda(lambda inputs: tf.image.rgb_to_grayscale(inputs))(L_grave)
        L_bar = Lambda(MASK)(L_bar)
        mask.append(L_bar)

    # outputs = Conv2D(num_classes, (1, 1), activation=output_activation)(x)
    return [L_prime,mask]

def background_branch(
        down_layers,
        num_classes=3,
        use_batch_norm=True,
        upsample_mode='deconv',  # 'deconv' or 'simple'
        use_dropout_on_upsampling=False,
        dropout=0.0,
        dropout_change_per_layer=0.0,
        filters=16,
        dilation_rate=2,
        output_activation='tanh'):  # 'sigmoid' or 'softmax'

    if upsample_mode == 'deconv':
        upsample = upsample_conv
    else:
        upsample = upsample_simple

    if not use_dropout_on_upsampling:
        dropout = 0.
        dropout_change_per_layer = 0.0

    x = down_layers[4]
    B_prime = []
    for conv in reversed(down_layers[0:4]):
        filters //= 2  # decreasing number of filters with each layer
        dropout -= dropout_change_per_layer
        x = upsample(filters, (2, 2), strides=(2, 2), padding='same')(x)
        x = concatenate([x, conv])
        x = conv2dblock(inputs=x, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout)

        B_grave = Conv2D(num_classes, (1, 1), activation=output_activation)(x)
        B_grave = Lambda(denormalize_m11)(B_grave)
        B_grave = Lambda(normalize_01)(B_grave)
        B_prime.append(B_grave)

    # outputs = Conv2D(num_classes, (1, 1), activation=output_activation)(x)
    return B_prime


def cal_branch(
        inputs,
        I,
        L_prime,
        mask,
        B_prime,
        num_classes=3,
        use_batch_norm=True,
        use_dropout_on_upsampling=False,
        dropout=0.0,
        dropout_change_per_layer=0.0,
        filters=16,
        output_activation='tanh'):  # 'sigmoid' or 'softmax'

    if not use_dropout_on_upsampling:
        dropout = 0.
        dropout_change_per_layer = 0.0

    x = inputs

    for i in range(4):
        filters //= 2  # decreasing number of filters with each layer
        dropout -= dropout_change_per_layer


        BFormula = Lambda(lambda inputs: tf.add(tf.subtract(inputs[0], inputs[1]), inputs[2]))([I[3-i], L_prime[i], B_prime[i]])
        BFormula = Lambda(get_CAL)([BFormula, BFormula, mask[i]])

        x = Transpose2D_CAL(x, filters)
        x = concatenate([BFormula, x])
        x = conv2dblock(inputs=x, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout)


    B = Conv2D(num_classes, (1, 1), activation=output_activation)(x)
    B = Lambda(denormalize_m11)(B)

    L_prime_outputs = Lambda(denormalize_01)(L_prime[3])
    B_prime_outputs = Lambda(denormalize_01)(B_prime[3])

    # outputs = Conv2D(num_classes, (1, 1), activation=output_activation)(x)
    return [L_prime_outputs,B_prime_outputs, B]


def coarse_net(
        input_shape,
        num_classes=3,
        use_batch_norm=True,
        upsample_mode='deconv',  # 'deconv' or 'simple'
        use_dropout_on_upsampling=False,
        dropout=0.0,
        dropout_change_per_layer=0.0,
        filters=1024,
        output_activation='tanh'):  # 'sigmoid' or 'softmax'

    # Build U-Net model
    backbone = build_backbone(backbone_name='vgg16', input_shape=input_shape, isfreeze=False)
    inputs = Input(input_shape)
    I = Lambda(normalize_m11)(inputs)
    down_layers = backbone(I)

    I_input=[]

    h, w = inputs.shape[1], inputs.shape[2]
    for i in range(4):
        x = Lambda(size_normalize(h, w))(inputs)
        I_input.append(Lambda(normalize_01)(x))
        h //= 2
        w //= 2

    lightspot,mask = lightspot_branch(down_layers,
                                   num_classes=num_classes,
                                   use_batch_norm=use_batch_norm,
                                   upsample_mode=upsample_mode,  # 'deconv' or 'simple'
                                   use_dropout_on_upsampling=use_dropout_on_upsampling,
                                   dropout=dropout,
                                   dropout_change_per_layer=dropout_change_per_layer,
                                   filters=filters,
                                   output_activation=output_activation)

    background = background_branch(down_layers,
                                   num_classes=num_classes,
                                   use_batch_norm=use_batch_norm,
                                   upsample_mode=upsample_mode,  # 'deconv' or 'simple'
                                   use_dropout_on_upsampling=use_dropout_on_upsampling,
                                   dropout=dropout,
                                   dropout_change_per_layer=dropout_change_per_layer,
                                   filters=filters,
                                   output_activation=output_activation)

    L_prime,B_prime, B= cal_branch(down_layers[4],
                                   I_input,
                                   lightspot,
                                   mask,
                                   background,
                                   num_classes=num_classes,
                                   use_batch_norm=use_batch_norm,
                                   use_dropout_on_upsampling=use_dropout_on_upsampling,
                                   dropout=dropout,
                                   dropout_change_per_layer=dropout_change_per_layer,
                                   filters=filters,
                                   output_activation=output_activation)

    model = Model(inputs=[inputs], outputs=[L_prime,B_prime, B])
    return model

def build_coarse(img_shape, output_activation='tanh'):
    """coarse-Net Generator"""
    net = coarse_net(input_shape=img_shape,
                     num_classes=3,
                     use_batch_norm=True,
                     upsample_mode='deconv',  # 'deconv' or 'simple'
                     use_dropout_on_upsampling=False,
                     dropout=0.0,
                     dropout_change_per_layer=0.0,
                     filters=1024,
                     output_activation=output_activation)

    return Model(inputs=net.input, outputs=net.output)

def build_refine(
        img_shape,
        output_activation='tanh',
        num_classes=3,
        use_batch_norm=True,
        upsample_mode='deconv',  # 'deconv' or 'simple'
        use_dropout_on_upsampling=False,
        dropout=0.0,
        dropout_change_per_layer=0.0,
        filters=1024):  # 'sigmoid' or 'softmax'
    """refine-Net Generator"""
    backbone = build_backbone(backbone_name='vgg16', input_shape=img_shape, isfreeze=False)
    inputs = Input(img_shape)
    I = Lambda(normalize_m11)(inputs)
    down_layers = backbone(I)
    net = refine_net(down_layers,
                     num_classes=num_classes,
                     use_batch_norm=use_batch_norm,
                     upsample_mode=upsample_mode,  # 'deconv' or 'simple'
                     use_dropout_on_upsampling=use_dropout_on_upsampling,
                     dropout=dropout,
                     dropout_change_per_layer=dropout_change_per_layer,
                     filters=filters,
                     output_activation=output_activation)
    model = Model(inputs=[inputs], outputs=[net])
    return model

def refine_net(
        down_layers,
        num_classes=3,
        use_batch_norm=True,
        upsample_mode='deconv',  # 'deconv' or 'simple'
        use_dropout_on_upsampling=False,
        dropout=0.0,
        dropout_change_per_layer=0.0,
        filters=16,
        dilation_rate = 2,
        output_activation='tanh'):  # 'sigmoid' or 'softmax'

    if upsample_mode == 'deconv':
        upsample = upsample_conv
    else:
        upsample = upsample_simple

    if not use_dropout_on_upsampling:
        dropout = 0.0
        dropout_change_per_layer = 0.0

    dilation_layers = []

    rate = dilation_rate
    for l in range(4):
        x = Conv2D(128, kernel_size=(3, 3), strides=1, padding='same', dilation_rate=rate)(down_layers[3])
        rate *= 2
        dilation_layers.append(x)

    rate = dilation_rate
    for l in range(4):
        x = Conv2D(64, kernel_size=(3, 3), strides=1, padding='same', dilation_rate=rate)(down_layers[2])
        rate *= 2
        dilation_layers.append(x)

    x = down_layers[4]
    k = 0
    for conv in reversed(down_layers[0:4]):
        filters //= 2  # decreasing number of filters with each layer
        dropout -= dropout_change_per_layer
        x = upsample(filters, (2, 2), strides=(2, 2), padding='same')(x)
        if (k == 0):
            for i in range(0, 4):
                x = concatenate([x, dilation_layers[i]])
            k = k + 1
        elif (k == 1):
            for i in range(4, 8):
                x = concatenate([x, dilation_layers[i]])
            k = k + 1
        else:
            x = concatenate([x, conv])
        x = conv2dblock(inputs=x, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout)

    outputs = Conv2D(num_classes, (1, 1), activation=output_activation)(x)
    outputs=Lambda(denormalize_m11)(outputs)
    return outputs

def generator_model(input_shape):
    # Build the networks
    coarse_network = build_coarse(img_shape=input_shape, output_activation='tanh')
    refine_network = build_refine(img_shape=input_shape, output_activation='tanh')

    # Input images
    I = Input(shape=input_shape)
    L_prime, B_prime, B = coarse_network(I)
    Residual = refine_network([B])


    BFormula = Lambda(lambda inputs: tf.add(tf.subtract(inputs[0], inputs[1]), inputs[2]))([I, L_prime, B_prime])
    Clean = Add()([Residual, BFormula])

    model = Model(inputs=[I], outputs=[L_prime, B_prime, B, Clean])
    return model


def MASK(L):
    y = []
    mask_shape = L.get_shape().as_list()
    group = tf.split(L, mask_shape[0] if mask_shape[0] != None else 1, axis=0)
    for img in group:
        mean = tf.reduce_mean(img)
        yi = tf.where(img > mean, tf.ones(tf.shape(img)), tf.zeros(tf.shape(img)))
        y.append(yi)
    y = tf.concat(y, axis=0)
    return y


def conv2dblock(
        inputs,
        use_batch_norm=True,
        dropout=0.,
        filters=16,
        kernel_size=(3, 3),
        kernel_initializer='he_normal',
        padding='same'):
    c = Conv2D(filters, kernel_size, activation=None, kernel_initializer=kernel_initializer, padding=padding)(
        inputs)
    c = LeakyReLU(alpha=0.2)(c)
    if use_batch_norm:
        c = BatchNormalization()(c)
    if dropout > 0.0:
        c = Dropout(dropout)(c)
    c = Conv2D(filters, kernel_size, activation=None, kernel_initializer=kernel_initializer, padding=padding)(c)
    c = LeakyReLU(alpha=0.2)(c)
    if use_batch_norm:
        c = BatchNormalization()(c)
    return c

def Transpose2D_CAL(layer_input, filters):
    """Layers used during upsampling"""
    u = UpSampling2D(size=2)(layer_input)
    u = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='same')(u)
    u = BatchNormalization(momentum=0.8)(u)
    u = LeakyReLU(alpha=0.2)(u)
    return u


def get_CAL(ts):
    return contextual_attention(ts[0], ts[1], ts[2], 3, 1, rate=2)

def d_layer(layer_input, filters, f_size=5, bn=True):
    """Discriminator layer"""
    d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
    if bn:
        d = BatchNormalization(momentum=0.8)(d)
    d = LeakyReLU(alpha=0.2)(d)
    return d


def discriminator_model(input_shape, filters=64):
    I = Input(input_shape)
    L_grave = Input(input_shape)
    B_grave = Input(input_shape)
    B = Input(input_shape)
    I1 = Lambda(normalize_m11)(I)
    L_grave1 = Lambda(normalize_m11)(L_grave)
    B_grave1 = Lambda(normalize_m11)(B_grave)
    B1 = Lambda(normalize_m11)(B)
    combined_imgs = Concatenate(axis=-1)([I1, L_grave1, B_grave1, B1])
    # Concatenate image and conditioning image by channels to produce input

    d1 = d_layer(combined_imgs, filters, bn=False)
    d2 = d_layer(d1, filters * 2)
    d3 = d_layer(d2, filters * 4)
    d4 = d_layer(d3, filters * 8)

    validity = Conv2D(1, kernel_size=5, strides=1, padding='same')(d4)

    model = Model(inputs=[I, L_grave, B_grave, B], outputs=validity)
    return model