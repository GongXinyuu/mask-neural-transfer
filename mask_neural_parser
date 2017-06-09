#!/usr/bin/python
# -*- coding:utf8 -*-
'''Neural style transfer with Keras.


It is preferable to run this script on GPU, for speed.

Example result: https://twitter.com/fchollet/status/686631033085677568

# Details

Style transfer consists in generating an image
with the same "content" as a base image, but with the
"style" of a different picture (typically artistic).

This is achieved through the optimization of a loss function
that has 3 components: "style loss", "content loss",
and "total variation loss":

- The total variation loss imposes local spatial continuity between
the pixels of the combination image, giving it visual coherence.

- The style loss is where the deep learning keeps in --that one is defined
using a deep convolutional neural network. Precisely, it consists in a sum of
L2 distances between the Gram matrices of the representations of
the base image and the style reference image, extracted from
different layers of a convnet (trained on ImageNet). The general idea
is to capture color/texture information at different spatial
scales (fairly large scales --defined by the depth of the layer considered).

 - The content loss is a L2 distance between the features of the base
image (extracted from a deep layer) and the features of the combination image,
keeping the generated image close enough to the original one.

# References
    - [A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576)
'''

from __future__ import print_function
from keras.preprocessing.image import load_img, img_to_array
from scipy.misc import imsave
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time
from keras.utils import plot_model
from keras.applications import vgg16
from keras import backend as K
from vgg16featuremap import *
import argparse

parser = argparse.ArgumentParser(description='Neural style transfer with Keras.')
parser.add_argument('base_image_path', metavar='base', type=str,
                    help='Path to the image to transform.')
parser.add_argument('mask_path', metavar='mask', type=str,
                    help='Path to the mask image.')
parser.add_argument('style_reference_background_image_path', metavar='ref_back', type=str,
                    help='Path to the background style reference image.')
parser.add_argument('style_reference_key_image_path', metavar='ref_key', type=str,
                    help='Path to the key element style reference image.')
parser.add_argument('result_prefix', metavar='res_prefix', type=str,
                    help='Prefix for the saved results.')
parser.add_argument('--iter', type=int, default=10, required=False,
                    help='Number of iterations to run.')
parser.add_argument('--content_weight', type=float, default=0, required=False,
                    help='Content weight.')
parser.add_argument('--style_weight', type=float, default=100.0, required=False,
                    help='Style weight.')
parser.add_argument('--tv_weight', type=float, default=1.0, required=False,
                    help='Total Variation weight.')

args = parser.parse_args()
base_image_path = args.base_image_path
mask_path = args.mask_path
style_reference_background_image_path = args.style_reference_background_image_path
style_reference_key_image_path = args.style_reference_key_image_path
result_prefix = args.result_prefix
iterations = args.iter

# these are the weights of the different loss components
total_variation_weight = 1#8.5e-5 # A larger value may cause blur
style_weight = args.style_weight
content_weight = args.content_weight
# dimensions of the generated picture.
width, height = load_img(base_image_path).size
img_nrows = height
img_ncols = width

# util function to open, resize and format pictures into appropriate tensors


def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_nrows, img_ncols))  # 创建实例
    img = img_to_array(img) # 将图片实例转化为张量
    img = np.expand_dims(img, axis=0)   # 在首部增加一个维度
    img = vgg16.preprocess_input(img)   # 零均值化，即减去训练vgg16的数据集每个通道的均值
    return img

# util function to convert a tensor into a valid image


def deprocess_image(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, img_nrows, img_ncols))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((img_nrows, img_ncols, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')  # 将数值限制在0~255
    return x

# get tensor representations of our images
base_image = K.variable(preprocess_image(base_image_path))  # 创建base预处理图片实例
style_reference_background_image = K.variable(preprocess_image(style_reference_background_image_path))    # 创建style预处理图片实例
style_reference_key_image = K.variable(preprocess_image(style_reference_key_image_path))    # 创建style预处理图片实例
mask_image = img_to_array(load_img(mask_path, target_size=(img_nrows, img_ncols)))  # mask矩阵化
mask_key_bool = (mask_image > 0) * 1.0
mask_background_bool = (mask_image == 0) * 1.0

# this will contain our generated image
if K.image_data_format() == 'channels_first':
    combination_image = K.placeholder((1, 3, img_nrows, img_ncols)) # 占位符
else:
    combination_image = K.placeholder((1, img_nrows, img_ncols, 3))


# combine the 3 images into a single Keras tensor
# 作为一个串联的整体输入，类似于一个batch
input_tensor = K.concatenate([combination_image * mask_key_bool,
                              combination_image * mask_background_bool], axis=0)   # 即input_tensor具有四维

# build the VGG16 network with our 3 images as input
# the model will be loaded with pre-trained ImageNet weights
model = vgg16.VGG16(input_tensor=input_tensor,
                    weights='imagenet', include_top=False)
print('Model loaded.')

# get the symbolic outputs of each "key" layer (we gave them unique names).
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

# compute the neural style loss
# first we need to define 4 util functions

# the gram matrix of an image tensor (feature-wise outer product)


def gram_matrix(x):
    # assert K.ndim(x) == 3
    if K.image_data_format() == 'channels_first':
        features = K.batch_flatten(x)   # 张成二维张量
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

# the "style loss" is designed to maintain
# the style of the reference image in the generated image.
# It is based on the gram matrices (which capture style) of
# feature maps from the style reference image
# and from the generated image


def style_loss(style, combination):
    # assert K.ndim(style) == 3
    # assert K.ndim(combination) == 3
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

# an auxiliary loss function
# designed to maintain the "content" of the
# base image in the generated image
def content_loss(base, combination):
    return K.sum(K.square(combination - base))

# the 3rd loss function, total variation loss,
# designed to keep the generated image locally coherent


def total_variation_loss(x):
    assert K.ndim(x) == 4
    if K.image_data_format() == 'channels_first':
        a = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, 1:, :img_ncols - 1]) #错位相减，保持平滑
        b = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, :img_nrows - 1, 1:]) #错位相减，保持平滑
    else:
        a = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
        b = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

# combine these loss functions into a single scalar

feature_layers = ['block1_conv1', 'block2_conv1',
                  'block3_conv1', 'block4_conv1',
                  'block5_conv1']

modelFM = VGG16FM(input_tensor=None,
                    weights='imagenet', include_top=False)
features_back = modelFM.predict(preprocess_image(style_reference_background_image_path))
features_key = modelFM.predict(preprocess_image(style_reference_key_image_path))
features_content = modelFM.predict(preprocess_image(base_image_path))
count = 1

loss = K.variable(0.)
layer_features = outputs_dict['block4_conv2']
base_image_features = features_content[4][0]
combination_features = layer_features[0, :, :, :] + layer_features[1, :, :, :]
loss += content_weight * content_loss(base_image_features,
                                      combination_features)

for layer_name in feature_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_key_features = features_key[count][0]
    style_reference_background_features = features_back[count][0]
    combination_key_features = layer_features[0, :, :, :]
    combination_background_features = layer_features[1, :, :, :]
    sl_key = style_loss(style_reference_key_features, combination_key_features)
    sl_background = style_loss(style_reference_background_features, combination_background_features)
    loss += (style_weight / len(feature_layers)) * (sl_key + sl_background)
    count += 1
loss += total_variation_weight * total_variation_loss(combination_image)

# get the gradients of the generated image wrt the loss
grads = K.gradients(loss, combination_image)

outputs = [loss]
if isinstance(grads, (list, tuple)):
    outputs += grads
else:
    outputs.append(grads)

f_outputs = K.function([combination_image], outputs)    # Output values as Numpy arrays


def eval_loss_and_grads(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((1, 3, img_nrows, img_ncols))
    else:
        x = x.reshape((1, img_nrows, img_ncols, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1]
        grad_values = grad_values.flatten().astype('float64')

    else:
        grad_values = np.array(outs[1:])
        grad_values = grad_values.flatten().astype('float64')

    return loss_value, grad_values

# this Evaluator class makes it possible
# to compute loss and gradients in one pass
# while retrieving them via two separate functions,
# "loss" and "grads". This is done because scipy.optimize
# requires separate functions for loss and gradients,
# but computing them separately would be inefficient.


class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)    # key import
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()

# run scipy-based optimization (L-BFGS) over the pixels of the generated image
# so as to minimize the neural style loss
# if K.image_data_format() == 'channels_first':
#     x = np.zeros_like([1, 3, img_nrows, img_ncols])
# else:
#     x = np.zeros_like([1, img_nrows, img_ncols, 3])
# x = 0 * preprocess_image(base_image_path)
x = preprocess_image(base_image_path)   # initial with base image
# plot_model(model, to_file='nerual_transfer_model.png', show_shapes = True)
for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                     fprime=evaluator.grads, maxfun=20)
    print('Current loss value:', min_val)
    # save current generated image
    img = deprocess_image(x.copy())
    fname = result_prefix + '_at_iteration_%d.png' % i
    if i % 5 == 0:
        imsave(fname, img)
        print('Image saved as', fname)
    end_time = time.time()
    print('Iteration %d completed in %ds' % (i, end_time - start_time))
