#!/usr/bin/python
# -*- coding:utf8 -*-
from __future__ import print_function
from keras.preprocessing.image import load_img, img_to_array
from scipy.misc import imsave
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time
import argparse
import pylab
from keras.applications import vgg16
from keras import backend as K
import tensorflow as tf

# base_image_path = "/Users/gxy/Desktop/CS/CNN/Project/keras/Kexamples2.0/pic/Taylor2.JPeG"
# mask_path = "/Users/gxy/Desktop/CS/CNN/Project/keras/Kexamples2.0/pic/Taylor2_pascal_voc.png"
# width, height = load_img(base_image_path).size
# # img_nrows = 400 #height
# # img_ncols = int(width * img_nrows / height) #width
#
# def image_maskprocess(image_path, mask_path):
#     img_origin = load_img(image_path)  # 创建实例
#     img_origin = img_to_array(img_origin) # 将图片实例转化为张量
#     mask = load_img(mask_path)
#     mask = img_to_array(mask)
#     img = (mask > 0) * img_origin
#     return img
#
# mask_image = image_maskprocess(base_image_path, mask_path)
# mask_image = np.clip(mask_image, 0, 255).astype('uint8')
# pylab.imshow(mask_image)
# pylab.show()

W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(linear_model, {x:[1,2,3,4]}))