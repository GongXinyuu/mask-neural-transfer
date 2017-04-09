#!/usr/bin/python
# -*- coding:utf8 -*-
# from __future__ import print_function
# from keras.preprocessing.image import load_img, img_to_array
# from scipy.misc import imsave
# import numpy as np
# from scipy.optimize import fmin_l_bfgs_b
# import time
# import argparse
# import pylab
# from keras.applications import vgg16
# from keras import backend as K
# import tensorflow as tf
#
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

import numpy as np
import tensorflow as tf
# Declare list of features, we only have one real-valued feature
def model(features, labels, mode):
  # Build a linear model and predict values
  W = tf.get_variable("W", [1], dtype=tf.float64)
  b = tf.get_variable("b", [1], dtype=tf.float64)
  y = W*features['x'] + b
  # Loss sub-graph
  loss = tf.reduce_sum(tf.square(y - labels))
  # Training sub-graph
  global_step = tf.train.get_global_step()
  optimizer = tf.train.GradientDescentOptimizer(0.01)
  train = tf.group(optimizer.minimize(loss),
                   tf.assign_add(global_step, 1))
  # ModelFnOps connects subgraphs we built to the
  # appropriate functionality.
  return tf.contrib.learn.ModelFnOps(
      mode=mode, predictions=y,
      loss=loss,
      train_op=train)

estimator = tf.contrib.learn.Estimator(model_fn=model)
# define our data set
x = np.array([1., 2., 3., 4.])
y = np.array([0., -1., -2., -3.])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x}, y, 4, num_epochs=1000)

# train
estimator.fit(input_fn=input_fn, steps=1000)
# evaluate our model
print(estimator.evaluate(input_fn=input_fn, steps=10))
