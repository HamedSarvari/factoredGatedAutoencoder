from http import client

import tensorflow as tf

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
# #print(device_lib.list_local_devices())
# tf.test.gpu_device_name()
#
#
# # from tensorflow.python.client import device_lib
# # print(device_lib.list_local_devices())
#
#with tf.device('/gpu:0'):
# with tf.device('/job:localhost/replica:0/task:0/device:XLA_GPU:0 '):
#
#     a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
#     b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
#     c = tf.matmul(a, b)
#
# with tf.Session() as sess:
#     print (sess.run(c))
#

# import tensorflow as tf
# if tf.test.gpu_device_name():
#     print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
# else:
#     print("Please install GPU version of TF")

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices()))