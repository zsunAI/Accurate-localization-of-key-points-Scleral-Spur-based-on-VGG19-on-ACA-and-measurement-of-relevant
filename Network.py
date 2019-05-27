import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *



def network(x, reuse):

    VGG_MEAN = [103.939, 116.779, 123.68]
    with tf.variable_scope('VGG_19', reuse=reuse):
        b, g, r = tf.split(x, 3, 3)
        bgr = tf.concat([b - VGG_MEAN[0], g - VGG_MEAN[1], r - VGG_MEAN[2]], axis=3)
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
        net_in = InputLayer(bgr, name='input')
        # 构建网络
        """conv1"""
        network = Conv2d(net_in, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', name='conv1_1')
        network = Conv2d(network, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', name='conv1_2')
        network = MaxPool2d(network, (2, 2), (2, 2), padding='SAME', name='pool1')
        '''conv2'''
        network = Conv2d(network, 128, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', name='conv2_1')
        network = Conv2d(network, 128, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', name='conv2_2')
        network = MaxPool2d(network, (2, 2), (2, 2), padding='SAME', name='pool2')
        '''conv3'''
        network = Conv2d(network, 256, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', name='conv3_1')
        network = Conv2d(network, 256, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', name='conv3_2')
        network = Conv2d(network, 256, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', name='conv3_3')
        network = Conv2d(network, 256, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', name='conv3_4')
        network = MaxPool2d(network, (2, 2), (2, 2), padding='SAME', name='pool3')
        '''conv4'''
        network = Conv2d(network, 512, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', name='conv4_1')
        network = Conv2d(network, 512, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', name='conv4_2')
        network = Conv2d(network, 512, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', name='conv4_3')
        network = Conv2d(network, 512, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', name='conv4_4')
        network = MaxPool2d(network, (2, 2), (2, 2), padding='SAME', name='pool4')
        '''conv5'''
        network = Conv2d(network, 512, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', name='conv5_1')
        network = Conv2d(network, 512, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', name='conv5_2')
        network = Conv2d(network, 512, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', name='conv5_3')
        network = Conv2d(network, 512, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', name='conv5_4')
        network = MaxPool2d(network, (2, 2), (2, 2), padding='SAME', name='pool5')
        conv = network
        """fc6-8"""
        network = FlattenLayer(network, name='flatten')
        network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc6')
        network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc7')
        network = DenseLayer(network, n_units=1000, act=tf.identity, name='fc8')
        # print('finish the bulid %fs' % (time.time() - start_time))
        return network, conv








