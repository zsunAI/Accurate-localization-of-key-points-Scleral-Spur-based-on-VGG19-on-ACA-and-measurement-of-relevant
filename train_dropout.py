import tensorflow as tf
import numpy as np
import cv2
import glob
import random
from Network import network
from tensorlayer.layers import *
import tensorlayer as tl
import os
from _read_data import read_train_data, read_test_data

lr_init = 1e-5 # 1e-5
batch_size =  32
x = tf.placeholder('float32', [None, 224, 224, 3])
y = tf.placeholder('float32', [None, 2])

keep_prob = tf.placeholder('float32')

def train(is_training):

    net_vgg, conv = network(x, reuse=False)
    ft_output = FlattenLayer(conv, name='flatten_1')
    ft_output = DenseLayer(ft_output, n_units=4096, act=tf.nn.relu, name='fc6_1')
    # ft_output = DropoutLayer(ft_output, keep=keep_prob, name='keep_1')
    ft_output = tf.nn.dropout(ft_output.outputs, keep_prob=keep_prob)
    ft_output = InputLayer(ft_output, name='drop_1')
    ft_output = DenseLayer(ft_output, n_units=4096, act=tf.nn.relu, name='fc7_1')
    # ft_output = DropoutLayer(ft_output, keep=keep_prob, name='keep_2')
    ft_output = tf.nn.dropout(ft_output.outputs, keep_prob=keep_prob)
    ft_output = InputLayer(ft_output, name='drop_2')
    ft_output = DenseLayer(ft_output, n_units=2, act=tf.identity, name='fc8_1')

    ##### ======================== DEFINE_TRAIN_OP =================###
    mse_loss = tl.cost.mean_squared_error(ft_output.outputs, y, is_mean=True)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)
    d_optim = tf.train.AdamOptimizer(lr_v, beta1=0.9).minimize(mse_loss)
    correct_pred = tf.sqrt(tf.abs(y[:, 0] - ft_output.outputs[:, 0])**2 + tf.abs(y[:, 1] - ft_output.outputs[:, 1]) ** 2) <= 15
    accur = tf.reduce_mean(tf.cast(correct_pred, 'float'))

    ##### ===================== load checkpoint  ============ ###
    # sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    sess = tf.Session()
    tl.layers.initialize_global_variables(sess)
    if tf.train.get_checkpoint_state('model'):
        saver = tf.train.Saver()
        saver.restore(sess, './model/latest')

    #### ============== load vgg params ============== #####
    if is_training:
        vgg_npy_path = 'vgg19.npy'
        if not os.path.isfile(vgg_npy_path):
            print('Please download vgg19.npz from : https://github.com/machrisaa/tensorflow-vgg')
            exit()

        npz = np.load(vgg_npy_path, encoding='latin1').item()
        params = []
        for var in sorted(npz.items()):
            W = np.asarray(var[1][0])
            b = np.asarray(var[1][1])
            params.extend([W, b])
        tl.files.assign_params(sess, params, net_vgg)
    ## =============== TRAIN ================== ###
    # 载入数据


        # 第六步：循环epoch，循环num_batch
        losses = []
        train_accur = []
        test_accur = []
        lossescs = []
        for e in range(500):#500
            train_vec_x, train_y = read_train_data()
            #train_vec_xnew=train_vec_x[0:1000]
            #train_ynew=train_vec_x[0:1000]
            for i in range(len(train_vec_x) // batch_size):
                # 第七步：根据索引值构造batch_x和batch_y
                batch_x = train_vec_x[i * batch_size:(i + 1) * batch_size]
                batch_y = train_y[i * batch_size:(i + 1) * batch_size]
                # 第八步：使用sess.run执行train_op和loss
                _, _loss, _accur = sess.run([d_optim, mse_loss, accur],
                                            feed_dict={x: batch_x, y: batch_y, keep_prob:0.7})
                test_x, test_y = read_test_data()
                test_x_batch = test_x[0:32]#[i * batch_size:(i + 1) * batch_size]#[0:64]#[0:-1]
                test_y_batch = test_y[0:32]#[i * batch_size:(i + 1) * batch_size]#[0:64]#[0:-1]
                _, _losscs, _accur1 = sess.run([d_optim, mse_loss, accur],
                                            feed_dict={x: test_x_batch, y: test_y_batch, keep_prob:0.7})				
                # 第九步：如果迭代1000次打印结果
                if i % 10 == 0 and i != 0:
                    _accur = sess.run(accur, feed_dict={x: batch_x, y: batch_y, keep_prob:1.0})
                    test_x, test_y = read_test_data()
                    test_x_batch = test_x[0:32]#[i * batch_size:(i + 1) * batch_size]#[0:64]#[0:-1]
                    test_y_batch = test_y[0:32]#[i * batch_size:(i + 1) * batch_size]#[0:64]#[0:-1]
                    _accur_test = sess.run(accur, feed_dict={x: test_x_batch, y: test_y_batch, keep_prob:1.0})
                    print('epoch', e, 'iter', i, 'loss:', _loss, '_accur:', _accur, '_accur_test', _accur_test,'losscs', _losscs)
                    #print('epoch', e, 'iter', i, '_accur_test:', _accur_test)
                    losses.append(_loss)
                    train_accur.append(_accur)
                    test_accur.append(_accur_test)
                    lossescs.append(_losscs)
                    with open('loss.txt', 'w') as f:
                        b = [f.write(str(i) + '\n') for i in losses]

                    with open('_accur.txt', 'w') as ac:
                        b = [ac.write(str(i) + '\n') for i in train_accur]

                    with open('_test_accur.txt', 'w') as af:
                        b = [af.write(str(i) + '\n') for i in test_accur]

                    with open('_losscs.txt', 'w') as af:
                        b = [af.write(str(i) + '\n') for i in lossescs]

            # 如果迭代了两次保存结果
            saver = tf.train.Saver()
            if not os.path.exists('model/'):
                os.makedirs('model/')
            if e % 20 == 0:
                saver.save(sess, './model/latest', write_meta_graph=False)  # .ckpt ???
				# saver.save(sess, './model/latest01')
    #### ==================== Test ================ ###
    else:
        show_num = 10
        test_x, test_y = read_train_data()
        test_x_show = test_x[0:show_num]
        test_y_show = test_y[0:show_num]
        _accur_test, _ft_output= sess.run([accur, ft_output.outputs],
                                    feed_dict={x: test_x_show, y: test_y_show, keep_prob:1.0})
        print('_accur', _accur_test)
        for i in range(show_num):
            clone_img_1 = test_x_show[i].copy()
            cv2.circle(clone_img_1, (_ft_output[i, 0], _ft_output[i, 1]), 3, (0, 0, 255), -1)
            cv2.circle(clone_img_1, (test_y[i, 0], test_y[i, 1]), 3, (0, 255, 0), -1)
            cv2.imshow('img', clone_img_1)
            cv2.waitKey(0)

if __name__ == '__main__':
    train(is_training=True)


