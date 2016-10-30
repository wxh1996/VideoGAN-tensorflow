from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *

class VideoGAN(object):

    def __init__(self, sess, image_size=128, batch_size=64, frame_size=32, crop_size=64):
        self.sess = sess
        self.batch_size = batch_size
        self.image_size = image_size

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')

        # if not self.y_dim:
        self.d_bn3 = batch_norm(name='d_bn3')

        self.gb_bn0 = batch_norm(name='gb_bn0')
        self.gb_bn1 = batch_norm(name='gb_bn1')
        self.gb_bn2 = batch_norm(name='gb_bn2')
        self.gb_bn3 = batch_norm(name='gb_bn3')

        self.gf_bn0 = batch_norm(name='gf_bn0')
        self.gf_bn1 = batch_norm(name='gf_bn1')
        self.gf_bn2 = batch_norm(name='gf_bn2')
        self.gf_bn3 = batch_norm(name='gf_bn3')
        # self.dataset_name = dataset_name
        # self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def g_background(self, z):

        self.z_, self.h0_w, self.h0_b = linear(z, 512*4*4, 'gb_h0_lin', with_w=True)

        self.h0 = tf.reshape(self.z_, [-1, 4, 4, 512])
        h0 = tf.nn.relu(self.gb_bn0(self.h0))

        self.h1, self.h1_w, self.h1_b = deconv2d(h0,
            [self.batch_size, 8, 8, 256], name='gb_h1', with_w=True)
        h1 = tf.nn.relu(self.gb_bn1(self.h1))

        h2, self.h2_w, self.h2_b = deconv2d(h1,
            [self.batch_size, 16, 16, 128], name='gb_h2', with_w=True)
        h2 = tf.nn.relu(self.gb_bn2(h2))

        h3, self.h3_w, self.h3_b = deconv2d(h2,
            [self.batch_size, 32, 32, 64], name='gb_h3', with_w=True)
        h3 = tf.nn.relu(self.gb_bn3(h3))

        h4, self.h4_w, self.h4_b = deconv2d(h3,
            [self.batch_size, 64, 64, 3], name='gb_h4', with_w=True)

        return tf.nn.tanh(h4)

    def g_foreground(self, z):

        self.z_, self.h0_w, self.h0_b = linear(z, 512*4*4*2, 'gf_h0_lin', with_w=True)

        self.h0 = tf.reshape(self.z_, [-1, 2, 4, 4, 512])
        h0 = tf.nn.relu(self.gf_bn0(self.h0))

        self.h1, self.h1_w, self.h1_b = deconv3d(h0,
            [self.batch_size, 4, 8, 8, 256], name='gf_h1', with_w=True)
        h1 = tf.nn.relu(self.gf_bn1(self.h1))

        h2, self.h2_w, self.h2_b = deconv3d(h1,
            [self.batch_size, 8, 16, 16, 128], name='gf_h2', with_w=True)
        h2 = tf.nn.relu(self.gf_bn2(h2))

        h3, self.h3_w, self.h3_b = deconv3d(h2,
            [self.batch_size, 16, 32, 32, 64], name='gf_h3', with_w=True)
        h3 = tf.nn.relu(self.gf_bn3(h3))

        h4, self.h4_w, self.h4_b = deconv3d(h3,
            [self.batch_size, 32, 64, 64, 3], name='gb_h4', with_w=True)

        mask1 = deconv3d(h3,
            [self.batch_size, 32, 64, 64, 1], name='mask1', with_w=False)

        mask = L1Penalty(tf.nn.sigmoid(mask1))

        # mask*h4 + (1 - mask)*

        return tf.nn.tanh(h4), mask

    def generator(self, z):

        gf4, mask = self.g_foreground(z)

        gb4 = self.g_background(z)

        gb4 = tf.reshape(gb4, [self.batch_size, 1, 64, 64, 3])
        gb4 = tf.tile(gb4, [1, 32, 1, 1, 1])

        return mask * gf4 + (1 - mask) * gb4

    def descriminator(self, video):

        h0 = lrelu(conv3d(video, 64, name='d_h0_conv'))
        h1 = lrelu(self.d_bn1(conv3d(h0, 128, name='d_h1_conv')))
        h2 = lrelu(self.d_bn2(conv3d(h1, 256, name='d_h2_conv')))
        h3 = lrelu(self.d_bn3(conv3d(h2, 512, name='d_h3_conv')))
        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

        return h4

    # def loss_function(self, z):
    def build_model(self):

        self.videos = tf.placeholder(tf.float32, [None, self.frame_size, self.crop_size, self.crop_size, self.c_dim],
                                    name='real_v')
        self.z =  tf.placeholder(tf.float32, [None, self.z_dim],
                                name='z')
        self.fake = self.generator(self.z)

        self.predict1 = self.discriminator(self.videos)
        self.lossf1 = (predict1, label)

        self.predict2 = self.discriminator(self.fake)
        self.lossf2 = ()
