from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from data import DataLoader
from ops import *
from utils import *


class VideoGAN(object):

    def __init__(self, sess, image_size=128, batch_size=64, frame_size=32, crop_size=64,
                 learning_rate=0.0002, beta1=0.5, z_dim=100):
        self.checkpoint_dir = 'output'
        self.sess = sess
        self.batch_size = batch_size
        self.image_size = image_size
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.z_dim = z_dim
        self.frame_size = frame_size
        self.crop_size = crop_size
        # batch normalization : deals with poor initialization helps gradient flow
        # self.d_bn1 = batch_norm(name='d_bn1')
        # self.d_bn2 = batch_norm(name='d_bn2')
        #
        # # if not self.y_dim:
        # self.d_bn3 = batch_norm(name='d_bn3')
        #
        # self.gb_bn0 = batch_norm(name='g_b_bn0')
        # self.gb_bn1 = batch_norm(name='g_b_bn1')
        # self.gb_bn2 = batch_norm(name='g_b_bn2')
        # self.gb_bn3 = batch_norm(name='g_b_bn3')
        #
        # self.gf_bn0 = batch_norm(name='g_f_bn0')
        # self.gf_bn1 = batch_norm(name='g_f_bn1')
        # self.gf_bn2 = batch_norm(name='g_f_bn2')
        # self.gf_bn3 = batch_norm(name='g_f_bn3')
        self.dataset_name = 'Golf'
        # self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def g_background(self, z):

        self.z_, self.h0_w, self.h0_b = linear(
            z, 512 * 4 * 4, 'g_b_h0_lin', with_w=True)

        self.h0 = tf.reshape(self.z_, [-1, 4, 4, 512])
        h0 = tf.nn.relu(tf.contrib.layers.batch_norm(self.h0, scope='g_b_bn0'))

        self.h1, self.h1_w, self.h1_b = deconv2d(h0,
                                                 [self.batch_size, 8, 8, 256], name='g_b_h1', with_w=True)
        h1 = tf.nn.relu(tf.contrib.layers.batch_norm(self.h1, scope='g_b_bn1'))

        h2, self.h2_w, self.h2_b = deconv2d(h1,
                                            [self.batch_size, 16, 16, 128], name='g_b_h2', with_w=True)
        h2 = tf.nn.relu(tf.contrib.layers.batch_norm(h2, scope='g_b_bn2'))

        h3, self.h3_w, self.h3_b = deconv2d(h2,
                                            [self.batch_size, 32, 32, 64], name='g_b_h3', with_w=True)
        h3 = tf.nn.relu(tf.contrib.layers.batch_norm(h3, scope='g_b_bn3'))

        h4, self.h4_w, self.h4_b = deconv2d(h3,
                                            [self.batch_size, 64, 64, 3], name='g_b_h4', with_w=True)

        return tf.nn.tanh(h4)

    def g_foreground(self, z):

        self.z_, self.h0_w, self.h0_b = linear(
            z, 512 * 4 * 4 * 2, 'g_f_h0_lin', with_w=True)

        self.h0 = tf.reshape(self.z_, [-1, 2, 4, 4, 512])
        h0 = tf.nn.relu(tf.contrib.layers.batch_norm(self.h0, scope='g_f_bn0'))

        self.h1, self.h1_w, self.h1_b = deconv3d(h0,
                                                 [self.batch_size, 4, 8, 8, 256], name='g_f_h1', with_w=True)
        h1 = tf.nn.relu(tf.contrib.layers.batch_norm(self.h1, scope='g_f_bn1'))

        h2, self.h2_w, self.h2_b = deconv3d(h1,
                                            [self.batch_size, 8, 16, 16, 128], name='g_f_h2', with_w=True)
        h2 = tf.nn.relu(tf.contrib.layers.batch_norm(h2, scope='g_f_bn2'))

        h3, self.h3_w, self.h3_b = deconv3d(h2,
                                            [self.batch_size, 16, 32, 32, 64], name='g_f_h3', with_w=True)
        h3 = tf.nn.relu(tf.contrib.layers.batch_norm(h3, scope='g_f_bn3'))

        h4, self.h4_w, self.h4_b = deconv3d(h3,
                                            [self.batch_size, 32, 64, 64, 3], name='g_f_h4', with_w=True)

        mask1 = deconv3d(h3,
                         [self.batch_size, 32, 64, 64, 1], name='g_mask1', with_w=False)

        mask = l1Penalty(tf.nn.sigmoid(mask1))

        # mask*h4 + (1 - mask)*

        return tf.nn.tanh(h4), mask

    def generator(self, z):

        gf4, mask = self.g_foreground(z)

        gb4 = self.g_background(z)

        gb4 = tf.reshape(gb4, [self.batch_size, 1, 64, 64, 3])
        gb4 = tf.tile(gb4, [1, 32, 1, 1, 1])

        return mask * gf4 + (1 - mask) * gb4

    def discriminator(self, video, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        h0 = lrelu(conv3d(video, 64, name='d_h0_conv'))
        h1 = lrelu(tf.contrib.layers.batch_norm(
            conv3d(h0, 128, name='d_h1_conv'), scope='d_bn1'))
        h2 = lrelu(tf.contrib.layers.batch_norm(
            conv3d(h1, 256, name='d_h2_conv'), scope='d_bn2'))
        h3 = lrelu(tf.contrib.layers.batch_norm(
            conv3d(h2, 512, name='d_h3_conv'), scope='d_bn3'))
        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

        return tf.nn.sigmoid(h4), h4

    # def loss_function(self, z):

    def save(self, checkpoint_dir, step):
        model_name = "VideoGAN.model"
        model_dir = "%s_%s" % (self.dataset_name, self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def build_model(self):

        self.videos = tf.placeholder(tf.float32, [self.batch_size, self.frame_size, self.crop_size, self.crop_size, 3],
                                     name='videos')
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim],
                                name='z')
        self.z_sum = tf.histogram_summary("z", self.z)
        self.G = self.generator(self.z)

        self.D, self.D_logits = self.discriminator(self.videos)
        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))

        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            self.D_logits_, tf.zeros_like(self.D_)))

        self.d_sum = tf.histogram_summary("d", self.D)
        self.d__sum = tf.histogram_summary("d_", self.D_)
        self.G_1 = tf.reshape(self.G[0, 0, :, :, :], [1, 64, 64, 3])
        self.G_sum = tf.image_summary("G", self.G_1)

        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            self.D_logits_, tf.ones_like(self.D_)))

        self.d_loss_real_sum = tf.scalar_summary(
            "d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.scalar_summary(
            "d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = tf.scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self):

        d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)
        self.sess.run(tf.initialize_all_variables())

        self.g_sum = tf.merge_summary([self.z_sum, self.d__sum, self.G_sum,
                                       self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.merge_summary(
            [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = tf.train.SummaryWriter("./logs", self.sess.graph)

        sample_z = np.random.uniform(-1, 1, size=(self.crop_size, self.z_dim))

        counter = 1
        start_time = time.time()
        # DATA PREPARED
        #.......
        dataloader = DataLoader()
        data_size = dataloader.size
        for epoch in xrange(50):

            batch_idxs = data_size // self.batch_size

            for idx in xrange(0, batch_idxs):

                batch_videos = dataloader.get_batch()
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]) \
                            .astype(np.float32)
                # Update D network
                _, summary_str = self.sess.run([d_optim, self.d_sum],
                                               feed_dict={self.videos: batch_videos, self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                errD_fake = self.d_loss_fake.eval(
                    {self.z: batch_z}, session=self.sess)
                errD_real = self.d_loss_real.eval(
                    {self.videos: batch_videos}, session=self.sess)
                errG = self.g_loss.eval({self.z: batch_z}, session=self.sess)

                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f"
                      % (epoch, idx, batch_idxs,
                         time.time() - start_time, errD_fake + errD_real, errG))

                # print batch_z
                if counter % 100 == 0:
                    G_sample = self.sess.run(
                        self.G, feed_dict={self.z: batch_z})
                    make_gif(G_sample[0],
                             './samples/traint_gif_%s.gif' % (idx))

                if np.mod(counter, 500) == 2:
                    self.save(self.checkpoint_dir, counter)

                counter += 1

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    sess = tf.Session()
    videogan = VideoGAN(sess)
    videogan.train()
