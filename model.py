from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
import json

from ops import *
from utils import *

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

class DCGAN(object):
  def __init__(self, sess, input_height=108, input_width=108, crop=True,
         batch_size=64, sample_num = 64, output_height=64, output_width=64,
         y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
         gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
         input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None):
    """

    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      y_dim: (optional) Dimension of dim for y. [None]
      z_dim: (optional) Dimension of dim for Z. [100]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
      gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
      c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
    """
    self.sess = sess
    self.crop = crop

    self.batch_size = batch_size
    self.sample_num = sample_num

    self.input_height = input_height
    self.input_width = input_width
    self.output_height = output_height
    self.output_width = output_width

    self.y_dim = y_dim
    self.z_dim = z_dim

    self.gf_dim = gf_dim
    self.df_dim = df_dim

    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim

    # batch normalization : deals with poor initialization helps gradient flow
    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')

    if not self.y_dim:
      self.d_bn3 = batch_norm(name='d_bn3')

    self.g_bn0 = batch_norm(name='g_bn0')
    self.g_bn1 = batch_norm(name='g_bn1')
    self.g_bn2 = batch_norm(name='g_bn2')

    if not self.y_dim:
      self.g_bn3 = batch_norm(name='g_bn3')

    self.dataset_name = dataset_name
    self.input_fname_pattern = input_fname_pattern
    self.checkpoint_dir = checkpoint_dir

    if self.dataset_name == 'mnist':
      self.data_X, self.data_y = self.load_mnist_w_digit(0) #(70000, 28, 28, 1), (70000, 10)
      self.c_dim = self.data_X[0].shape[-1]
    else:
      self.data = glob(os.path.join("./data", self.dataset_name, self.input_fname_pattern))
      self.c_dim = imread(self.data[0]).shape[-1]

    self.grayscale = (self.c_dim == 1)

    self.build_model()

  def build_model(self):
    if self.y_dim:
      self.y= tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y') # (64, 10)

    if self.crop:
      image_dims = [self.output_height, self.output_width, self.c_dim]
    else:
      image_dims = [self.input_height, self.input_width, self.c_dim] #(64, 64, 1)

    self.inputs = tf.placeholder(
      tf.float32, [self.batch_size] + image_dims, name='real_images') 
    self.sample_inputs = tf.placeholder(
      tf.float32, [self.sample_num] + image_dims, name='sample_inputs')

    inputs = self.inputs #(64, 28, 28, 1)
    sample_inputs = self.sample_inputs #(64, 28, 28, 1)

    self.z = tf.placeholder(
      tf.float32, [None, self.z_dim], name='z') #(?, 64)
    self.z_sum = histogram_summary("z", self.z)

    if self.y_dim:
      self.G = self.generator(self.z, self.y)
      self.D, self.D_logits = \
          self.discriminator(inputs, self.y, reuse=False)

      self.sampler = self.sampler(self.z, self.y, with_act=True)
      self.D_, self.D_logits_ = \
          self.discriminator(self.G, self.y, reuse=True)

      self.samplerD = self.samplerDis(inputs, self.y, with_act=True)
      self.samplerD_ = self.samplerDis(self.G, self.y, with_act=True)
    else:
      self.G = self.generator(self.z)
      self.D, self.D_logits = self.discriminator(inputs)

      self.sampler = self.sampler(self.z, with_act=True)
      self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

      self.samplerD = self.samplerDis(inputs, with_act=True)
      self.samplerD_ = self.samplerDis(self.G, with_act=True)

    self.d_sum = histogram_summary("d", self.D)
    self.d__sum = histogram_summary("d_", self.D_)
    self.G_sum = image_summary("G", self.G)

    def sigmoid_cross_entropy_with_logits(x, y):
      try:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
      except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

    self.d_loss_real = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
    self.d_loss_fake = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
    self.g_loss = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

    self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
    self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
                          
    self.d_loss = self.d_loss_real + self.d_loss_fake

    self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
    self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

    t_vars = tf.trainable_variables()

    #JP: print all variables' name
    #for var in t_vars:
    #  print var.name
    #  print var.shape

    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]

    self.saver = tf.train.Saver()

  def train(self, config):
    d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.d_loss, var_list=self.d_vars)
    g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.g_loss, var_list=self.g_vars)
    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()

    self.g_sum = merge_summary([self.z_sum, self.d__sum,
      self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
    self.d_sum = merge_summary(
        [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
    self.writer = SummaryWriter("./logs", self.sess.graph)

    sample_z = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim))
    
    if config.dataset == 'mnist':
      #sample_inputs = self.data_X[0:self.sample_num]
      #sample_labels = self.data_y[0:self.sample_num]
      
      '''
      # control the input of samples
      wanted = [1, 4, 7, 9, 0, 2, 3, 5];
      rowsize = math.sqrt(self.sample_num)
      idlist = []
      accid = 0;
      for wid in range(0, len(wanted)):
        counter = 0;
        for idx in range(accid, self.data_y.shape[0]):
          if self.data_y[idx][wanted[wid]]==1:
            idlist.append(idx)
            counter += 1
            if counter==rowsize:
              accid = idx+1;
              break;
      sample_inputs = []
      sample_labels = []
      for sid in range(0, len(idlist)):
        sample_inputs.append(self.data_X[idlist[sid]])
        sample_labels.append(self.data_y[idlist[sid]])
      '''

    else:
      sample_files = self.data[0:self.sample_num]
      sample = [
          get_image(sample_file,
                    input_height=self.input_height,
                    input_width=self.input_width,
                    resize_height=self.output_height,
                    resize_width=self.output_width,
                    crop=self.crop,
                    grayscale=self.grayscale) for sample_file in sample_files]
      if (self.grayscale):
        sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
      else:
        sample_inputs = np.array(sample).astype(np.float32)
  
    counter = 1
    start_time = time.time()
    could_load, checkpoint_counter = self.load(self.checkpoint_dir)
    if could_load:
      counter = checkpoint_counter
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    for epoch in xrange(config.epoch):
      if config.dataset == 'mnist':
        batch_idxs = min(len(self.data_X), config.train_size) // config.batch_size
      else:      
        self.data = glob(os.path.join(
          "./data", config.dataset, self.input_fname_pattern))
        batch_idxs = min(len(self.data), config.train_size) // config.batch_size

      for idx in xrange(0, batch_idxs):
        if config.dataset == 'mnist':
          batch_images = self.data_X[idx*config.batch_size:(idx+1)*config.batch_size]
          batch_labels = self.data_y[idx*config.batch_size:(idx+1)*config.batch_size]
        else:
          batch_files = self.data[idx*config.batch_size:(idx+1)*config.batch_size]
          batch = [
              get_image(batch_file,
                        input_height=self.input_height,
                        input_width=self.input_width,
                        resize_height=self.output_height,
                        resize_width=self.output_width,
                        crop=self.crop,
                        grayscale=self.grayscale) for batch_file in batch_files]
          if self.grayscale:
            batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
          else:
            batch_images = np.array(batch).astype(np.float32)

        batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
              .astype(np.float32)

        if config.dataset == 'mnist':
          # Update D network
          _, summary_str = self.sess.run([d_optim, self.d_sum],
            feed_dict={ 
              self.inputs: batch_images,
              self.z: batch_z,
              self.y:batch_labels,
            })
          self.writer.add_summary(summary_str, counter)

          # Update G network
          _, summary_str = self.sess.run([g_optim, self.g_sum],
            feed_dict={
              self.z: batch_z, 
              self.y:batch_labels,
            })
          self.writer.add_summary(summary_str, counter)

          # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
          _, summary_str = self.sess.run([g_optim, self.g_sum],
            feed_dict={ 
              self.z: batch_z, 
              self.y:batch_labels 
            })
          self.writer.add_summary(summary_str, counter)
          
          errD_fake = self.d_loss_fake.eval({
              self.z: batch_z, 
              self.y:batch_labels
          })
          errD_real = self.d_loss_real.eval({
              self.inputs: batch_images,
              self.y:batch_labels
          })
          errG = self.g_loss.eval({
              self.z: batch_z,
              self.y: batch_labels
          })
        else:
          # Update D network
          _, summary_str = self.sess.run([d_optim, self.d_sum],
            feed_dict={ self.inputs: batch_images, self.z: batch_z })
          self.writer.add_summary(summary_str, counter)

          # Update G network
          _, summary_str = self.sess.run([g_optim, self.g_sum],
            feed_dict={ self.z: batch_z })
          self.writer.add_summary(summary_str, counter)

          # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
          _, summary_str = self.sess.run([g_optim, self.g_sum],
            feed_dict={ self.z: batch_z })
          self.writer.add_summary(summary_str, counter)
          
          errD_fake = self.d_loss_fake.eval({ self.z: batch_z })
          errD_real = self.d_loss_real.eval({ self.inputs: batch_images })
          errG = self.g_loss.eval({self.z: batch_z})

        counter += 1
        print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
          % (epoch, idx, batch_idxs,
            time.time() - start_time, errD_fake+errD_real, errG))

        num_sps = 640;
        num_bch = int(num_sps/self.sample_num);
        print num_bch;
        if np.mod(counter, 3) == 1:
          if config.dataset == 'mnist':
            d_loss = errD_fake+errD_real
            g_loss = errG;

            g_d_smp = None;
            g_d_h0r = None;
            g_d_h1r = None;
            g_d_h2r = None;
            g_samplesD = None;
            for itr in xrange(num_bch):
              sample_inputs = self.data_X[itr*self.sample_num : (itr+1)*self.sample_num]
              sample_labels = self.data_y[itr*self.sample_num : (itr+1)*self.sample_num]

              [samplesD, d_smp, d_h0, d_h0r, d_h1, d_h1r, d_h2, d_h2r, d_h3] = self.sess.run(self.samplerD, 
                feed_dict={
                  self.inputs: sample_inputs,
                  self.y: sample_labels,
                }
              )

              if itr==0:
                g_d_smp = d_smp;
                g_d_h0r = d_h0r;
                g_d_h1r = d_h1r;
                g_d_h2r = d_h2r.reshape(64,32,32,1);
                g_samplesD = samplesD;
              else:
                g_d_smp = np.concatenate([g_d_smp, d_smp], 0);
                g_d_h0r = np.concatenate([g_d_h0r, d_h0r], 0);
                g_d_h1r = np.concatenate([g_d_h1r, d_h1r], 0);
                g_d_h2r = np.concatenate([g_d_h2r, d_h2r.reshape(64,32,32,1)], 0);
                g_samplesD = np.concatenate([g_samplesD, samplesD], 0);

            #use the new order to sort the five array along the first dimension
            g_spD = np.array(g_samplesD).flatten().tolist();
            idxodr = np.argsort(g_spD);

            # reorder
            g_d_smp = g_d_smp[idxodr];
            g_d_h0r = g_d_h0r[idxodr];
            g_d_h1r = g_d_h1r[idxodr];
            g_d_h2r = g_d_h2r[idxodr];
            g_samplesD = g_samplesD[idxodr];

            # save the reordered results
            for itr in xrange(num_bch):
              manifold_h = int(np.ceil(np.sqrt(d_smp.shape[0])))
              manifold_w = int(np.floor(np.sqrt(d_smp.shape[0])))
              fc_size = int(np.sqrt(self.gfc_dim))

              directory = '{}/{:04d}'.format(config.sample_dir, counter)
              if not os.path.exists(directory):
                os.makedirs(directory)

              # save the activation map of D
              dirD = '{}/D'.format(directory)
              if not os.path.exists(dirD):
                os.makedirs(dirD)
              save_images(g_d_smp[itr*self.sample_num:(itr+1)*self.sample_num,:,:,:], [manifold_h, manifold_w],
                    './{}/T_img_{:02d}_{:04d}_{:03d}.png'.format(dirD, epoch, idx, itr))
              '''
              with open('./{}/T_h3_{:02d}_{:04d}_{:03d}.txt'.format(dirD, epoch, idx, itr), 'w') as f:
                for i in range(0, d_smp.shape[0]):
                  f.write("%f\n" % g_samplesD[itr*self.sample_num+i, 0]) 
              # save D as a json file
              dataD = [];
              append_layer(dataD, 'input', [64, 28, 28, 1],  g_d_smp[itr*self.sample_num:(itr+1)*self.sample_num,:,:,:]);
              append_layer(dataD, 'relu',   [64, 14, 14, 11],  g_d_h0r[itr*self.sample_num:(itr+1)*self.sample_num,:,:,:]);
              append_layer(dataD, 'relu',   [64, 7, 7, 74],   g_d_h1r[itr*self.sample_num:(itr+1)*self.sample_num,:,:,:]);
              append_layer(dataD, 'relu',   [64, 32, 32, 1],  g_d_h2r[itr*self.sample_num:(itr+1)*self.sample_num,:,:,:]);
              append_layer(dataD, 'sigmoid', [64, 1, 1, 1],  g_samplesD[itr*self.sample_num:(itr+1)*self.sample_num,0]);
              with open('./{}/T_{:02d}_{:04d}_{:03d}.json'.format(dirD, epoch, idx, itr), 'w') as f:
                json.dump(dataD, f)
              '''

            dataD = [];
            append_layer(dataD, 'input', [num_sps, 28, 28, 1],  g_d_smp);
            append_layer(dataD, 'relu',  [num_sps, 14, 14, 11],  g_d_h0r);
            append_layer(dataD, 'relu',  [num_sps, 7, 7, 74],   g_d_h1r);
            append_layer(dataD, 'relu',  [num_sps, 32, 32, 1],  g_d_h2r);
            append_layer(dataD, 'sigmoid', [num_sps, 1, 1, 1],  g_samplesD);
            with open('./{}/T_{:02d}_{:04d}.json'.format(dirD, epoch, idx), 'w') as f:
              json.dump(dataD, f)

            with open('./{}/prob_{:02d}_{:04d}.csv'.format(dirD, epoch, idx), 'w') as f:
              f.write("n_odr,o_odr,prob\n")
              for i in range(0, num_sps):
                f.write('{:d},{:d},{:f}\n'.format(i, idxodr[i], g_samplesD[i,0]));

            '''
            [samples, h0, h0r, h1, h1r, h2, h2r, h3], d_loss, g_loss = self.sess.run(
              [self.sampler, self.d_loss, self.g_loss],
              feed_dict={
                  self.z: sample_z,
                  self.inputs: sample_inputs,
                  self.y:sample_labels,
              }
            )

            [samplesD, d_smp, d_h0, d_h0r, d_h1, d_h1r, d_h2, d_h2r, d_h3], \
              [samplesD_, d_smp_, d_h0_, d_h0r_, d_h1_, d_h1r_, d_h2_, d_h2r_, d_h3_] = self.sess.run(
              [self.samplerD, self.samplerD_], 
              feed_dict={
                self.inputs: sample_inputs,
                self.y: sample_labels,
                self.G: samples,
              }
            )

            manifold_h = int(np.ceil(np.sqrt(samples.shape[0])))
            manifold_w = int(np.floor(np.sqrt(samples.shape[0])))
            fc_size = int(np.sqrt(self.gfc_dim))

            directory = '{}/{:04d}'.format(config.sample_dir, counter)
            if not os.path.exists(directory):
              os.makedirs(directory)

            dirG = '{}/G'.format(directory)
            if not os.path.exists(dirG):
              os.makedirs(dirG)
            # save the activation map of G
            save_images(samples, [manifold_h, manifold_w],
                  './{}/G_smp_{:02d}_{:04d}.png'.format(dirG, epoch, idx))
            save_images(np.reshape(h0, [64, fc_size, fc_size, 1]), [manifold_h, manifold_w],
                  './{}/G_h0_{:02d}_{:04d}.png'.format(dirG, epoch, idx))
            for ii in range(0, h1.shape[-1]):
              save_images(np.reshape(h1[:,:,:,ii], [h1.shape[0],h1.shape[1],h1.shape[2], 1]), [manifold_h, manifold_w],
                  './{}/G_h1_{:02d}_{:04d}_{:03d}.png'.format(dirG, epoch, idx, ii))
            for ii in range(0, h2.shape[-1]):
              save_images(np.reshape(h2[:,:,:,ii], [h2.shape[0],h2.shape[1],h2.shape[2], 1]), [manifold_h, manifold_w],
                  './{}/G_h2_{:02d}_{:04d}_{:03d}.png'.format(dirG, epoch, idx, ii))
            save_images(h3, [manifold_h, manifold_w],
                  './{}/G_h3_{:02d}_{:04d}.png'.format(dirG, epoch, idx))
            # save G as a json file
            data = [];
            append_layer(data, 'linear', [64, 32, 32, 1],   h0);
            append_layer(data, 'relu',   [64, 32, 32, 1],   h0r);
            append_layer(data, 'linear', [64, 7, 7, 128],   h1);
            append_layer(data, 'relu',   [64, 7, 7, 128],   h1r);
            append_layer(data, 'deconv', [64, 14, 14, 128], h2);
            append_layer(data, 'relu',   [64, 14, 14, 128], h2r);
            append_layer(data, 'deconv', [64, 28, 28, 1],   h3);
            append_layer(data, 'sigmoid', [64, 28, 28, 1],  samples);
            with open('./{}/G_{:02d}_{:04d}.json'.format(dirG, epoch, idx), 'w') as f:
              json.dump(data, f)
            
            # save the activation map of D
            dirD = '{}/D'.format(directory)
            if not os.path.exists(dirD):
              os.makedirs(dirD)
            save_images(d_smp, [manifold_h, manifold_w],
                  './{}/T_img_{:02d}_{:04d}.png'.format(dirD, epoch, idx))
            for ii in range(0, d_h0.shape[-1]):
              save_images(np.reshape(d_h0[:,:,:,ii], [d_h0.shape[0],d_h0.shape[1],d_h0.shape[2], 1]), [manifold_h, manifold_w],
                  './{}/T_h0_{:02d}_{:04d}_{:03d}.png'.format(dirD, epoch, idx, ii))
            for ii in range(0, d_h1.shape[-1]):
              save_images(np.reshape(d_h1[:,:,:,ii], [d_h1.shape[0],d_h1.shape[1],d_h1.shape[2], 1]), [manifold_h, manifold_w],
                  './{}/T_h1_{:02d}_{:04d}_{:03d}.png'.format(dirD, epoch, idx, ii))
            save_images(np.reshape(d_h2, [64, fc_size, fc_size, 1]), [manifold_h, manifold_w],
                  './{}/T_h2_{:02d}_{:04d}.png'.format(dirD, epoch, idx))
            with open('./{}/T_h3_{:02d}_{:04d}.txt'.format(dirD, epoch, idx), 'w') as f:
              for i in range(0, samplesD.shape[0]):
                f.write("%f\n" % samplesD[i, 0]) 
            # save D as a json file
            dataD = [];
            append_layer(dataD, 'input', [64, 28, 28, 1],  d_smp);
            append_layer(dataD, 'conv',   [64, 14, 14, 11], d_h0);
            append_layer(dataD, 'relu',   [64, 14, 14, 11],  d_h0r);
            append_layer(dataD, 'conv',   [64, 7, 7, 74],   d_h1);
            append_layer(dataD, 'relu',   [64, 7, 7, 74],   d_h1r);
            append_layer(dataD, 'linear', [64, 32, 32, 1],  d_h2);
            append_layer(dataD, 'relu',   [64, 32, 32, 1],  d_h2r);
            append_layer(dataD, 'linear', [64, 1, 1, 1],    d_h3);
            append_layer(dataD, 'sigmoid', [64, 1, 1, 1],  samplesD);
            with open('./{}/T_{:02d}_{:04d}.json'.format(dirD, epoch, idx), 'w') as f:
              json.dump(dataD, f)

            # save the activation map of D_
            dirD_ = '{}/D_'.format(directory)
            if not os.path.exists(dirD_):
              os.makedirs(dirD_)
            save_images(d_smp_, [manifold_h, manifold_w],
                  './{}/F_img_{:02d}_{:04d}.png'.format(dirD_, epoch, idx))
            for ii in range(0, d_h0_.shape[-1]):
              save_images(np.reshape(d_h0_[:,:,:,ii], [d_h0_.shape[0],d_h0_.shape[1],d_h0_.shape[2], 1]), [manifold_h, manifold_w],
                  './{}/F_h0_{:02d}_{:04d}_{:03d}.png'.format(dirD_, epoch, idx, ii))
            for ii in range(0, d_h1_.shape[-1]):
              save_images(np.reshape(d_h1_[:,:,:,ii], [d_h1_.shape[0],d_h1_.shape[1],d_h1_.shape[2], 1]), [manifold_h, manifold_w],
                  './{}/F_h1_{:02d}_{:04d}_{:03d}.png'.format(dirD_, epoch, idx, ii))
            save_images(np.reshape(d_h2_, [64, fc_size, fc_size, 1]), [manifold_h, manifold_w],
                  './{}/F_h2_{:02d}_{:04d}.png'.format(dirD_, epoch, idx))
            with open('./{}/F_h3_{:02d}_{:04d}.txt'.format(dirD_, epoch, idx), 'w') as f:
              for i in range(0, samplesD_.shape[0]):
                f.write("%f\n" % samplesD_[i, 0]) 
            # save D_ as a json file
            dataD_ = [];
            append_layer(dataD_, 'input', [64, 28, 28, 1],  d_smp_);
            append_layer(dataD_, 'conv',   [64, 14, 14, 11], d_h0_);
            append_layer(dataD_, 'relu',   [64, 14, 14, 11],  d_h0r_);
            append_layer(dataD_, 'conv',   [64, 7, 7, 74],   d_h1_);
            append_layer(dataD_, 'relu',   [64, 7, 7, 74],   d_h1r_);
            append_layer(dataD_, 'linear', [64, 32, 32, 1],  d_h2_);
            append_layer(dataD_, 'relu',   [64, 32, 32, 1],  d_h2r_);
            append_layer(dataD_, 'linear', [64, 1, 1, 1],    d_h3_);
            append_layer(dataD_, 'sigmoid', [64, 1, 1, 1],  samplesD_);
            with open('./{}/F_{:02d}_{:04d}.json'.format(dirD_, epoch, idx), 'w') as f:
              json.dump(dataD_, f)
            '''
            
            print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

          else:
            try:
              [samples, z, h0, h0r, h1, h1r, h2, h2r, h3, h3r, h4], d_loss, g_loss = self.sess.run(
                [self.sampler, self.d_loss, self.g_loss],
                feed_dict={
                    self.z: sample_z,
                    self.inputs: sample_inputs,
                },
              )

              [samplesD, d_smp, d_h0, d_h0r, d_h1, d_h1r, d_h2, d_h2r, d_h3, d_h3r, d_h4], \
                [samplesD_, d_smp_, d_h0_, d_h0r_, d_h1_, d_h1r_, d_h2_, d_h2r_, d_h3_, d_h3r_, d_h4_] = self.sess.run(
                [self.samplerD, self.samplerD_], 
                feed_dict={
                  self.inputs: sample_inputs,
                  self.G: samples,
                }
              )
              
              manifold_h = int(np.ceil(np.sqrt(samples.shape[0])))
              manifold_w = int(np.floor(np.sqrt(samples.shape[0])))

              directory = '{}/{:04d}'.format(config.sample_dir, counter)
              if not os.path.exists(directory):
                os.makedirs(directory)

              dirG = '{}/G'.format(directory)
              if not os.path.exists(dirG):
                os.makedirs(dirG)
              # save the activation map of G
              save_images(samples, [manifold_h, manifold_w],
                    './{}/G_smp_{:02d}_{:04d}.png'.format(dirG, epoch, idx))
              
              z_sqr = int(np.sqrt(self.z_dim))
              save_images(np.reshape(z, [self.batch_size, z_sqr, z_sqr, 1]), [manifold_h, manifold_w],
                    './{}/G_z_{:02d}_{:04d}.png'.format(dirG, epoch, idx))
              
              for ii in range(0, h0.shape[-1]):
                save_images(np.reshape(h0[:,:,:,ii], [h0.shape[0],h0.shape[1],h0.shape[2], 1]), [manifold_h, manifold_w],
                    './{}/G_h0_{:02d}_{:04d}_{:03d}.png'.format(dirG, epoch, idx, ii))
              
              for ii in range(0, h1.shape[-1]):
                save_images(np.reshape(h1[:,:,:,ii], [h1.shape[0],h1.shape[1],h1.shape[2], 1]), [manifold_h, manifold_w],
                    './{}/G_h1_{:02d}_{:04d}_{:03d}.png'.format(dirG, epoch, idx, ii))

              for ii in range(0, h2.shape[-1]):
                save_images(np.reshape(h2[:,:,:,ii], [h2.shape[0],h2.shape[1],h2.shape[2], 1]), [manifold_h, manifold_w],
                    './{}/G_h2_{:02d}_{:04d}_{:03d}.png'.format(dirG, epoch, idx, ii))

              for ii in range(0, h3.shape[-1]):
                save_images(np.reshape(h3[:,:,:,ii], [h3.shape[0],h3.shape[1],h3.shape[2], 1]), [manifold_h, manifold_w],
                    './{}/G_h3_{:02d}_{:04d}_{:03d}.png'.format(dirG, epoch, idx, ii))

              save_images(h4, [manifold_h, manifold_w],
                './{}/G_h4_{:02d}_{:04d}.png'.format(dirG, epoch, idx))
              
              # save G as a json file
              data = [];
              bz = self.batch_size
              append_layer(data, 'input',   [bz, 10, 10, 1],   z);
              append_layer(data, 'linear',  [bz, 4, 4, 512],   h0);
              append_layer(data, 'relu',    [bz, 4, 4, 512],   h0r);
              append_layer(data, 'deconv',  [bz, 8, 8, 256],   h1);
              append_layer(data, 'relu',    [bz, 8, 8, 256],   h1r);
              append_layer(data, 'deconv',  [bz, 16, 16, 128], h2);
              append_layer(data, 'relu',    [bz, 16, 16, 128], h2r);
              append_layer(data, 'deconv',  [bz, 32, 32, 64],  h3);
              append_layer(data, 'relu',    [bz, 32, 32, 64],  h3r);
              append_layer(data, 'deconv',  [bz, 64, 64, 3],   h4);
              append_layer(data, 'tanh',    [bz, 64, 64, 3],   samples);

              with open('./{}/G_{:02d}_{:04d}.json'.format(dirG, epoch, idx), 'w') as f:
                json.dump(data, f)

              # dump data of D
              dirD = '{}/D'.format(directory)
              if not os.path.exists(dirD):
                os.makedirs(dirD)

              save_images(d_smp, [manifold_h, manifold_w],
                    './{}/T_img_{:02d}_{:04d}.png'.format(dirD, epoch, idx))
              for ii in range(0, d_h0.shape[-1]):
                save_images(np.reshape(d_h0[:,:,:,ii], [d_h0.shape[0],d_h0.shape[1],d_h0.shape[2], 1]), [manifold_h, manifold_w],
                    './{}/T_h0_{:02d}_{:04d}_{:03d}.png'.format(dirD, epoch, idx, ii))
              for ii in range(0, d_h1.shape[-1]):
                save_images(np.reshape(d_h1[:,:,:,ii], [d_h1.shape[0],d_h1.shape[1],d_h1.shape[2], 1]), [manifold_h, manifold_w],
                    './{}/T_h1_{:02d}_{:04d}_{:03d}.png'.format(dirD, epoch, idx, ii))
              for ii in range(0, d_h2.shape[-1]):
                save_images(np.reshape(d_h2[:,:,:,ii], [d_h2.shape[0],d_h2.shape[1],d_h2.shape[2], 1]), [manifold_h, manifold_w],
                    './{}/T_h2_{:02d}_{:04d}_{:03d}.png'.format(dirD, epoch, idx, ii))
              for ii in range(0, d_h3.shape[-1]):
                save_images(np.reshape(d_h3[:,:,:,ii], [d_h3.shape[0],d_h3.shape[1],d_h3.shape[2], 1]), [manifold_h, manifold_w],
                    './{}/T_h3_{:02d}_{:04d}_{:03d}.png'.format(dirD, epoch, idx, ii))
              with open('./{}/T_h4_{:02d}_{:04d}.txt'.format(dirD, epoch, idx), 'w') as f:
                for i in range(0, samplesD.shape[0]):
                  f.write("%f\n" % samplesD[i, 0]) 

              # save D as a json file
              dataD = [];
              append_layer(dataD, 'input',  [bz, 64, 64, 3],    d_smp);
              append_layer(dataD, 'conv',   [bz, 32, 32, 64],   d_h0);
              append_layer(dataD, 'relu',   [bz, 32, 32, 64],   d_h0r);
              append_layer(dataD, 'conv',   [bz, 16, 16, 128],  d_h1);
              append_layer(dataD, 'relu',   [bz, 16, 16, 128],  d_h1r);
              append_layer(dataD, 'conv',   [bz, 8, 8, 256],    d_h2);
              append_layer(dataD, 'relu',   [bz, 8, 8, 256],    d_h2r);
              append_layer(dataD, 'conv',   [bz, 4, 4, 512],    d_h3);
              append_layer(dataD, 'relu',   [bz, 4, 4, 512],    d_h3r);
              append_layer(dataD, 'linear', [bz, 1, 1, 1],      d_h4);
              append_layer(dataD, 'sigmoid',[bz, 1, 1, 1],      samplesD);

              with open('./{}/T_{:02d}_{:04d}.json'.format(dirD, epoch, idx), 'w') as f:
                json.dump(dataD, f)

              # dump data of D_
              dirD_ = '{}/D_'.format(directory)
              if not os.path.exists(dirD_):
                os.makedirs(dirD_)

              save_images(d_smp_, [manifold_h, manifold_w],
                    './{}/F_img_{:02d}_{:04d}.png'.format(dirD_, epoch, idx))
              for ii in range(0, d_h0_.shape[-1]):
                save_images(np.reshape(d_h0_[:,:,:,ii], [d_h0_.shape[0],d_h0_.shape[1],d_h0_.shape[2], 1]), [manifold_h, manifold_w],
                    './{}/F_h0_{:02d}_{:04d}_{:03d}.png'.format(dirD_, epoch, idx, ii))
              for ii in range(0, d_h1_.shape[-1]):
                save_images(np.reshape(d_h1_[:,:,:,ii], [d_h1_.shape[0],d_h1_.shape[1],d_h1_.shape[2], 1]), [manifold_h, manifold_w],
                    './{}/F_h1_{:02d}_{:04d}_{:03d}.png'.format(dirD_, epoch, idx, ii))
              for ii in range(0, d_h2_.shape[-1]):
                save_images(np.reshape(d_h2_[:,:,:,ii], [d_h2_.shape[0],d_h2_.shape[1],d_h2_.shape[2], 1]), [manifold_h, manifold_w],
                    './{}/F_h2_{:02d}_{:04d}_{:03d}.png'.format(dirD_, epoch, idx, ii))
              for ii in range(0, d_h3_.shape[-1]):
                save_images(np.reshape(d_h3_[:,:,:,ii], [d_h3_.shape[0],d_h3_.shape[1],d_h3_.shape[2], 1]), [manifold_h, manifold_w],
                    './{}/F_h3_{:02d}_{:04d}_{:03d}.png'.format(dirD_, epoch, idx, ii))
              with open('./{}/F_h4_{:02d}_{:04d}.txt'.format(dirD_, epoch, idx), 'w') as f:
                for i in range(0, samplesD_.shape[0]):
                  f.write("%f\n" % samplesD_[i, 0]) 

              # save D as a json file
              dataD_ = [];
              append_layer(dataD_, 'input',  [bz, 64, 64, 3],    d_smp_);
              append_layer(dataD_, 'conv',   [bz, 32, 32, 64],   d_h0_);
              append_layer(dataD_, 'relu',   [bz, 32, 32, 64],   d_h0r_);
              append_layer(dataD_, 'conv',   [bz, 16, 16, 128],  d_h1_);
              append_layer(dataD_, 'relu',   [bz, 16, 16, 128],  d_h1r_);
              append_layer(dataD_, 'conv',   [bz, 8, 8, 256],    d_h2_);
              append_layer(dataD_, 'relu',   [bz, 8, 8, 256],    d_h2r_);
              append_layer(dataD_, 'conv',   [bz, 4, 4, 512],    d_h3_);
              append_layer(dataD_, 'relu',   [bz, 4, 4, 512],    d_h3r_);
              append_layer(dataD_, 'linear', [bz, 1, 1, 1],      d_h4_);
              append_layer(dataD_, 'sigmoid',[bz, 1, 1, 1],      samplesD_);

              with open('./{}/F_{:02d}_{:04d}.json'.format(dirD_, epoch, idx), 'w') as f:
                json.dump(dataD_, f)

              print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
            except:
              print("one pic error!...")

        if np.mod(counter, 500) == 2:
          self.save(config.checkpoint_dir, counter)

        #JP: print out the value of trained variables each iteration
        #for var in self.g_vars:
        #  print "========"
        #  print var.name, var.shape
        #  print var.eval()

        #for var in self.d_vars:
        #  print "========"
        #  print var.name, var.shape
        #  print var.eval()

  def discriminator(self, image, y=None, reuse=False):
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()

      if not self.y_dim:
        h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
        h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
        h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

        return tf.nn.sigmoid(h4), h4
      else:
        yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
        x = conv_cond_concat(image, yb)

        h0 = lrelu(conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv'))
        h0 = conv_cond_concat(h0, yb)

        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))
        h1 = tf.reshape(h1, [self.batch_size, -1])      
        h1 = concat([h1, y], 1)
        
        h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
        h2 = concat([h2, y], 1)

        h3 = linear(h2, 1, 'd_h3_lin')
        
        return tf.nn.sigmoid(h3), h3

  def samplerDis(self, image, y=None, with_act=False):
    with tf.variable_scope("discriminator") as scope:
      scope.reuse_variables()

      if not self.y_dim:
        h0 = conv2d(image, self.df_dim, name='d_h0_conv')
        h0r = lrelu(h0)

        h1 = conv2d(h0r, self.df_dim*2, name='d_h1_conv')
        h1r = lrelu(self.d_bn1(h1, train=False))

        h2 = conv2d(h1r, self.df_dim*4, name='d_h2_conv')
        h2r = lrelu(self.d_bn2(h2, train=False))

        h3 = conv2d(h2r, self.df_dim*8, name='d_h3_conv')
        h3r = lrelu(self.d_bn3(h3, train=False))

        h4 = linear(tf.reshape(h3r, [self.batch_size, -1]), 1, 'd_h3_lin')
        h4r = tf.nn.sigmoid(h4)

        if with_act:
          return h4r, image, h0, h0r, h1, h1r, h2, h2r, h3, h3r, h4
        else:
          return h4r
      else:
        yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
        x = conv_cond_concat(image, yb)

        h0_ = conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv')
        h0r = lrelu(h0_)
        h0 = conv_cond_concat(h0r, yb)

        h1_ = conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')
        h1r = lrelu(self.d_bn1(h1_, train=False))
        h1 = tf.reshape(h1r, [self.batch_size, -1])      
        h1 = concat([h1, y], 1)
        
        h2_ = linear(h1, self.dfc_dim, 'd_h2_lin')
        h2r = lrelu(self.d_bn2(h2_, train=False))
        h2 = concat([h2r, y], 1)

        h3_ = linear(h2, 1, 'd_h3_lin')
        h3 = tf.nn.sigmoid(h3_)

        if with_act:
          return h3, image, h0_, h0r, h1_, h1r, h2_, h2r, h3_
        else:
          return h3

  def generator(self, z, y=None):
    with tf.variable_scope("generator") as scope:
      if not self.y_dim:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        # project `z` and reshape
        self.z_, self.h0_w, self.h0_b = linear(
            z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)
        self.h0 = tf.reshape(
            self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
        h0 = tf.nn.relu(self.g_bn0(self.h0))

        self.h1, self.h1_w, self.h1_b = deconv2d(
            h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1', with_w=True)
        h1 = tf.nn.relu(self.g_bn1(self.h1))

        h2, self.h2_w, self.h2_b = deconv2d(
            h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2', with_w=True)
        h2 = tf.nn.relu(self.g_bn2(h2))

        h3, self.h3_w, self.h3_b = deconv2d(
            h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3', with_w=True)
        h3 = tf.nn.relu(self.g_bn3(h3))

        h4, self.h4_w, self.h4_b = deconv2d(
            h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

        return tf.nn.tanh(h4)
      else:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_h4 = int(s_h/2), int(s_h/4)
        s_w2, s_w4 = int(s_w/2), int(s_w/4)

        # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
        yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim]) #(64, 1, 1, 10)
        z = concat([z, y], 1) #(64, 110)

        h0 = tf.nn.relu(
            self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin'))) #(64, 1024)
        h0 = concat([h0, y], 1) #(64, 1034)

        h1 = tf.nn.relu(self.g_bn1(
            linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin')))
        h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2]) #(64, 7, 7, 128)

        h1 = conv_cond_concat(h1, yb) #(64, 7,7, 138)

        h2 = tf.nn.relu(self.g_bn2(deconv2d(h1,
            [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2')))

        h2 = conv_cond_concat(h2, yb)

        return tf.nn.sigmoid(
            deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

  def sampler(self, z, y=None, with_act=False):
    with tf.variable_scope("generator") as scope:
      scope.reuse_variables()

      if not self.y_dim:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        # project `z` and reshape
        h0 = tf.reshape(
            linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin'),
            [-1, s_h16, s_w16, self.gf_dim * 8])
        h0r = tf.nn.relu(self.g_bn0(h0, train=False))

        h1 = deconv2d(h0r, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1')
        h1r = tf.nn.relu(self.g_bn1(h1, train=False))

        h2 = deconv2d(h1r, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2')
        h2r = tf.nn.relu(self.g_bn2(h2, train=False))

        h3 = deconv2d(h2r, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3')
        h3r = tf.nn.relu(self.g_bn3(h3, train=False))

        h4 = deconv2d(h3r, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4')
        h4r = tf.nn.tanh(h4)
        #JP: here, we can also return h0, h1, h2, h3 (activation map) for visualization
        if with_act:
          return h4r, z, h0, h0r, h1, h1r, h2, h2r, h3, h3r, h4
        else:
          return h4r
      else:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_h4 = int(s_h/2), int(s_h/4)
        s_w2, s_w4 = int(s_w/2), int(s_w/4)

        # yb = tf.reshape(y, [-1, 1, 1, self.y_dim])
        yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
        z = concat([z, y], 1)

        h0_ = linear(z, self.gfc_dim, 'g_h0_lin')
        h0r = tf.nn.relu(self.g_bn0(h0_, train=False))
        h0 = concat([h0r, y], 1)

        h1 = linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin');
        h1_ = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
        h1 = tf.nn.relu(self.g_bn1(h1, train=False))
        h1r = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
        h1 = conv_cond_concat(h1r, yb)

        h2_ = deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2')
        h2r = tf.nn.relu(self.g_bn2(h2_, train=False))
        h2 = conv_cond_concat(h2r, yb)

        h3_ = deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3')
        h3 = tf.nn.sigmoid(h3_)

        if with_act:
          return h3, h0_, h0r, h1_, h1r, h2_, h2r, h3_
        else:
          return h3

  def load_mnist(self):
    data_dir = os.path.join("./data", self.dataset_name)
    
    fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.float)

    trY = np.asarray(trY)
    teY = np.asarray(teY)
    
    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)
    
    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)
    
    y_vec = np.zeros((len(y), self.y_dim), dtype=np.float)
    for i, label in enumerate(y):
      y_vec[i,y[i]] = 1.0
    
    return X/255.,y_vec


  def load_mnist_w_digit(self, digit):
    data_dir = os.path.join("./data", self.dataset_name)
    
    fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.float)

    trY = np.asarray(trY)
    teY = np.asarray(teY)
    
    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)
    
    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)
    
    y_vec = np.zeros((len(y), self.y_dim), dtype=np.float)
    for i, label in enumerate(y):
      y_vec[i,y[i]] = 1.0
    

    sel = []
    for i in xrange(len(y)):
      if y[i]==digit:
        sel.append(True)
      else:
        sel.append(False)

    XD = X[sel];
    XD = XD[0:6400];
    yD_vec = y_vec[sel];
    yD_vec = yD_vec[0:6400];
  
    return XD/255.,yD_vec

  @property
  def model_dir(self):
    return "{}_{}_{}_{}".format(
        self.dataset_name, self.batch_size,
        self.output_height, self.output_width)
      
  def save(self, checkpoint_dir, step):
    model_name = "DCGAN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

  def load(self, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0
