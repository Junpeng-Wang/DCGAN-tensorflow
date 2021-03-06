"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import pprint
import os
import scipy.misc
import numpy as np
from time import gmtime, strftime
from six.moves import xrange

import tensorflow as tf
import tensorflow.contrib.slim as slim

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

# data: json object array
def append_layer(data, name, size, lrdt):
  layer = {}
  layer['type'] = name
  layer['shape'] = size
  layer['data'] = lrdt.flatten().tolist()
  data.append(layer)

def show_all_variables():
  model_vars = tf.trainable_variables()
  slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def get_image(image_path, input_height, input_width,
              resize_height=64, resize_width=64,
              crop=True, grayscale=False):
  image = imread(image_path, grayscale)
  return transform(image, input_height, input_width,
                   resize_height, resize_width, crop)

def save_images(images, size, image_path):
  return imsave(inverse_transform(images), size, image_path)

def imread(path, grayscale = False):
  if (grayscale):
    return scipy.misc.imread(path, flatten = True).astype(np.float)
  else:
    return scipy.misc.imread(path).astype(np.float)

def merge_images(images, size):
  return inverse_transform(images)

def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  if (images.shape[3] in (3,4)):
    c = images.shape[3]
    img = np.zeros((h * size[0], w * size[1], c))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w, :] = image
    return img
  elif images.shape[3]==1:
    img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
    return img
  else:
    raise ValueError('in merge(images,size) images parameter '
                     'must have dimensions: HxW or HxWx3 or HxWx4')

def imsave(images, size, path):
  image = np.squeeze(merge(images, size))
  return scipy.misc.imsave(path, image)

def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
  if crop_w is None:
    crop_w = crop_h
  h, w = x.shape[:2]
  j = int(round((h - crop_h)/2.))
  i = int(round((w - crop_w)/2.))
  return scipy.misc.imresize(
      x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform(image, input_height, input_width, 
              resize_height=64, resize_width=64, crop=True):
  if crop:
    cropped_image = center_crop(
      image, input_height, input_width, 
      resize_height, resize_width)
  else:
    cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
  return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
  return (images+1.)/2.

def to_json(output_path, *layers):
  with open(output_path, "w") as layer_f:
    lines = ""
    for w, b, bn in layers:
      layer_idx = w.name.split('/')[0].split('h')[1]

      B = b.eval()

      if "lin/" in w.name:
        W = w.eval()
        depth = W.shape[1]
      else:
        W = np.rollaxis(w.eval(), 2, 0)
        depth = W.shape[0]

      biases = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(B)]}
      if bn != None:
        gamma = bn.gamma.eval()
        beta = bn.beta.eval()

        gamma = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(gamma)]}
        beta = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(beta)]}
      else:
        gamma = {"sy": 1, "sx": 1, "depth": 0, "w": []}
        beta = {"sy": 1, "sx": 1, "depth": 0, "w": []}

      if "lin/" in w.name:
        fs = []
        for w in W.T:
          fs.append({"sy": 1, "sx": 1, "depth": W.shape[0], "w": ['%.2f' % elem for elem in list(w)]})

        lines += """
          var layer_%s = {
            "layer_type": "fc", 
            "sy": 1, "sx": 1, 
            "out_sx": 1, "out_sy": 1,
            "stride": 1, "pad": 0,
            "out_depth": %s, "in_depth": %s,
            "biases": %s,
            "gamma": %s,
            "beta": %s,
            "filters": %s
          };""" % (layer_idx.split('_')[0], W.shape[1], W.shape[0], biases, gamma, beta, fs)
      else:
        fs = []
        for w_ in W:
          fs.append({"sy": 5, "sx": 5, "depth": W.shape[3], "w": ['%.2f' % elem for elem in list(w_.flatten())]})

        lines += """
          var layer_%s = {
            "layer_type": "deconv", 
            "sy": 5, "sx": 5,
            "out_sx": %s, "out_sy": %s,
            "stride": 2, "pad": 1,
            "out_depth": %s, "in_depth": %s,
            "biases": %s,
            "gamma": %s,
            "beta": %s,
            "filters": %s
          };""" % (layer_idx, 2**(int(layer_idx)+2), 2**(int(layer_idx)+2),
               W.shape[0], W.shape[3], biases, gamma, beta, fs)
    layer_f.write(" ".join(lines.replace("'","").split()))

def make_gif(images, fname, duration=2, true_image=False):
  import moviepy.editor as mpy

  def make_frame(t):
    try:
      x = images[int(len(images)/duration*t)]
    except:
      x = images[-1]

    if true_image:
      return x.astype(np.uint8)
    else:
      return ((x+1)/2*255).astype(np.uint8)

  clip = mpy.VideoClip(make_frame, duration=duration)
  clip.write_gif(fname, fps = len(images) / duration)

def visualize(sess, dcgan, config, option):
  image_frame_dim = int(math.ceil(config.batch_size**.5))
  if option == 0:
    z_sample = np.random.uniform(-0.5, 0.5, size=(config.batch_size, dcgan.z_dim))
    samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
    save_images(samples, [image_frame_dim, image_frame_dim], './samples/test_%s.png' % strftime("%Y%m%d%H%M%S", gmtime()))
  elif option == 1:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in xrange(100):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      if config.dataset == "mnist":
        y = np.random.choice(10, config.batch_size)
        y_one_hot = np.zeros((config.batch_size, 10))
        y_one_hot[np.arange(config.batch_size), y] = 1

        [samples, h0, h0r, h1, h1r, h2, h2r, h3] = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_one_hot})
      else:
        [samples, z, h0, h0r, h1, h1r, h2, h2r, h3, h3r, h4] = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})

      save_images(samples, [image_frame_dim, image_frame_dim], './samples/test_arange_%s.png' % (idx))
  elif option == 2:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in [random.randint(0, 99) for _ in xrange(100)]:
      print(" [*] %d" % idx)
      z = np.random.uniform(-0.2, 0.2, size=(dcgan.z_dim))
      z_sample = np.tile(z, (config.batch_size, 1))
      #z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      if config.dataset == "mnist":
        y = np.random.choice(10, config.batch_size)
        y_one_hot = np.zeros((config.batch_size, 10))
        y_one_hot[np.arange(config.batch_size), y] = 1

        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_one_hot})
      else:
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})

      try:
        make_gif(samples, './samples/test_gif_%s.gif' % (idx))
      except:
        save_images(samples, [image_frame_dim, image_frame_dim], './samples/test_%s.png' % strftime("%Y%m%d%H%M%S", gmtime()))
  elif option == 3:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in xrange(100):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
      make_gif(samples, './samples/test_gif_%s.gif' % (idx))
  elif option == 4:
    image_set = []
    values = np.arange(0, 1, 1./config.batch_size)

    for idx in xrange(100):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample): z[idx] = values[kdx]

      image_set.append(sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample}))
      make_gif(image_set[-1], './samples/test_gif_%s.gif' % (idx))

    new_image_set = [merge(np.array([images[idx] for images in image_set]), [10, 10]) \
        for idx in range(64) + range(63, -1, -1)]
    make_gif(new_image_set, './samples/test_gif_merged.gif', duration=8)
  elif option == 5: # JP: my version of dump the current state of the D and G
    directory = '{}/Test'.format(config.sample_dir)
    if not os.path.exists(directory):
      os.makedirs(directory)

    values = np.arange(0, 1, 1./config.batch_size)
    for idx in xrange(3):
      digit = 0;
      curdir = '{}/{:04d}'.format(directory, idx)
      if not os.path.exists(curdir):
        os.makedirs(curdir)

      print(" [*] %d" % idx)
      #z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      #for kdx, z in enumerate(z_sample):
      #  z[idx] = values[kdx]
      z_sample = np.random.uniform(-1, 1, size=(config.batch_size , dcgan.z_dim))

      if config.dataset == "mnist":
        #y = np.random.choice(10, config.batch_size)
        y = np.full((config.batch_size), digit, dtype=np.int)
        y_one_hot = np.zeros((config.batch_size, 10))
        y_one_hot[np.arange(config.batch_size), y] = 1

        [samples, h0, h0r, h1, h1r, h2, h2r, h3] = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_one_hot})
        [samplesD_, d_smp_, d_h0_, d_h0r_, d_h1_, d_h1r_, d_h2_, d_h2r_, d_h3_] = sess.run(dcgan.samplerD_, feed_dict={dcgan.G: samples, dcgan.y: y_one_hot})

        # save all G's
        dirG = '{}/G'.format(curdir)
        if not os.path.exists(dirG):
          os.makedirs(dirG)
        data = [];
        append_layer(data, 'linear', [64, 32, 32, 1],   h0);
        append_layer(data, 'relu',   [64, 32, 32, 1],   h0r);
        append_layer(data, 'linear', [64, 7, 7, 128],   h1);
        append_layer(data, 'relu',   [64, 7, 7, 128],   h1r);
        append_layer(data, 'deconv', [64, 14, 14, 128], h2);
        append_layer(data, 'relu',   [64, 14, 14, 128], h2r);
        append_layer(data, 'deconv', [64, 28, 28, 1],   h3);
        append_layer(data, 'sigmoid',[64, 28, 28, 1],   samples);
        with open('./{}/G_{:04d}.json'.format(dirG, idx), 'w') as f:
          json.dump(data, f)
        save_images(samples, [image_frame_dim, image_frame_dim], './{}/G_{:04d}.png'.format(dirG, idx))

        # save all D_'s
        dirD_ = '{}/D_'.format(curdir)
        if not os.path.exists(dirD_):
          os.makedirs(dirD_)
        dataD_ = [];
        append_layer(dataD_, 'input',  [64, 28, 28, 1],  d_smp_);
        append_layer(dataD_, 'conv',   [64, 14, 14, 11], d_h0_);
        append_layer(dataD_, 'relu',   [64, 14, 14, 11], d_h0r_);
        append_layer(dataD_, 'conv',   [64, 7, 7, 74],   d_h1_);
        append_layer(dataD_, 'relu',   [64, 7, 7, 74],   d_h1r_);
        append_layer(dataD_, 'linear', [64, 32, 32, 1],  d_h2_);
        append_layer(dataD_, 'relu',   [64, 32, 32, 1],  d_h2r_);
        append_layer(dataD_, 'linear', [64, 1, 1, 1],    d_h3_);
        append_layer(dataD_, 'sigmoid',[64, 1, 1, 1],    samplesD_);
        with open('./{}/F_{:04d}.json'.format(dirD_, idx), 'w') as f:
          json.dump(dataD_, f)

        bz = 64;
        scale = 20;
        result = np.zeros((bz,scale,scale,1))
        for ii in xrange(bz):
          for jj in xrange(scale):
            for kk in xrange(scale):
              if samplesD_[ii]>0.5:
                result[ii, jj, kk, 0] = 1;
              else:
                result[ii, jj, kk, 0] = 0;

        save_images(result, [image_frame_dim, image_frame_dim], './{}/F_{:04d}.png'.format(dirD_, idx))
        tn = 0; fn = 0;
        for ii in xrange(bz):
          if samplesD_[ii]>0.5:
            tn = tn+1;
          else:
            fn = fn+1;
        print tn, fn;
      else:
        [samples, z, h0, h0r, h1, h1r, h2, h2r, h3, h3r, h4] = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})


