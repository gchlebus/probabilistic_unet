# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# **InsertLicense** code

__author__ = 'gchlebus'

import sys
import argparse
import os
import tensorflow as tf
import numpy as np
from dnn.remote import minus_one_rpc
from importlib.machinery import SourceFileLoader
from model.probabilistic_unet import ProbUNet
from utils import training_utils

CONFIG_FILENAME = "used_config.py"


class InferenceServer(object):
  def __init__(self):
    self._server = None
    self._punet = None
    self._session = None
    self._x = None  # placeholder for input image
    self._y = None  # placeholder for reference

  def set_server(self, minus_one_server):
    self._server = minus_one_server

  def load(self, experiment_dir):
    config_filename = os.path.join(experiment_dir, CONFIG_FILENAME)
    cf = SourceFileLoader('cf', config_filename).load_module()
    self._punet = ProbUNet(latent_dim=cf.latent_dim, num_channels=cf.num_channels,
                           num_1x1_convs=cf.num_1x1_convs,
                           num_classes=cf.num_classes, num_convs_per_block=cf.num_convs_per_block,
                           initializers={'w': training_utils.he_normal(),
                                         'b': tf.truncated_normal_initializer(stddev=0.001)},
                           regularizers={'w': tf.contrib.layers.l2_regularizer(1.0),
                                         'b': tf.contrib.layers.l2_regularizer(1.0)})
    self._x = tf.placeholder(tf.float32, shape=cf.network_input_shape)
    self._y = tf.placeholder(tf.uint8, shape=cf.label_shape)
    # is_training=True to get posterior_net as well
    self._punet(self._x, self._y, is_training=True, one_hot_labels=cf.one_hot_labels)
    saver = tf.train.Saver(save_relative_paths=True)
    self._session = tf.train.MonitoredTrainingSession()
    print("Experiment dir:", experiment_dir)
    latest_ckpt_path = tf.train.latest_checkpoint(experiment_dir)
    print("Loading model from:", latest_ckpt_path)
    saver.restore(self._session, latest_ckpt_path)

  def inference(self, input_img, ref_img, use_posterior_net=False, deterministic=False, external_sample=None):
    # output segmentation mask
    # latent_space
    # sample
    input_img = input_img[np.newaxis]
    print("input_img:", input_img.shape, input_img.dtype)
    if use_posterior_net:
      ref_img = ref_img[np.newaxis]
      print("ref_img:", ref_img.shape, ref_img.dtype)

    if use_posterior_net:
      latent_mu_op = self._punet._q.mean()
      latent_stddev_op = self._punet._q.stddev()
      latent_sample_op = self._punet._q.sample()
    else:
      latent_mu_op = self._punet._p.mean()
      latent_stddev_op = self._punet._p.stddev()
      latent_sample_op = self._punet._p.sample()

    if deterministic:
      out_seg_op = self._punet.reconstruct(z_q=latent_mu_op)
    else:
      out_seg_op = self._punet.reconstruct(z_q=latent_sample_op)

    feed_dict = {
      self._x: input_img
    }
    if use_posterior_net:
      feed_dict[self._y] = ref_img

    out_seg, latent_sample, latent_mu, latent_stddev = self._session.run([
      out_seg_op, latent_sample_op, latent_mu_op, latent_stddev_op
    ], feed_dict=feed_dict)
    print("out_seg", out_seg.shape)
    print("latent_sample", latent_sample)
    print("latent_mu", latent_mu)
    print("latent_stddev", latent_stddev)
    return out_seg, latent_sample, latent_mu, latent_stddev


def run(argv):
  print("classification_server.py @ run(" + str(argv) + ")")
  parser = argparse.ArgumentParser(description="Start probabilistic u-net inference server")
  parser.add_argument('--interface', '-i', default="*",
                      help='network interface to start listening on (default: %(default)s)')
  parser.add_argument('--port', '-p', type=int, default=4242,
                      help='fixed TCP port to listen on (default: %(default)d, must not be in use)')
  parser.add_argument('exp_dir', help="Experiment directory with saved model")
  args = parser.parse_args(argv)

  if not os.path.isdir(args.exp_dir):
    print("Experiment directory (%s) does not exist" % args.exp_dir)
    exit(1)

  inf_server = InferenceServer()
  inf_server.load(args.exp_dir)
  s = minus_one_rpc.Server(inf_server)
  inf_server.set_server(s)
  target = 'tcp://%s:%d' % (args.interface, args.port)
  s.bind(target)
  print('Successfully started server at %s' % target)
  sys.stdout.flush()
  s.run()
  s.close()
  exit(0)


if __name__ == '__main__':
  run(sys.argv[1:])
