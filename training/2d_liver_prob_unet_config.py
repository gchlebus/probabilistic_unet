# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# **InsertLicense** code

# Config for 2D liver ct segmentation

import os

config_path = os.path.realpath(__file__)

data_server = "localhost"
data_port = 11100
input_stream_name = "image"
labels_stream_name = "liver_mask"


num_classes = 2
batch_size = 10
patch_size = [512, 512]
n_train_batches = None
n_val_batches = 500 // batch_size

data_format = 'NCHW'
one_hot_labels = False

#########################################
#          network & training			#
#########################################

cuda_visible_devices = '0'
cpu_device = '/cpu:0'
gpu_device = '/gpu:0'

network_input_shape = (None, 1) + tuple(patch_size)
network_output_shape = (None, num_classes) + tuple(patch_size)
label_shape = (None, 1) + tuple(patch_size)
loss_mask_shape = label_shape

base_channels = 32
num_channels = [base_channels, 2*base_channels, 4*base_channels,
				6*base_channels, 6*base_channels, 6*base_channels, 6*base_channels]

num_convs_per_block = 3

n_training_batches = 240000
validation = {'n_batches': n_val_batches, 'every_n_batches': 2000}

learning_rate_schedule = 'piecewise_constant'
learning_rate_kwargs = {'values': [1e-4, 0.5e-4, 1e-5, 0.5e-6],
						'boundaries': [80000, 160000, 240000],
						'name': 'piecewise_constant_lr_decay'}
initial_learning_rate = learning_rate_kwargs['values'][0]

regularizarion_weight = 1e-5
latent_dim = 6
num_1x1_convs = 3
beta = 1.0
analytic_kl = True
use_posterior_mean = False
save_every_n_steps = n_training_batches // 3 if n_training_batches >= 100000 else n_training_batches
disable_progress_bar = False

exp_dir = "EXPERIMENT_OUTPUT_DIRECTORY_ABSOLUTE_PATH"
