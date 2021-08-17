import os

from qkeras import quantized_bits, QConv2D
from tensorflow.python.keras import Sequential, Input
from scipy.io import loadmat
import numpy as np
from qkeras import to_categorical
import random as rnd
import tensorflow as tf

import hls4ml
from hls4ml.converters import convert_from_keras_model

rnd.seed(42)


def dummy_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(
        QConv2D(16, (3, 3),
                kernel_initializer=tf.keras.initializers.RandomUniform(minval=-1, maxval=1, seed=96),
                bias_initializer=tf.keras.initializers.RandomUniform(minval=-1, maxval=1, seed=96),
                kernel_quantizer=quantized_bits(6, 0, alpha=1),
                bias_quantizer=quantized_bits(6, 0, alpha=1),
                name="qconv2d", input_shape=input_shape)
    )

    model.build(input_shape=input_shape)

    return model


def rgb2gray(images):
    return np.expand_dims(np.dot(images, [0.2990, 0.5870, 0.1140]), axis=3)


# train = loadmat('svhndataset/train_32x32.mat')
# test = loadmat('svhndataset/test_32x32.mat')
# train_img = np.array(train['X'])
# test_img = np.array(test['X'])
# train_label = train['y']
# test_label = test['y']
# train_img = np.moveaxis(train_img, -1, 0)
# test_img = np.moveaxis(test_img, -1, 0)
# train_label[train_label == 10] = 0
# test_label[test_label == 10] = 0
# X_train = rgb2gray(train_img).astype(np.float32)
# X_test = rgb2gray(test_img).astype(np.float32)
# X_train = X_train / 255.0
# X_test = X_test / 255.0
# train_label = to_categorical(train_label)
# test_label = to_categorical(test_label)

# np.save('svhndataset/test_label_first_100.npy', train_label[:100])
# np.save('svhndataset/test_32x32_first_100.npy', X_test[:100])

X = np.load('svhndataset/test_32x32_first_100.npy')
y = np.load('svhndataset/test_label_first_100.npy')

height = 32
width = 32
chan = 1
input_shape = (height, width, chan)

model = dummy_model(input_shape)
hls4ml.model.optimizer.OutputRoundingSaturationMode.layers = ['Activation']
hls4ml.model.optimizer.OutputRoundingSaturationMode.rounding_mode = 'AP_RND'
hls4ml.model.optimizer.OutputRoundingSaturationMode.saturation_mode = 'AP_SAT'

output_dir = 'vivado-accelerator-test-cosim-dummy-lb'
input_data = os.path.join(os.getcwd(), 'svhndataset/test_32x32_first_100.npy')
output_predictions = os.path.join(os.getcwd(), 'svhndataset/test_label_first_100.npy')
hls_config = hls4ml.utils.config_from_keras_model(model, granularity='name')
hls_config['Model'] = {}
hls_config['Model']['ReuseFactor'] = 1
hls_config['Model']['Strategy'] = 'Resource'
hls_config['Model']['Precision'] = 'ap_fixed<16,6>'
hls_config['Model']['ConvImplementation'] = 'Linebuffer'
hls_model = convert_from_keras_model(model=model, output_dir=output_dir, board='zcu102', clock_period=10,
                                     input_data_tb=input_data, output_data_tb=output_predictions,
                                     backend='VivadoAccelerator', io_type='io_stream', hls_config=hls_config)

hls_model.compile()
# y_dummy_cnn = hls_model.predict(np.ascontiguousarray(X))

# np.save('y_dummy_cnn_vivado_accelerator.npy', y_dummy_cnn)

# hls_model.build(csim=True, cosim=True, validation=True, synth=True, vsynth=False, export=False)
# hls4ml.report.read_vivado_report(output_dir)
