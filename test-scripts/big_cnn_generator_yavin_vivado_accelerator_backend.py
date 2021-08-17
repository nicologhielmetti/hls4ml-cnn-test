import os
from tensorflow_model_optimization.python.core.sparsity.keras import prune, pruning_callbacks, pruning_schedule
from tensorflow_model_optimization.sparsity.keras import strip_pruning
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from qkeras import QConv2D, QDense, Clip, QActivation
import numpy as np

from hls4ml.converters import convert_from_keras_model

os.environ['PATH'] = '/opt/Xilinx/Vivado/2020.1/bin:' + os.environ['PATH']

# WARNING: Invalid ReuseFactor=1 with "Resource" strategy in layer "conv_0". Using ReuseFactor=3 instead. Valid ReuseFactor(s): 3,9,27,54,108,216,432.
# WARNING: Invalid ReuseFactor=1 with "Resource" strategy in layer "conv_1". Using ReuseFactor=2 instead. Valid ReuseFactor(s): 2,3,4,6,8,9,12,16,18,24,36,48,72,144,288,576,1152,2304.
# WARNING: Invalid ReuseFactor=1 with "Resource" strategy in layer "conv_2". Using ReuseFactor=2 instead. Valid ReuseFactor(s): 2,3,4,6,8,9,12,16,18,24,36,48,72,144,288,432,576,864,1152,1728,3456.
# WARNING: Invalid ReuseFactor=1 with "Resource" strategy in layer "dense_0". Using ReuseFactor=2 instead. Valid ReuseFactor(s): 2,3,4,6,8,12,16,24,32,48,96,192,288,576,672,1344,2016,4032.
# WARNING: Invalid ReuseFactor=1 with "Resource" strategy in layer "dense_1". Using ReuseFactor=2 instead. Valid ReuseFactor(s): 2,3,6,7,14,21,42,84,168,336,672,1344,2688.
# WARNING: Invalid ReuseFactor=1 with "Resource" strategy in layer "output_dense". Using ReuseFactor=2 instead. Valid ReuseFactor(s): 2,4,8,16,32,64,128,320,640.

from tensorflow.keras.models import load_model
model = load_model('big_cnn.h5', custom_objects={'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude,
                                                 'QDense': QDense, 'QConv2D': QConv2D, 'Clip': Clip,
                                                 'QActivation': QActivation})

# model.summary()
from tensorflow_model_optimization.sparsity.keras import strip_pruning
model = strip_pruning(model)
model.summary()

X = np.load('x_big_cnn_test.npy')
y = np.load('y_big_cnn_test.npy')

import hls4ml
hls4ml.model.optimizer.OutputRoundingSaturationMode.layers = ['Activation']
hls4ml.model.optimizer.OutputRoundingSaturationMode.rounding_mode = 'AP_RND'
hls4ml.model.optimizer.OutputRoundingSaturationMode.saturation_mode = 'AP_SAT'

hls_config = hls4ml.utils.config_from_keras_model(model, granularity='name')
hls_config['Model'] = {}
hls_config['Model']['ReuseFactor'] = 1024
hls_config['Model']['Strategy'] = 'Resource'
hls_config['Model']['Precision'] = 'ap_fixed<16,6>'
hls_config['Model']['ConvImplementation'] = 'Encoded'

hls_config['LayerName']['conv_0']['ReuseFactor'] = 9
hls_config['LayerName']['conv_1']['ReuseFactor'] = 36
hls_config['LayerName']['conv_2']['ReuseFactor'] = 36
hls_config['LayerName']['dense_0']['ReuseFactor'] = 672
hls_config['LayerName']['dense_1']['ReuseFactor'] = 672
hls_config['LayerName']['output_dense']['ReuseFactor'] = 320

output_dir = 'vivado-test-cosim-encoded'
input_data = os.path.join(os.getcwd(), 'x_big_cnn_test.npy')
output_predictions = os.path.join(os.getcwd(), 'y_big_cnn_test.npy')
hls_model = convert_from_keras_model(model=model, output_dir=output_dir, board='zcu102', clock_period=10,
                                     input_data_tb=input_data, output_data_tb=output_predictions,
                                     backend='Vivado', io_type='io_stream', hls_config=hls_config)

hls_model.compile()
y_hls = hls_model.predict(np.ascontiguousarray(X))
np.save('y_big_cnn_hls_vivado.npy', y_hls)
hls_model.build(csim=True, cosim=True, validation=True, synth=True, vsynth=False, export=False)
hls4ml.report.read_vivado_report(output_dir)

# hls4ml.templates.VivadoAcceleratorBackend.make_bitfile(hls_model)
