import os

# Turning off oneDNN custom operations
os.environ["TF_ENABLE_ONEDNN_OPTS"]= "0"

import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

for gpu in tf.config.list_physical_devices('GPU'):
    print("GPU:", gpu)
    
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())