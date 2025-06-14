# DISABLE TF WARNING WHEN DEBUGGING
debug = 1
if debug:
    import os
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # hides INFO + WARNING
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Hide all messages except errors

from ast_core.nn_models.mlp import mlp

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

## REGARDING INPUT TO MLP
# Input must be:
# - a single tf.Tensor (e.g., tf.placeholder(...))
# - or a list of tf.Tensor s

inputs = [tf.placeholder(tf.float32, shape=(None, 4))] # Example input with 4 features, made as vector

layer_sizes = (100,100)

ds = mlp(inputs,
         layer_sizes)