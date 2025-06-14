from ast_core.nn_models.mlp import mlp

inputs = 1
layer_sizes = (100,100)

ds = mlp(inputs,
         layer_sizes)