# Disable VS Code IntelliSense Warning in settings.json
# Enable Tensorflow 1.x, in Tensorflow 2.x runtime
import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()

from rllab.core.serializable import Serializable
from sandbox.rocky.tf.core.parameterized import Parameterized
from ast_core.utils import tf_utils

WEIGHT_DEFAULT_NAME = "weights"
BIAS_DEFAULT_NAME = "bias"

def _weight_variable(
    shape,
    initializer=None,
    name=WEIGHT_DEFAULT_NAME
):
    """
    Returns a variable with a given shape.
    
    -param initializer  : Tensorflow initializer. Default initializer is Xavier.
    -param name         : Variable name.
    -param shape        : Variable shape.
    """
    if initializer is None:
        # initializer = tf.contrib.layes.xavier_initializer() # Deprecated version
        initializer = tf.keras.initializers.GlorotUniform()
        
    var = tf.get_variable(name, shape, initializer=initializer)
    return var

def _bias_variable(
    shape,
    initializer=None,
    name=BIAS_DEFAULT_NAME
):
    """
    Returns a bias variable with a given shape.
    
    -param intializer   : Tensorflow initializer. Default is zero.
    -param name         : Variable name.
    -param shape        : Variable shape.
    """
    if initializer is None:
        initializer = tf.constant_initializer(0.)
    
    return _weight_variable(
        shape, 
        initializer=initializer,
        name=name
    )
    
def affine(
    inp,
    units,
    bias=True,
    W_initializer=None,
    b_initializer=None,
    W_name=WEIGHT_DEFAULT_NAME,
    bias_name=BIAS_DEFAULT_NAME
):
    """
    Creates an affine layer. Affine layer is a basic building block in neural networks.
    It performs a linear transformation of the input, optionally followed by the addition of a bias term
    
    ```
    y = Wx + b
    ```
    
    -param inp              : Input tensor.
    -param units            : Number of units.
    -param bias             : Include bias term.
    -param W_initializer    : Initializer for the multiplicative weight.
    -param b_initializer    : Initializer for the bias term.
    -param W_name           : Name of the weight.
    -param bias_name        : Name of the bias.
    
    -return                 : Tensor defind as input.dot(weight) + bias
    """
    input_size = inp.get_shape()[-1].value
    W = _weight_variable([input_size, units],
                         initializer=W_initializer,
                         name=W_name)
    
    output = tf.matmul(inp, W)
    
    if bias:
        b = _bias_variable((units,),
                           initializer=b_initializer,
                           name=bias_name)
        output += b
    
    return output

def mlp(inputs,
        layer_sizes,
        nonlinearity=tf.nn.relu,
        output_nonlinearity=tf.nn.tanh,
        W_initializer=None,
        b_initializer=None):
    """
    Creates a multi-layer perceptron with given hidden sizes. A nonlinearity is applied after every hidden layer.
    
    Supports input tensors of rank 2 and rank 3. All inputs should have the same tensor rank. It is assumed that
    the vectors along the last axis are the data points, and an mlp is applied independently to each leading 
    dimension. If multiple inputs are provided, then the corresponding rank 1 vectors are concatenated along the 
    last axis. The leading dimensions of the network output are equal to the 'outer product' of the inputs' 
    shapes.
    """
    if type(inputs) is tf.Tensor:
        inputs = [inputs]
        
    squeeze_output = False
    if layer_sizes[-1] is None:
        squeeze_output = True
        layer_sizes = list(layer_sizes)
        layer_sizes[-1] = 1
    
    # Take care of the input layer separately to make use of broadcasting in
    # a case of several input tensors.
    with tf.variable_scope('layer0'):
        layer = _bias_variable(layer_sizes[0], b_initializer)
        for i, inp in enumerate(inputs):
            with tf.variable_scope('input' + str(i)):
                layer = layer + affine(inp=inp, 
                                       units=layer_sizes[0], 
                                       bias=False, 
                                       W_initializer=W_initializer, 
                                       b_initializer=b_initializer
                                       )
    
        layer = nonlinearity(layer)
        
    for i_layer, size in enumerate(layer_sizes[1:], 1):
        with tf.variable_scope('layer{0}'.format(i_layer)):
            layer = affine(layer, size,
                           W_initializer=W_initializer,
                           b_initializer=b_initializer)
            if i_layer < len(layer_sizes) - 1:
                layer = nonlinearity(layer)
    
    if output_nonlinearity is not None:
        layer = output_nonlinearity(layer)
    
    if squeeze_output:
        layer = tf.squeeze(layer, axis=-1)
    
    return layer

class MLPFunction(Parameterized, Serializable):
    
    def __init__(self, name, input_pls, hidden_layer_sizes, output_nonlinearity=None):
        Parameterized.__init__(self)
        Serializable.quick_init(self, locals())
        
        self._name = name
        self._input_pls = input_pls
        self._layer_sizes = list(hidden_layer_sizes) + [None]
        self._output_nonlinearity = output_nonlinearity
        
        self._output_t = self.get_output_for(*self._input_pls)
        
    def get_output_for(self, *inputs, reuse=False):
        with tf.variable_scope(self._name, reuse=reuse):
            value_t = mlp(
                inputs=inputs,
                output_nonlinearity=self._output_nonlinearity,
                layer_sizes=self._layer_sizes
            )
        return value_t
    
    def eval(self, *inputs):
        feeds = {pl: val for pl, val in zip(self._input_pl, inputs)}
        
        return tf_utils.get_default_session().run(self._output_t, feeds)
    
    def get_params_internal(self, **tags):
        if len(tags) > 0:
            raise NotImplementedError
        
        scope = tf.get_variable_scope().name
        scope += '/' + self._name + '/' if len(scope) else self._name + '/'
        
        return tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope
        )