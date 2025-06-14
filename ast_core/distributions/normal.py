"""
Multivariate normal distribution with mean and standard deviation outputted by a neural network
"""

import tensorflow as tf
import numpy as np

from ast_core.nn_models.mlp import mlp

# Need to figure out what is this
LOG_SIG_CAP_MAX = 2
LOG_SIG_CAP_MIN = -20

class Normal(object):
    def __init__(
        self,
        Dx,
        hidden_layers_sizes=(100, 100),
        reg=0.001,
        reparameterize=True,
        cond_t_lst=()
    ):
        self._cond_t_lst = cond_t_lst
        self._reg = reg # Regularization coefficient
        self._layer_sizes = list(hidden_layers_sizes) + [2 * Dx]
        # print(self._layer_sizes)
        self._reparameterize = reparameterize
        
        self._Dx = Dx
        
        self._create_placeholders()
        self._create_graph()
        
    def _create_placeholders(self):
        self._N_pl = tf.placeholder(
            tf.int32,
            shape=(),
            name='N'
        )
        
    def _create_graph(self):
        '''
        This method handles the normal distribution function.
        
        Accessing the distribution attributes is done using the property classes:
        - mu_t()        : Get the mean of the Gaussian
        - log_si_t()    : Get the log standard deviation
        - log_pi_t()    : Get the log likelihood of the sample
        - reg_loss_t()  : Get the regularization loss (L2) on the mean "μ" and the log standard deviation "log(σ)"
        - x_t()         : Get the sample from N(μ,σ**2) 
        '''
        ## Dimensionality of the distribution (following the action spaces)
        Dx = self._Dx
        
        ## Constructing the output tensor
        #  If no conditional input is provided, use static trainable parameter tensor
        if len(self._cond_t_lst) == 0:
            mu_and_logsig_t = tf.get_variable(
                'params', self._layer_sizes[-1],
                initializer=tf.random_normal_initializer(0, 0.1)
            )
        # Else it passes through a Multi-Layer Perception with architecture: 
        # hidden_layer-sizes + [2 * Dx]
        # Output size is [batch_size, 2 * Dx]:
        # - First Dx is the mean μ
        # - Second Dx is the log standard deviation log(σ)
        else:
            mu_and_logsig_t = mlp(
                inputs=self._cond_t_lst,
                layer_sizes=self._layer_sizes,
                output_nonlinearity=None
                # ... x K*Dx*2+K
            )
        
        ## Split the mean and log standard deviation
        self._mu_t = mu_and_logsig_t[..., :Dx]
        self._log_sig_t = tf.clip_by_value(mu_and_logsig_t[..., Dx:], LOG_SIG_CAP_MIN, LOG_SIG_CAP_MAX)
        
        ## Construct the normal distribution
        # Tensorflow's multivariate normal distribution supports reparameterization
        ds = tf.contrib.distributions
        dist = ds.MultivariateNormalDiag(loc=self._mu_t, scale_diag=tf.exp(self._log_sig_t))
        # Sample from the distribution
        x_t = dist.sample()
        if not self._reparameterize:
            x_t = tf.stop_gradient(x_t)
        # Compute the log-likelihood of the sample
        log_pi_t = dist.log_prob(x_t)
        
        # Store the outputs and the regularization
        self._dist = dist
        self._x_t = x_t
        self._log_pi_t = log_pi_t
        
        ## Then define the regularization loss
        # First compute the mean of the square of each component in μ
        # Then multiply it to the mean of the square to log(σ)
        # This whole operation is an L2 regularization process, 
        # which is adding penalization of squared values of the outputs (μ,log(σ))
        reg_loss_t = self._reg * 0.5 * tf.reduce_mean(self._log_sig_t ** 2)
        reg_loss_t += self._reg * 0.5 * tf.reduce_mean(self._mu_t ** 2)
        self._reg_loss_t = reg_loss_t
    
    
    # Access attribute using property-tagged method
    @property
    def mu_t(self):
        return self._mu_t
    
    @property
    def log_sig_t(self):
        return self._log_sig_t
    @property
    def log_pi_t(self):
        return self._log_pi_t
    
    @property
    def reg_loss_t(self):
        return self._reg_loss_t
    
    @property
    def x_t(self):
        return self._x_t