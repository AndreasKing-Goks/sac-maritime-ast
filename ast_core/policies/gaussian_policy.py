'''
Gaussian Policy
'''

from contextlib import contextmanager
import numpy as np
import tensorflow as tf

from rllab.misc.overrides import overrides
from rllab.misc import logger
from rllab.core.serializable import Serializable

from ast_core.distributions import Normal
from ast_core.policies.nn_policy import NNPolicy
from ast_core.utils import tf_utils

EPS = 1e-6

class GaussianPolicy(NNPolicy, Serializable):
    def __init__(self, 
                 env_spec, 
                 hidden_layer_sizes=(100,100),
                 reg=1e-3,
                 squash=True,
                 reparameterize=True,
                 name='gaussian_policy'):
        '''
        Args:
        - env_spec (`rllab.EnvSpec`):
            Specification of the environment to create the policy for.
        - hidden_layer_sizes (`list` of `init`):
            Sizes for the Multi Layer Perceptron's hidden layer.
        - reg (`float`):
            Regularization coefficient for the Gaussian parameters.
        - squash (`bool`):
            If True, squash the Gaussian action samples between -1 and 1 with the tanh function
        - reparameterize (`bool`):
            If True, gradients will flow directly through the action samples.
            
        NOTE:
        - Reparameterize is used because gaussian policy is a stochastic function, hence not having gradients.
          Gradients is used to optimize the loss function. Reparameterize basically representing a policy as
          a action generator function with gradients.
        '''
        Serializable.quick_init(self, locals())
        
        self._hidden_layers = hidden_layer_sizes
        self._Da = env_spec.action_space.flat_dim
        self._Ds = env_spec.observation_space.flat_dim
        self._is_deterministic = False
        self._fixed_h = None
        self._squash = squash
        self._reparameterize = reparameterize
        self._reg = reg
        
        self._name = name
        self.build()
        
        self._scope_name = (
            tf.get_variable_scope().name + "/" + name
        ).lstrip("/")
        
        super(NNPolicy, self).__init__(env_spec)
        
    def actions_for(self, observation, latents=None,
                    name=None, reuse=tf.AUTO_REUSE,
                    with_log_pis=False, reguralize=False):
        name = name or self._name
        
        with tf.variable_scope(name, reuse=reuse):
            distribution = Normal(
                hidden_layers_sizes=self._hidden_layers,
                Dx=self._Da,
                reparameterize=self._reparameterize,
                cond_t_lst=(observation,),
                reg=self._reg
            )
        raw_actions = distribution.x_t
        actions = tf.tanh(raw_actions) if self._squash else raw_actions
        
        # TODO: should always return same shape out
        # Figure out how to make the interface for `log_pis` cleaner
        if with_log_pis:
            # TODO.code_consolidation: should come from log_pis_for
            log_pis = distribution.log_pi_t
            if self._squash:
                log_pis -= self._squash_correction(raw_actions)
            return actions, log_pis
        
        return actions
    
    def log_pis_for(self, actions):
        if self._squash:
            raw_actions = tf.atanh(actions)
            log_pis = self._distribution.log_prob(raw_actions)
            log_pis -= self._squash_correction(raw_actions)
            return log_pis
        return self._distribution.log_prob(raw_actions)
    
    def build(self):
        self._observations_ph = tf.placeholder(
            dtype=tf.float32,
            shape=(None, self._Ds),
            name='observations'
        )
        
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.distribution = Normal(
                hidden_layers_sizes=self._hidden_layers,
                Dx=self._Da,
                reparameterize=self._reparameterize,
                cond_t_lst=(self._observations_ph,),
                reg=self._reg
            )
        
        raw_actions = tf.stop_gradient(self.distribution.x_t)
        self._actions = tf.tanh(raw_actions) if self._squash else raw_actions
    
    @overrides
    def get_actions(self, observations):
        '''
        Sample actions based on the observations.
        
        If `self._is_determenistic` is Truem returns the mean action for the
        observations. If False, return stochastically sampled action.
        
        TODO.code_consolidation: This should be somewhat similar with
        `LatenSpacePolicy.get_actions()`
        '''
        if self._is_deterministic: # Handle the deterministic case separately
            feed_dict = {self._observations_ph: observations}
            
            # TODO.code_consolidation: these shapes should be double checked
            # for case where `observation.shape[0] > 1`
            mu = tf.get_default_session().run(
                self.distribution.mu_t, feed_dict   # 1 x Da
            )
            if self._squash:
                mu = np.tanh(mu)
            
            return mu
        
        return super(GaussianPolicy, self).get_actions(observations)
    
    def _squash_correction(self, actions):
        if not self._squash:
            return 0
        return tf.reduce_sum(tf.log(1 - tf.tanh(actions) ** 2 + EPS), axis=1)
    
    @contextmanager
    def deterministic(self, set_deterministic=True, latent=None):
        '''
        Context manager for changing the deterministic characteristic of the policy.
        
        See `self.get_action` for further information about the effect of
        self._is_deterministic.
        
        Args:
        - set_deterministic (`bool`): 
            Value to set the self._is_deterministic to during the context. The value
            will be reset back to the previous value when the context exists.
        - latent (`Number`):
            Value to set the latent variable to over the deterministic context.
        '''
        was_deterministic = self._is_deterministic
        
        self._is_determenistic = set_deterministic
        
        yield
        
        self._is_determenistic = was_deterministic
    
    def log_diagnostics(self, iteration, batch):
        '''
        Records diagnostic information to the logger.
        
        Records the mean, min, max, and the standard deviation of the GMM
        means, component weights, and covariances.
        '''
        
        feeds = {self._observations_ph: batch['observations']}
        sess = tf_utils.get_default_session()
        mu, log_sig, log_pi = sess.run(
            (
            self.distribution.mu_t,
            self.distribution.log_sig_t,
            self.distribution.log_pi_t
            ),
            feeds
        )
        
        logger.record_tabular('policy-mus-mean', np.mean(mu))
        logger.record_tabular('policy-mus-min', np.min(mu))
        logger.record_tabular('policy-mus-max', np.max(mu))
        logger.record_tabular('policy-mus-std', np.std(mu))
        logger.record_tabular('log-sigs-mean', np.mean(log_sig))
        logger.record_tabular('log-sigs-min', np.min(log_sig))
        logger.record_tabular('log-sigs-max', np.max(log_sig))
        logger.record_tabular('log-sigs-std', np.std(log_sig))
        logger.record_tabular('log-pi-mean', np.mean(log_pi))
        logger.record_tabular('log-pi-max', np.max(log_pi))
        logger.record_tabular('log-pi-min', np.min(log_pi))
        logger.record_tabular('log-pi-std', np.std(log_pi))
        