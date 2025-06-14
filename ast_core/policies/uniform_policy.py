from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides
from sandbox.rocky.tf.policies.base import Policy

import numpy as np

class UniformPolicy(Policy, Serializable):
    """
    A fixed policy that randomly samples actions uniformly at random.
    
    This policy is used during initial exploration to collect initial RL states for the replay pool.
    """
    def __init__(self, env_spec):
        Serializable.quick_init(self, locals())
        self._Da = env_spec.action_space.flat_dim
        
        super(UniformPolicy, self).__init__(env_spec)
        
    # Assuming that action sapces are normalized to be the interval of [-1, 1]
    @overrides
    def get_action(self, observation):
        return np.random.uniform(-1., 1., self._Da), None
    
    @overrides
    def get_actions(self, observation):
        pass
    
    @overrides
    def log_diagnostics(self, paths):
        pass
    
    @overrides
    def get_params_internal(self, **tags):
        pass