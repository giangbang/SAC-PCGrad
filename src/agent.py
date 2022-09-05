import gym
import torch
from .sac_discrete import SACDiscrete
from .sac_continuous import SAC 
from .buffer import ReplayBuffer

class PCGradAgent(object):
    def __init__(self,
        env: gym.vector.VectorEnv, 
        learning_rate: float = 3e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int = 1,
        gradient_steps: int = 1,
        seed: int = None,
        device: str = 'cpu',
        *args, **kwargs  # other parameters
    ):
        assert isinstance(env, gym.vector.VectorEnv), (
            "Only support multitask environments"
        )

        self.env = env
        self.buffer = ReplayBuffer(env.observation_space, env.action_space, 
                buffer_size, batch_size, device)
        discrete_action = isinstance(env.action_space, gym.spaces.Discrete)

        if discrete_action:
            self.agent = SACDiscrete(self.buffer.obs_shape,
                    self.buffer.action_dim, device, *args, **kwargs)
        else:
            self.agent = SAC(self.buffer.obs_shape,
                    self.buffer.action_dim, device, *args, **kwargs)

        