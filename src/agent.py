import gym
import torch

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
        *args, **kwargs  # not used parameters
    ):
        assert isinstance(env, gym.vector.VectorEnv), (
            "Only support multitask environments"
        )