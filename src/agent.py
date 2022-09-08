import gym
import torch
from .sac_discrete import SACDiscrete
from .sac_continuous import SAC 
from .buffer import ReplayBuffer
import torch.nn.functional as F
import numpy as np
import os
from .buffer.buffer import Transition
from .pcgrad import PCGrad

def evaluate(env, agent, n_rollout):
    n_envs = env.num_envs
    cnt = np.zeros((n_envs,), dtype=np.int32)
    rewards = np.zeros((n_envs,), dtype=np.float32)
    states = env.reset()
    while (cnt < n_rollout).any():
        action = agent.select_action(states, deterministic=True)
        
        next_state, reward, done, info = env.step(action)
        
        assert rewards.shape == reward.shape
        assert done.shape == cnt.shape
        
        rewards += reward * (cnt < n_rollout) 
        cnt += done
        
        states = state
    return np.mean(rewards) / n_rollout

class PCGradAgent(object):
    def __init__(self,
        env: gym.vector.VectorEnv, 
        eval_env: gym.vector.VectorEnv=None,
        learning_rate: float = 3e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        critic_tau: float = 0.005,
        gamma: float = 0.99,
        seed: int = None,
        device: str = 'cpu',
        *args, **kwargs  # other parameters
    ):
        assert isinstance(env, gym.vector.VectorEnv), (
            "Only support multitask environments"
        )

        self.env = env
        self.eval_env = eval_env
        self.action_space = env.action_space
        observation_space = env.single_observation_space
        action_space = env.single_action_space
        
        self.buffer = ReplayBuffer(observation_space, action_space, 
                buffer_size, batch_size, device, num_envs = env.num_envs)
        discrete_action = isinstance(action_space, gym.spaces.Discrete)
        self.batch_size = batch_size
        self.num_envs = env.num_envs
        self.device = device

        if discrete_action:
            # Onehot representation of task
            obs_shape = self.num_envs + self.buffer.obs_shape
            self.agent = SACDiscrete(obs_shape,
                    self.buffer.action_dim, device, *args, **kwargs)
            
            self.target_entropy = np.log(self.buffer.action_dim) / 5
        else:
            assert not self.buffer.is_image_obs
            # Onehot representation of task
            obs_shape = [self.num_envs + np.prod(self.buffer.obs_shape).item()]
            print(obs_shape)
            self.agent = SAC(obs_shape,
                    self.buffer.action_dim, device, *args, **kwargs)
            self.target_entropy = -np.prod(self.buffer.action_dim)
                    
        self.actor_optimizer = PCGrad(
            torch.optim.Adam(self.agent.actor.parameters(), lr=learning_rate)
        )
        
        self.critic_optimizer = PCGrad(
            torch.optim.Adam(
                self.agent.critic._online_q.parameters(), lr=learning_rate,
            )
        )
        
        # Each task has a separated entropy coefficient
        self.log_ent_coef = torch.log(.5*torch.ones(self.num_envs, device=device)).requires_grad_(True)
        
        self.ent_coef_optimizer = torch.optim.Adam([self.log_ent_coef], 
                lr=learning_rate, 
        )
        
    
    def update_critic(self, batch):
        critic_losses = []
        with torch.no_grad():
            ent_coef = torch.exp(self.log_ent_coef)
            
        for i_task in range(self.num_envs):
            loss = self.agent._critic_loss(batch.get_task(i_task), ent_coef[i_task])
            critic_losses.append(loss)
        self.critic_optimizer.pc_backward(critic_losses)
        self.critic_optimizer.step()
        
        critic_losses = [loss.cpu().item() for loss in critic_losses]
        return np.mean(critic_losses)
        
    def update_actor(self, batch):
        actor_losses = []
        with torch.no_grad():
            ent_coef = torch.exp(self.log_ent_coef)
            
        for i_task in range(self.num_envs):
            loss = self.agent._actor_loss(batch.get_task(i_task), ent_coef[i_task])
            actor_losses.append(loss)
        self.actor_optimizer.pc_backward(actor_losses)
        self.actor_optimizer.step()
        
        actor_losses = [loss.cpu().item() for loss in actor_losses]
        return np.mean(actor_losses)
        
    def update_alpha(self, batch):
        alpha_loss = 0
        for i_task in range(self.num_envs):
            alpha_loss += self.agent._alpha_loss
        
        self.ent_coef_optimizer.zero_grad()
        alpha_loss.backward()
        self.ent_coef_optimizer.step()
        
        return alpha_loss.item()

    def train(self, gradient_steps: int):
        critic_losses, actor_losses, alpha_losses = [], [], []
        for gradient_step in range(gradient_steps):
            # batch has size (batch_size x n_task x ...)
            batch = self.buffer.sample()
            states = create_task_onehot

            
            critic_losses.append(self.update_critic(batch))
            actor_losses.append(self.update_actor(batch))
            alpha_losses.append(self.update_alpha(batch))
        
        return np.mean(critic_losses), np.mean(actor_losses), np.mean(alpha_losses)

    
    def learn(self, total_timesteps, start_step, 
                gradient_steps=1, train_freq=1, num_eval_episodes=10, 
                eval_interval=10000, log_dir='./output',
                
    ):
        num_steps = 0
        env = self.env
        buffer = self.buffer
        states = env.reset()
        loss, val = [], []
        for step in range(total_timesteps):
            if step < start_step: 
                actions = env.action_space.sample()
            else: 
                actions = self.agent.select_action(states)
                
            next_states, rewards, dones, infos = env.step(actions)
            
            buffer.add(states, actions, rewards, next_states, dones, infos)
            
            states = next_states
            if step % train_freq == 0:
                loss.append(self.train(gradient_steps))
            if step % eval_interval == 0:
                eval_return = evaluate(self.eval_env, self, num_eval_episodes)
                val.append(eval_return)
                
                print('mean reward after {} env step: {:.2f}'.format(env_step+1, eval_return))
                print('critic loss: {:.2f} | actor loss: {:.2f} | alpha loss: {:.2f}'.format(
                        *np.mean(list(zip(*loss[-10:])), axis=-1)
                        ))
                print('alpha: {:.2f}'.format(self.log_ent_coef.exp().mean().item()))
            
        import matplotlib.pyplot as plt
        x, y = np.linspace(0, total_env_step, len(val)), val
        plt.plot(x, y)
        
        plt.savefig('res.png')

        import pandas as pd 
        data_dict = {'rollout/ep_rew_mean': y, 'time/total_timesteps': x} # formated as stable baselines
        df = pd.DataFrame(data_dict)

        df.to_csv('sac_progress.csv', index=False)
        sac_agent.save('model', total_env_step)
            
    
    def select_action(self, state, deterministic=False):
        state = self.create_task_onehot(state, torch.arange(self.env.num_envs, dtype=long)).to(self.device)
        return self.agent.select_action(state, deterministic=deterministic)
    
    def create_task_onehot(self, states: torch.Tensor, task_indices):
        """
        Create onehot representations of the task indices
        and append them to the states
        """
        if isinstance(task_indices, int):
            task_indices = torch.LongTensor([task_indices]).to(self.device)
        assert isinstance(task_indices, torch.LongTensor)
        assert len(task_indices.shape) == 1
        onehots = F.one_hot(task_indices, num_classes=self.num_envs)
        
        if len(states.shape) == 1: states = states.unsqueeze(0)
        
        if task_indices.shape[0] == 1:
            broadcast_shape = (*states.shape[:-1], -1)
            task_indices = task_indices.expand(*broadcast_shape)
            
        return torch.cat((states, task_indices), dim=1)
        