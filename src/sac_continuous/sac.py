import torch
import torch.nn as nn
from .model import Actor, Critic
import torch.nn.functional as F
import numpy as np

# https://github.com/giangbang/Continuous-SAC
class SAC:
    def __init__(self, 
        obs_shape: np.ndarray,
        action_shape: np.ndarray,
        device='cpu',
        hidden_dim=50,
        discount=0.99,
        learning_rate=3e-4,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        critic_tau=0.005,
        gradient_steps=1,
        num_layers=3,
        init_temperature=1,
        reward_scale=1.,
        *args, **kwargs
    ):
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.reward_scale = reward_scale
        self.gradient_steps = gradient_steps
        self.actor  = Actor(*obs_shape, action_shape, num_layers, 
                hidden_dim, actor_log_std_min, actor_log_std_max).to(device)
                
        self.critic = Critic(*obs_shape, action_shape, num_layers, hidden_dim).to(device)
        
    def _critic_loss(self, batch, ent_coef):
        # Compute target Q 
        with torch.no_grad():
            next_pi, next_log_pi  = self.actor.sample(batch.next_states, compute_log_pi=True)
            next_q_vals = self.critic.target_q(batch.next_states, next_pi)
            next_q_val  = torch.minimum(*next_q_vals)
            
            next_q_val  = next_q_val - ent_coef * next_log_pi
            
            target_q_val= self.reward_scale*batch.rewards + (1-batch.dones)*self.discount*next_q_val
            
        current_q_vals  = self.critic.online_q(batch.states, batch.actions)
        critic_loss     = .5*sum(F.mse_loss(current_q, target_q_val) for current_q in current_q_vals)
        
        return critic_loss
        
    def _actor_loss(self, batch, ent_coef):
        pi, log_pi = self.actor.sample(batch.states, compute_log_pi=True)
        
        q_vals = self.critic.online_q(batch.states, pi)
        q_val  = torch.minimum(*q_vals)
        
        actor_loss = (ent_coef * log_pi - q_val).mean()
        
        return actor_loss
    
    def _alpha_loss(self, batch, ent_coef, target_entropy):
        with torch.no_grad():
            pi, log_pi = self.actor.sample(batch.states, compute_log_pi=True)
        alpha_loss = -(ent_coef * (log_pi + target_entropy).detach()).mean()
        
        return alpha_loss
        
    def update(self, buffer):
        actor_losses, critic_losses, alpha_losses = [], [], []
        
        for _ in range(self.gradient_steps):
            batch = buffer.sample()
            
            critic_loss = self._update_critic(batch)
            actor_loss = self._update_actor(batch)
            alpha_loss = self._update_alpha(batch)
            
            critic_losses.append(critic_loss)
            actor_losses.append(actor_loss)
            alpha_losses.append(alpha_loss)
        
        return np.mean(critic_losses), np.mean(actor_losses), np.mean(alpha_losses)

    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            return self.actor.sample(state, deterministic=deterministic)[0].cpu().numpy()