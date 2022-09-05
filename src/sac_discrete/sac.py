import torch
import torch.nn as nn
from .model import Actor, Critic
import torch.nn.functional as F
import numpy as np

class SACDiscrete:
    def __init__(self, 
        obs_shape: np.ndarray,
        action_dim: int,
        device='cpu',
        hidden_dim=256,
        discount=0.99,
        alpha_lr=3e-4,
        actor_lr=3e-4,
        critic_lr=3e-4,
        critic_tau=0.005,
        gradient_steps=1,
        num_layers=3,
        init_temperature=1,
        optimizer_args:dict=None,
        *args, **kwargs
    ):
        self.discount = discount
        self.critic_tau = critic_tau
        self.gradient_steps = gradient_steps
        self.device = device
        
        self.actor  = Actor(obs_shape, action_dim, 
                    num_layers, hidden_dim, ).to(device)
                    
        self.critic = Critic(obs_shape, action_dim, num_layers, hidden_dim).to(device)
        
        # automatically set target entropy if needed
        # This roughly equivalent to epsilon-greedy policy
        # with 20% exploration
        self.target_entropy = np.log(action_dim) / 5
        
        if optimizer_args is None:
            optimizer_args = {}
        assert isinstance(optimizer_args, dict)
        
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, **optimizer_args
        )
        
        self.critic_optimizer = torch.optim.Adam(
            self.critic._online_q.parameters(), lr=critic_lr, **optimizer_args
        )
        
        self.log_ent_coef = torch.log(init_temperature*torch.ones(1, device=device)).requires_grad_(True)
        
        self.ent_coef_optimizer = torch.optim.Adam([self.log_ent_coef], 
            lr=alpha_lr, **optimizer_args
        )
        
    def _update_critic(self, batch):
        # Compute target Q 
        with torch.no_grad():
            next_pi, next_entropy  = self.actor.probs(batch.next_states, compute_log_pi=True)
            
            next_q_vals = self.critic.target_q(batch.next_states)
            next_q_val  = torch.minimum(*next_q_vals)
            
            next_q_val = (next_q_val * next_pi).sum(
                dim=1, keepdims=True
            )
            
            ent_coef    = torch.exp(self.log_ent_coef)
            next_q_val  = next_q_val + ent_coef * next_entropy.reshape(-1, 1)
            
            target_q_val= batch.rewards + (1-batch.dones)*self.discount*next_q_val
            
        current_q_vals  = self.critic.online_q(batch.states)
        current_q_vals = [
            current_q.gather(1, batch.actions)
            for current_q in current_q_vals
        ]
        critic_loss     = .5*sum(F.mse_loss(current_q, target_q_val) for current_q in current_q_vals)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        self.critic.polyak_update(self.critic_tau)
        
        return critic_loss.item()
        
    def _update_actor(self, batch):
        pi, ent = self.actor.probs(batch.states, compute_log_pi=True)
        
        q_vals = self.critic.online_q(batch.states)
        q_val  = torch.minimum(*q_vals)
        
        with torch.no_grad():
            ent_coef = torch.exp(self.log_ent_coef)
        
        actor_loss = (pi * q_val).sum(
            dim=1, keepdims=True
        ) + ent_coef * ent.reshape(-1, 1)
        actor_loss = -actor_loss.mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return actor_loss.item()
        
    def _update_alpha(self, batch):
        with torch.no_grad():
            pi, entropy = self.actor.probs(batch.states, compute_log_pi=True)
        alpha_loss = -(
            self.log_ent_coef * (-entropy + self.target_entropy).detach()
        ).mean()
        
        self.ent_coef_optimizer.zero_grad()
        alpha_loss.backward()
        self.ent_coef_optimizer.step()
        
        return alpha_loss.item()
        
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
            state = torch.FloatTensor(state).to(self.device)
            if len(state.shape) == 1: state = state.unsqueeze(0)
            return self.actor.sample(state, False, deterministic)[0].cpu().numpy().item()
    
    def save(self, model_dir, step):
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )
        
    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )
        