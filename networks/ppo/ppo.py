"""
networks/ppo/ppo.py — PPO algorithm with Actor-Critic architecture.

Features:
- Continuous action space (MultivariateNormal)
- Clipped surrogate objective
- Advantage normalization
- Entropy decay (action_std annealing)
- Numerical stability guards (clamp, nan_to_num)
- Reward normalization
"""

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np
import parameters as params


# ======================================================================
# Rollout Buffer
# ======================================================================
class RolloutBuffer:
    """Stores transitions for on-policy PPO updates."""

    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def store(self, state, action, log_prob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

    def __len__(self):
        return len(self.states)


# ======================================================================
# Actor-Critic Network
# ======================================================================
class ActorCritic(nn.Module):
    """
    Actor-Critic with shared feature extractor.
    Actor: Gaussian policy (learnable log_std).
    Critic: state value function.
    """

    def __init__(self, state_dim, action_dim,
                 hidden_dim=256, action_std_init=None):
        super().__init__()
        self.action_dim = action_dim

        # Shared feature extractor
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Actor head (outputs action mean)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh(),   # bound output to [-1, 1]
        )

        # Critic head (outputs scalar value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Learnable log standard deviation
        init_std = action_std_init or params.ACTION_STD_INIT
        self.log_std = nn.Parameter(
            torch.full((action_dim,), fill_value=float(torch.tensor(init_std).log()))
        )

    def forward(self, state):
        """Returns (action_mean, value)."""
        features = self.feature(state)
        action_mean = self.actor(features)
        value = self.critic(features)
        return action_mean, value

    def act(self, state):
        """
        Sample action from policy.
        Returns (action, log_prob).
        """
        features = self.feature(state)
        action_mean = self.actor(features)

        # Clamp mean for stability
        action_mean = torch.clamp(action_mean, -1.0, 1.0)

        # Build covariance (diagonal)
        action_std = torch.exp(self.log_std)
        cov_var = torch.clamp(action_std ** 2, min=1e-4)
        cov_matrix = torch.diag_embed(cov_var)

        dist = MultivariateNormal(action_mean, cov_matrix)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # Guard NaN
        log_prob = torch.nan_to_num(log_prob, nan=0.0)

        return action.detach(), log_prob.detach()

    def evaluate(self, state, action):
        """
        Evaluate actions under current policy.
        Returns (log_prob, value, entropy).
        """
        features = self.feature(state)
        action_mean = self.actor(features)
        value = self.critic(features).squeeze(-1)

        # Clamp
        action_mean = torch.clamp(action_mean, -1.0, 1.0)

        action_std = torch.exp(self.log_std)
        cov_var = torch.clamp(action_std ** 2, min=1e-4)
        cov_matrix = torch.diag(cov_var)

        dist = MultivariateNormal(action_mean, cov_matrix)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        # Guard NaN
        log_prob = torch.nan_to_num(log_prob, nan=0.0)
        entropy = torch.nan_to_num(entropy, nan=0.0)
        value = torch.nan_to_num(value, nan=0.0)

        return log_prob, value, entropy

    def set_action_std(self, new_std):
        """Update action std (for entropy decay)."""
        new_std = max(new_std, params.ACTION_STD_MIN)
        self.log_std.data = torch.full_like(
            self.log_std, fill_value=float(torch.tensor(new_std).log())
        )


# ======================================================================
# PPO Algorithm
# ======================================================================
class PPO:
    """Proximal Policy Optimization with clipping."""

    def __init__(self, state_dim, action_dim, device="cpu"):
        self.device = device

        # Networks
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=params.PPO_LR
        )

        # Loss
        self.mse_loss = nn.MSELoss()

        # Buffer
        self.buffer = RolloutBuffer()

        # Reward normalization stats (running)
        self._reward_mean = 0.0
        self._reward_var = 1.0
        self._reward_count = 0

        # Current action std
        self.action_std = params.ACTION_STD_INIT

    def select_action(self, state):
        """
        Select action using old policy (for rollout collection).
        state: tensor (state_dim,)
        Returns (action, log_prob, value).
        """
        with torch.no_grad():
            state_t = torch.FloatTensor(state).to(self.device)
            if state_t.dim() == 1:
                state_t = state_t.unsqueeze(0)

            action, log_prob = self.policy_old.act(state_t)

            # Get value estimate
            _, value = self.policy_old(state_t)
            value = value.squeeze()

            return (
                action.squeeze(0).cpu().numpy(),
                log_prob.cpu().item(),
                value.cpu().item(),
            )

    def store(self, state, action, log_prob, reward, done, value):
        """Store transition in buffer."""
        self.buffer.store(state, action, log_prob, reward, done, value)

    def _normalize_rewards(self, rewards):
        """Normalize rewards using running statistics."""
        rewards_np = rewards.cpu().numpy()
        batch_mean = rewards_np.mean()
        batch_var = rewards_np.var()
        batch_count = len(rewards_np)

        # Update running stats (Welford's online algorithm)
        total_count = self._reward_count + batch_count
        if total_count > 0:
            delta = batch_mean - self._reward_mean
            self._reward_mean += delta * batch_count / total_count
            self._reward_var = (
                (self._reward_var * self._reward_count + batch_var * batch_count)
                / total_count
            )
            self._reward_count = total_count

        std = max(self._reward_var ** 0.5, 1e-4)
        return (rewards - self._reward_mean) / std

    def learn(self):
        """
        PPO update using collected rollout buffer.
        """
        if len(self.buffer) == 0:
            return {}

        # Convert buffer to tensors
        states = torch.from_numpy(np.array(self.buffer.states)).float().to(self.device)
        actions = torch.from_numpy(np.array(self.buffer.actions)).float().to(self.device)
        old_log_probs = torch.FloatTensor(self.buffer.log_probs).to(self.device)
        rewards_raw = torch.FloatTensor(self.buffer.rewards).to(self.device)
        dones = torch.FloatTensor(self.buffer.dones).to(self.device)
        values = torch.FloatTensor(self.buffer.values).to(self.device)

        # Normalize rewards
        rewards = self._normalize_rewards(rewards_raw)

        # Compute returns and advantages (GAE)
        returns = []
        advantages = []
        gae = 0.0
        next_value = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = 0.0
            else:
                next_val = values[t + 1].item()

            delta = (
                rewards[t].item()
                + params.PPO_GAMMA * next_val * (1.0 - dones[t].item())
                - values[t].item()
            )
            gae = delta + params.PPO_GAMMA * params.PPO_LAMBDA * (
                1.0 - dones[t].item()
            ) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t].item())

        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)

        # Advantage normalization
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-7
            )

        # PPO epochs
        total_loss = 0.0
        for _ in range(params.PPO_K_EPOCHS):
            log_probs, state_values, entropy = self.policy.evaluate(
                states, actions
            )

            # Ratio
            ratios = torch.exp(log_probs - old_log_probs)

            # Clipped surrogate
            surr1 = ratios * advantages
            surr2 = torch.clamp(
                ratios, 1.0 - params.PPO_CLIP, 1.0 + params.PPO_CLIP
            ) * advantages

            # Loss = - surrogate + value_loss - entropy_bonus
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = self.mse_loss(state_values, returns)
            entropy_loss = -entropy.mean()

            loss = (
                policy_loss
                + params.PPO_VALUE_COEFF * value_loss
                + params.PPO_ENTROPY_COEFF * entropy_loss
            )

            # Guard NaN loss
            if torch.isnan(loss):
                print("[WARN] NaN loss detected, skipping update step.")
                continue

            self.optimizer.zero_grad()
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

            total_loss += loss.item()

        # Copy policy to old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Clear buffer
        self.buffer.clear()

        return {
            "loss": total_loss / max(params.PPO_K_EPOCHS, 1),
            "buffer_size": len(states),
        }

    def decay_action_std(self):
        """
        Decay action std: multiply by decay rate, clamp to minimum.
        """
        self.action_std = max(
            self.action_std * params.ACTION_STD_DECAY_RATE,
            params.ACTION_STD_MIN,
        )
        self.policy.set_action_std(self.action_std)
        self.policy_old.set_action_std(self.action_std)
        return self.action_std

    def save(self, path):
        """Save model checkpoint."""
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "action_std": self.action_std,
            "reward_mean": self._reward_mean,
            "reward_var": self._reward_var,
            "reward_count": self._reward_count,
        }, path)

    def load(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.policy_old.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.action_std = checkpoint.get("action_std", params.ACTION_STD_INIT)
        self._reward_mean = checkpoint.get("reward_mean", 0.0)
        self._reward_var = checkpoint.get("reward_var", 1.0)
        self._reward_count = checkpoint.get("reward_count", 0)
        self.policy.set_action_std(self.action_std)
        self.policy_old.set_action_std(self.action_std)
