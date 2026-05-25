"""
networks/ppo/agent.py — PPO Agent wrapper.

Combines the CNN encoder + PPO for end-to-end action selection.
Handles image encoding, state concatenation, action clamping, and checkpoints.
"""

import os
import logging
import numpy as np
import torch

import parameters as params
from autoencoder.encoder import CNNEncoder
from networks.ppo.ppo import PPO

logger = logging.getLogger(__name__)


class PPOAgent:
    """High-level agent: image → encoder → concat nav → PPO → action."""

    def __init__(self, device=None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"PPOAgent using device: {self.device}")

        # CNN encoder (frozen during PPO training by default)
        self.encoder = CNNEncoder(latent_dim=params.LATENT_DIM).to(self.device)
        self.encoder.eval()  # inference mode for encoder

        # PPO
        self.ppo = PPO(
            state_dim=params.STATE_DIM,
            action_dim=params.ACTION_DIM,
            device=self.device,
        )

        # Timestep counter (for update scheduling)
        self.total_timesteps = 0
        self.update_count = 0

    # ------------------------------------------------------------------
    # Action Selection
    # ------------------------------------------------------------------
    def select_action(self, observation):
        """
        Process observation and select action.

        Args:
            observation: dict with 'image' (H,W,3 uint8) and 'navigation' (7,)
        Returns:
            action: np.array (2,) [steer, throttle] clamped to [-1, 1]
            log_prob: float
            value: float
        """
        # Encode image
        image = observation["image"]  # (H, W, 3) uint8
        image_t = (
            torch.from_numpy(image.copy()).float()
            .permute(2, 0, 1)          # (3, H, W)
            .unsqueeze(0)              # (1, 3, H, W)
            .to(self.device)
        )

        with torch.no_grad():
            latent = self.encoder(image_t)  # (1, 95)

        # Concatenate with navigation features
        nav = observation["navigation"]   # (7,) float32
        nav_t = torch.FloatTensor(nav).unsqueeze(0).to(self.device)  # (1, 7)

        state = torch.cat([latent, nav_t], dim=1)  # (1, 102)
        state_np = state.squeeze(0).cpu().numpy()

        # PPO action
        action, log_prob, value = self.ppo.select_action(state_np)

        # Clamp action
        action = np.clip(action, -1.0, 1.0)

        # Throttle: map from [-1,1] to [0,1]
        action[1] = (action[1] + 1.0) / 2.0

        self.total_timesteps += 1

        return action, log_prob, value, state_np

    # ------------------------------------------------------------------
    # Training Interface
    # ------------------------------------------------------------------
    def store_transition(self, state, action, log_prob, reward, done, value):
        """Store a transition in the PPO buffer."""
        self.ppo.store(state, action, log_prob, reward, done, value)

    def should_update(self):
        """Check if it's time for a PPO update (every N timesteps)."""
        return len(self.ppo.buffer) >= params.PPO_UPDATE_TIMESTEPS

    def update(self):
        """Run PPO learning step. Returns training info dict."""
        info = self.ppo.learn()
        self.update_count += 1

        # Entropy decay
        if self.update_count % params.ACTION_STD_DECAY_FREQ == 0:
            new_std = self.ppo.decay_action_std()
            logger.info(f"Action std decayed to {new_std:.4f}")
            info["action_std"] = new_std

        return info

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------
    def save(self, path=None):
        """Save full checkpoint (encoder + PPO)."""
        if path is None:
            os.makedirs(params.CHECKPOINT_DIR, exist_ok=True)
            path = os.path.join(params.CHECKPOINT_DIR, "latest.pth")

        # Save encoder + PPO together
        checkpoint = {
            "encoder_state_dict": self.encoder.state_dict(),
            "total_timesteps": self.total_timesteps,
            "update_count": self.update_count,
        }
        torch.save(checkpoint, path.replace(".pth", "_encoder.pth"))

        # PPO saves separately (has its own format)
        self.ppo.save(path)

        logger.info(f"Checkpoint saved: {path}")

    def load(self, path):
        """Load checkpoint."""
        # Load PPO
        self.ppo.load(path)

        # Load encoder if available
        encoder_path = path.replace(".pth", "_encoder.pth")
        if os.path.exists(encoder_path):
            checkpoint = torch.load(encoder_path, map_location=self.device)
            self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
            self.total_timesteps = checkpoint.get("total_timesteps", 0)
            self.update_count = checkpoint.get("update_count", 0)
            logger.info(f"Encoder loaded from {encoder_path}")

        self.encoder.eval()
        logger.info(f"Checkpoint loaded: {path}")
