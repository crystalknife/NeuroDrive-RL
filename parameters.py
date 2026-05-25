"""
parameters.py — Central configuration for the CARLA + PPO autonomous driving system.
All hyperparameters, paths, and constants in one place.
"""

import os

# ==============================================================================
# CARLA Connection
# ==============================================================================
CARLA_HOST = "localhost"
CARLA_PORT = 2000
CARLA_TIMEOUT = 20.0
CONNECTION_RETRIES = 5
SYNC_MODE = True
FIXED_DELTA_SECONDS = 0.05  # 20 FPS simulation

# ==============================================================================
# Camera / Sensor
# ==============================================================================
CAMERA_WIDTH = 160
CAMERA_HEIGHT = 80
CAMERA_FOV = 110
SENSOR_TICK = 0.0  # every simulation tick

# ==============================================================================
# CNN Encoder
# ==============================================================================
LATENT_DIM = 95

# ==============================================================================
# Observation Space
# Navigation features: speed, dist_center, angle_road, dist_next_wp, dx, dy, curvature
# ==============================================================================
NAV_FEATURE_DIM = 7
STATE_DIM = LATENT_DIM + NAV_FEATURE_DIM  # 102

# ==============================================================================
# Action Space  (continuous: [steer, throttle])
# ==============================================================================
ACTION_DIM = 2

# ==============================================================================
# PPO Hyperparameters
# ==============================================================================
PPO_LR = 3e-5
PPO_GAMMA = 0.99
PPO_LAMBDA = 0.95           # GAE lambda
PPO_CLIP = 0.2
PPO_ENTROPY_COEFF = 0.005
PPO_VALUE_COEFF = 0.5
PPO_K_EPOCHS = 10
PPO_UPDATE_TIMESTEPS = 2048  # update every N timesteps (not episodes)

# Action std (entropy decay)
ACTION_STD_INIT = 0.08
ACTION_STD_MIN = 0.03
ACTION_STD_DECAY_RATE = 0.95  # multiplied each update
ACTION_STD_DECAY_FREQ = 1      # decay every N updates

# ==============================================================================
# Reward (7-term symmetric system — weights are hardcoded in _compute_reward)
# ==============================================================================
REWARD_COLLISION = -50.0      # collision flat penalty
REWARD_SCALE = 0.1            # total reward *= scale (keep in ~[-5, +3])

# ==============================================================================
# Speed (used for observation normalization only, NOT in reward)
# ==============================================================================
MAX_SPEED = 50.0   # km/h

# ==============================================================================
# Waypoint / Navigation
# ==============================================================================
LOOKAHEAD_STEPS = 8           # waypoints ahead for direction
WAYPOINT_THRESHOLD = 3.0      # meters to consider waypoint reached
ROUTE_RESOLUTION = 2.0        # GlobalRoutePlanner resolution

# ==============================================================================
# Action Smoothing
# ==============================================================================
STEER_SMOOTHING_ALPHA = 0.85   # steer = (1-α)*prev + α*new  (0.1 too sluggish for curves)

# ==============================================================================
# Curriculum Learning
# ==============================================================================
CURRICULUM_LEVELS = {
    1: {"description": "Straight roads only",        "min_wp": 5,  "max_wp": 10},
    2: {"description": "Mild curves",                "min_wp": 10, "max_wp": 20},
    3: {"description": "Mixed roads",                "min_wp": 20, "max_wp": 40},
    4: {"description": "Long routes with curves",    "min_wp": 40, "max_wp": 70},
    5: {"description": "Full map, any route",         "min_wp": 70, "max_wp": 200},
}
CURRICULUM_ADVANCE_EPISODES = 200  # advance difficulty every N episodes

# ==============================================================================
# Episode
# ==============================================================================
MAX_TIMESTEPS_PER_EPISODE = 1000
MAX_EPISODES = 10000

# ==============================================================================
# Checkpoints & Logging
# ==============================================================================
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
CHECKPOINT_FREQ = 50          # save every N episodes
LOG_DIR = os.path.join(os.path.dirname(__file__), "runs")
REWARD_AVG_WINDOW = 100       # rolling average window
