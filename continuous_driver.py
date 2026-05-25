"""
continuous_driver.py — Main training/testing loop for CARLA + PPO.

Features:
- Fault-tolerant: skip on reset failure, end episode on step failure
- Timestep-based PPO updates (every 2048 steps)
- Curriculum auto-advancement
- Entropy decay
- TensorBoard logging
- Checkpoint save/load
- Reward trend tracking
"""

import os
import sys

# CARLA core
sys.path.append(r"D:\car\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.8-py3.7-win-amd64.egg")

# CARLA agents (THIS IS WHAT YOU’RE MISSING)
sys.path.append(r"D:\car\WindowsNoEditor\PythonAPI\carla")
import argparse
import logging
import time
from collections import deque

import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

import parameters as params
from simulation.connection import CarlaConnection
from simulation.environment import CarlaEnvironment
from networks.ppo.agent import PPOAgent

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("driver")


def parse_args():
    parser = argparse.ArgumentParser(
        description="CARLA Autonomous Driving — PPO Training/Testing"
    )
    parser.add_argument(
        "--mode", type=str, default="train",
        choices=["train", "test"],
        help="Run mode: train or test",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to checkpoint file to load",
    )
    parser.add_argument(
        "--episodes", type=int, default=params.MAX_EPISODES,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--difficulty", type=int, default=1,
        choices=[1, 2, 3, 4, 5],
        help="Initial curriculum difficulty level",
    )
    return parser.parse_args()


# ======================================================================
# Training
# ======================================================================
def train(args):
    """Main training loop."""

    # --- Connect to CARLA ---
    conn = CarlaConnection()
    try:
        conn.connect()
    except ConnectionError as e:
        logger.error(f"Cannot start training: {e}")
        return

    # --- Environment & Agent ---
    env = CarlaEnvironment(conn, difficulty=args.difficulty)
    agent = PPOAgent()

    # Load checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        agent.load(args.checkpoint)
        logger.info(f"Resumed from checkpoint: {args.checkpoint}")

    # TensorBoard
    writer = None
    if SummaryWriter is not None:
        os.makedirs(params.LOG_DIR, exist_ok=True)
        writer = SummaryWriter(log_dir=params.LOG_DIR)
        logger.info(f"TensorBoard logging to: {params.LOG_DIR}")

    # Tracking
    reward_history = deque(maxlen=params.REWARD_AVG_WINDOW)
    total_timesteps = 0
    reset_failures = 0
    step_failures = 0
    current_difficulty = args.difficulty

    os.makedirs(params.CHECKPOINT_DIR, exist_ok=True)

    logger.info("=" * 60)
    logger.info("TRAINING STARTED")
    logger.info(f"  Episodes: {args.episodes}")
    logger.info(f"  Difficulty: {current_difficulty}")
    logger.info(f"  PPO update every: {params.PPO_UPDATE_TIMESTEPS} timesteps")
    logger.info(f"  Action std: {params.ACTION_STD_INIT} → {params.ACTION_STD_MIN}")
    logger.info("=" * 60)

    try:
        for episode in range(1, args.episodes + 1):
            # --- Reset ---
            obs = env.reset()
            if obs is None:
                reset_failures += 1
                logger.warning(
                    f"[EP {episode}] Reset failed "
                    f"(total failures: {reset_failures}). Skipping."
                )
                continue

            episode_reward = 0.0
            episode_steps = 0
            done = False

            # --- Episode loop ---
            while not done:
                # Select action
                action, log_prob, value, state = agent.select_action(obs)

                # Step environment
                next_obs, reward, done, info = env.step(action)

                if "error" in info:
                    step_failures += 1
                    logger.warning(
                        f"[EP {episode}] Step failed at t={episode_steps} "
                        f"(total failures: {step_failures})"
                    )
                    # done is already True from step() error handling
                    break

                # Store transition (using state vector, not raw obs)
                agent.store_transition(
                    state=state,
                    action=action,
                    log_prob=log_prob,
                    reward=reward,
                    done=done,
                    value=value,
                )

                episode_reward += reward
                episode_steps += 1
                total_timesteps += 1

                # PPO update (timestep-based)
                if agent.should_update():
                    update_info = agent.update()
                    if update_info:
                        logger.info(
                            f"  [UPDATE #{agent.update_count}] "
                            f"loss={update_info.get('loss', 0):.4f} "
                            f"buffer={update_info.get('buffer_size', 0)} "
                            f"std={update_info.get('action_std', agent.ppo.action_std):.4f}"
                        )
                        if writer:
                            writer.add_scalar(
                                "train/loss",
                                update_info.get("loss", 0),
                                total_timesteps,
                            )

                obs = next_obs

            # --- Episode stats ---
            reward_history.append(episode_reward)
            avg_reward = np.mean(reward_history)

            # Trend detection
            trend = "---"
            if len(reward_history) >= 20:
                recent = list(reward_history)
                first_half = np.mean(recent[: len(recent) // 2])
                second_half = np.mean(recent[len(recent) // 2 :])
                if second_half > first_half * 1.05:
                    trend = "↑ improving"
                elif second_half < first_half * 0.95:
                    trend = "↓ declining"
                else:
                    trend = "→ stable"

            print(
                f"EP {episode:>5d} | "
                f"Steps {episode_steps:>4d} | "
                f"Reward {episode_reward:>8.2f} | "
                f"Avg(100) {avg_reward:>8.2f} | "
                f"Trend {trend:>12s} | "
                f"Diff {current_difficulty} | "
                f"T {total_timesteps}"
            )

            # TensorBoard
            if writer:
                writer.add_scalar("train/episode_reward", episode_reward, episode)
                writer.add_scalar("train/avg_reward", avg_reward, episode)
                writer.add_scalar("train/episode_length", episode_steps, episode)
                writer.add_scalar("train/difficulty", current_difficulty, episode)
                writer.add_scalar(
                    "train/action_std", agent.ppo.action_std, episode
                )

            # Checkpoint
            if episode % params.CHECKPOINT_FREQ == 0:
                ckpt_path = os.path.join(
                    params.CHECKPOINT_DIR, f"ep{episode}.pth"
                )
                agent.save(ckpt_path)
                agent.save(
                    os.path.join(params.CHECKPOINT_DIR, "latest.pth")
                )

            # # Curriculum advancement
            # if (
            #     episode % params.CURRICULUM_ADVANCE_EPISODES == 0
            #     and current_difficulty < 5
            # ):
            #     current_difficulty += 1
            #     env.set_difficulty(current_difficulty)
            #     logger.info(
            #         f"[CURRICULUM] Advancing to level {current_difficulty}: "
            #         f"{params.CURRICULUM_LEVELS[current_difficulty]['description']}"
            #     )

    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)
    finally:
        # Final save
        agent.save(os.path.join(params.CHECKPOINT_DIR, "latest.pth"))
        logger.info("Final checkpoint saved.")

        # Stats
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print(f"  Total episodes attempted: {args.episodes}")
        print(f"  Total timesteps: {total_timesteps}")
        print(f"  Reset failures: {reset_failures}")
        print(f"  Step failures: {step_failures}")
        print(f"  Final avg reward: {np.mean(reward_history):.2f}")
        print(f"  Final difficulty: {current_difficulty}")
        print("=" * 60)

        env.close()
        conn.cleanup()
        if writer:
            writer.close()


# ======================================================================
# Testing
# ======================================================================
def test(args):
    """Run agent in test (greedy) mode."""

    if not args.checkpoint:
        logger.error("Test mode requires --checkpoint path.")
        return

    # --- Connect to CARLA ---
    conn = CarlaConnection()
    try:
        conn.connect()
    except ConnectionError as e:
        logger.error(f"Cannot start testing: {e}")
        return

    # --- Environment & Agent ---
    env = CarlaEnvironment(conn, difficulty=args.difficulty)
    env.set_difficulty(1)
    current_difficulty = 1
    agent = PPOAgent()
    agent.load(args.checkpoint)
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    reward_history = []

    logger.info("=" * 60)
    logger.info(f"TESTING — {args.episodes} episodes, difficulty {args.difficulty}")
    logger.info("=" * 60)

    try:
        for episode in range(1, args.episodes + 1):
            obs = env.reset()
            if obs is None:
                logger.warning(f"[EP {episode}] Reset failed. Skipping.")
                continue

            episode_reward = 0.0
            episode_steps = 0
            done = False

            while not done:
                action, _, _, _ = agent.select_action(obs)
                obs, reward, done, info = env.step(action)

                if "error" in info:
                    break

                episode_reward += reward
                episode_steps += 1

            reward_history.append(episode_reward)
            print(
                f"TEST EP {episode:>3d} | "
                f"Steps {episode_steps:>4d} | "
                f"Reward {episode_reward:>8.2f}"
            )

    except KeyboardInterrupt:
        logger.info("Testing interrupted.")
    finally:
        if reward_history:
            print(f"\nAvg test reward: {np.mean(reward_history):.2f}")
        env.close()
        conn.cleanup()


# ======================================================================
# Entry Point
# ======================================================================
def main():
    args = parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)
    else:
        logger.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
