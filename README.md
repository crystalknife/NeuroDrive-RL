# NeuroDrive-RL

Autonomous Driving using Reinforcement Learning and CARLA Simulator.

## Overview

NeuroDrive-RL is a reinforcement learning–based autonomous driving system developed using the CARLA Simulator and Proximal Policy Optimization (PPO). The project focuses on autonomous lane following, vehicle navigation, and real-time driving control inside a simulated urban environment.

The system uses camera-based observations and reinforcement learning techniques to train an intelligent driving agent capable of steering, throttle control, and route navigation.

---

## Features

- PPO-based Reinforcement Learning agent
- CARLA Simulator integration
- Autonomous lane following
- Real-time vehicle control
- Dynamic route generation
- Camera-based observation system
- Training and testing modes
- Vehicle steering smoothing
- Reward-based policy optimization

---

## Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.7 | Core programming language |
| CARLA 0.9.8 | Driving simulator |
| PyTorch | Reinforcement learning |
| PPO | Policy optimization algorithm |
| NumPy | Numerical processing |
| OpenCV | Image processing |
| Unreal Engine | Simulation rendering |

---

## System Architecture

1. CARLA generates the driving environment.
2. Camera sensors capture road observations.
3. PPO agent processes observations.
4. Agent predicts steering and throttle actions.
5. Vehicle controls are applied in simulation.
6. Rewards are calculated based on driving behavior.
7. PPO updates the driving policy.

---

## Project Structure

```bash
NeuroDrive-RL/
│
├── autoencoder/
├── checkpoints/
├── networks/
│   └── ppo/
├── simulation/
├── assets/
├── continuous_driver.py
├── parameters.py
├── README.md
```

---

## Training

Run CARLA:

```bash
CarlaUE4.exe -RenderOffScreen -quality-level=Low
```

Start training:

```bash
python continuous_driver.py \
--mode train \
--checkpoint checkpoints/latest.pth \
--episodes 100 \
--difficulty 1
```

---

## Testing

Run testing with rendering:

```bash
python continuous_driver.py \
--mode test \
--checkpoint checkpoints/latest.pth \
--episodes 10 \
--difficulty 1
```

---

## Results

The trained reinforcement learning agent successfully demonstrated:

- Autonomous lane following
- Straight-road navigation
- Basic curve handling
- Reward optimization through PPO training
- Stable vehicle control in CARLA simulation

---

## Screenshots

### Autonomous Driving Demo

![CARLA Demo](assets/carla_demo.png)

---

### Training Progress

![Training Progress](assets/training_progress.png)

---

## Future Improvements

- Traffic signal handling
- Multi-vehicle navigation
- Pedestrian detection
- Advanced obstacle avoidance
- Real-world sensor integration
- Deep RL performance optimization

---

## Author

Developed as a Reinforcement Learning and Autonomous Driving research project using CARLA Simulator and PPO.
