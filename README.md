# FruitSwap

A MuJoCo environment implementing the block-swapping task from the Diffusion Forcing paper using a Franka Panda robotic arm.

## Task Description

The task involves swapping positions of two blocks (red and blue) using a third empty slot. This is a non-Markovian task requiring memory of the initial configuration and demonstrates complex manipulation capabilities.

## Features

- **Non-Markovian Task**: Requires memory of initial configuration
- **Multiple Observation Modes**: Full state or partial (camera-based) observations
- **Advanced Reward Shaping**: Progress tracking with optional sparse rewards
- **Flexible Control**: End-effector delta control or joint velocity control
- **Franka Panda Integration**: Realistic robot simulation with MuJoCo

## Installation

### From Git Repository

```bash
pip install git+https://github.com/pranavnt/fruitswap.git
```

### For Development

```bash
git clone https://github.com/pranavnt/fruitswap.git
cd fruitswap
pip install -e .
```

## Dependencies

- `numpy>=1.21.0`
- `gymnasium>=0.28.0`
- `mujoco>=2.3.0`

## Quick Start

```python
from fruitswap import BlockSwapEnv

# Create environment
env = BlockSwapEnv(
    observation_mode='full',  # or 'partial' for camera-based obs
    reward_shaping=True,      # Enable shaped rewards
    sparse_reward=False,      # Use only sparse completion reward
)

# Reset and run
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

## Environment Configuration

### Parameters

- `observation_mode`: `'full'` (state-based) or `'partial'` (camera-based)
- `reward_shaping`: Enable detailed progress rewards (default: `True`)
- `sparse_reward`: Only reward task completion (default: `False`)
- `max_episode_steps`: Maximum episode length (default: `500`)

### Action Space

3D end-effector delta control: `[dx, dy, dz]`

### Observation Space

- **Full mode**: Complete state information including robot joint positions, block positions, and task progress
- **Partial mode**: Camera-based observations requiring visual processing

## Command Line Usage

Run a demo:

```bash
fruitswap
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
