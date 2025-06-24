# BlockSwap

A MuJoCo environment implementing the block-swapping task from the Diffusion Forcing paper using a Franka Panda robotic arm.

## Task Description

The task involves swapping positions of two blocks (red and blue) using a third empty slot. This is a non-Markovian task requiring memory of the initial configuration and demonstrates complex manipulation capabilities.

## Features

- **Non-Markovian Task**: Requires memory of initial configuration
- **Multiple Observation Modes**: Full state (single vector) or partial (dictionary with camera data)
- **Dense or Sparse Rewards**: Progress tracking with optional sparse reward mode
- **Flexible Control**: End-effector delta control or joint velocity control
- **Franka Panda Integration**: Realistic robot simulation with MuJoCo

## Observation Spaces

### Full Observability (`observation_mode='full'`)
Returns a single concatenated vector with 49 elements:
- Robot joint positions: 7 elements
- Robot joint velocities: 7 elements
- End-effector position: 3 elements
- End-effector quaternion: 4 elements
- Gripper state: 2 elements
- Red block position: 3 elements
- Red block quaternion: 4 elements
- Blue block position: 3 elements
- Blue block quaternion: 4 elements
- Slot occupancy: 3 elements
- Initial configuration: 3 elements
- Distance metrics: 6 elements

### Partial Observability (`observation_mode='partial'`)
Returns a dictionary with camera-based observations:
- `front_camera`: (128×128×3) RGB image
- `wrist_camera`: (84×84×3) RGB image
- `robot_qpos`: (7,) joint positions
- `gripper_state`: (2,) gripper joint positions
- `distance_metrics`: (6,) various distance measurements

## Installation

### From Git Repository

```bash
pip install git+https://github.com/pranavnt/blockswap.git
```

### For Development

```bash
git clone --recursive https://github.com/pranavnt/blockswap.git
cd blockswap
pip install -e .
```

**Note**: This package includes only the necessary Franka Panda assets from MuJoCo Menagerie, keeping the package size optimized (~5MB instead of 1.4GB). For development, the full mujoco_menagerie is available as a git submodule - clone with `--recursive` or run `git submodule update --init --recursive` after cloning.

## Dependencies

- `numpy>=1.21.0`
- `gymnasium>=0.28.0`
- `mujoco>=2.3.0`

## Quick Start

```python
from blockswap import BlockSwapEnv

# Create environment
env = BlockSwapEnv(
    observation_mode='full',  # 'full' = dict with all state info, 'partial' = single vector with cameras
    sparse_reward=False,      # If True, use only sparse completion reward
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
blockswap
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
