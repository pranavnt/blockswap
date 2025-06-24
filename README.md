# BlockSwap

A MuJoCo environment implementing the block-swapping task from the Diffusion Forcing paper using a Franka Panda robotic arm.

## Task Description

The task involves swapping positions of two blocks (red and blue) using a third empty slot. This is a non-Markovian task requiring memory of the initial configuration and demonstrates complex manipulation capabilities.

## Features

- **Non-Markovian Task**: Requires memory of initial configuration
- **Multiple Observation Modes**: Full state (dict) or partial (single concatenated vector with camera data)
- **Dense or Sparse Rewards**: Progress tracking with optional sparse reward mode
- **Flexible Control**: End-effector delta control or joint velocity control
- **Franka Panda Integration**: Realistic robot simulation with MuJoCo

## Observation Spaces

### Full Observability (`observation_mode='full'`)
Returns a dictionary with complete state information:
- `robot_qpos`: (7,) joint positions
- `robot_qvel`: (7,) joint velocities
- `ee_pos`: (3,) end-effector position
- `ee_quat`: (4,) end-effector quaternion
- `gripper_state`: (2,) gripper joint positions
- `red_block_pos`: (3,) red block position
- `red_block_quat`: (4,) red block quaternion
- `blue_block_pos`: (3,) blue block position
- `blue_block_quat`: (4,) blue block quaternion
- `slot_occupancy`: (3,) which block is in each slot
- `initial_config`: (3,) initial block configuration
- `distance_metrics`: (6,) various distance measurements

### Partial Observability (`observation_mode='partial'`)
Returns a single concatenated vector with 70,335 elements:
- Front camera (128×128×3): 49,152 elements (normalized to [0,1])
- Wrist camera (84×84×3): 21,168 elements (normalized to [0,1])
- Robot joint positions: 7 elements
- Gripper state: 2 elements
- Distance metrics: 6 elements

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
