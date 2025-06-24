"""
BlockSwap - Block Swapping Environment with Franka Panda Arm

A MuJoCo environment implementing the block-swapping task from the Diffusion Forcing paper.
Task: Swap positions of red and blue blocks using a third empty slot.
"""

from .blockswap_env import BlockSwapEnv
from .panda_ik import PandaIK

# Register the environment with gymnasium
from gymnasium.envs.registration import register

register(
    id='blockswap',
    entry_point='blockswap:BlockSwapEnv',
    max_episode_steps=500,
)

register(
    'blockswap-partially-observable',
    entry_point='blockswap:BlockSwapEnv',
    max_episode_steps=500,
    kwargs={
        'observation_mode': 'partial',
    }
)

__version__ = "0.1.0"
__author__ = "Pranav Teegavarapu"
__email__ = "pranavnt@outlook.com"

__all__ = ["BlockSwapEnv", "PandaIK"]