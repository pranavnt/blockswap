"""
FruitSwap - Block Swapping Environment with Franka Panda Arm

A MuJoCo environment implementing the block-swapping task from the Diffusion Forcing paper.
Task: Swap positions of red and blue blocks using a third empty slot.
"""

from .blockswap_env import BlockSwapEnv
from .panda_ik import PandaIK

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = ["BlockSwapEnv", "PandaIK"]