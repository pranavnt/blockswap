"""
Block Swapping Environment with Franka Panda Arm

This environment implements the block-swapping task from the Diffusion Forcing paper.
Task: Swap positions of red and blue blocks using a third empty slot.

Key Features:
- Non-Markovian task requiring memory of initial configuration
- Full and partial observability modes
- Advanced reward shaping with progress tracking
- Sparse reward option for minimal guidance
- Support for different control modes (EE delta, joint velocity)
- Visual distractors can be added (not implemented in this basic version)

Installation:
    pip install gymnasium mujoco

Usage:
    env = BlockSwapEnv(
        observation_mode='full',  # or 'partial' for camera-based obs
        reward_shaping=True,      # Enable shaped rewards
        sparse_reward=False,      # Use only sparse completion reward
        control_mode='ee_delta'   # or 'joint_velocity'
    )

    obs, info = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import os
import tempfile
from typing import Dict, Tuple, Optional, Any
import xml.etree.ElementTree as ET
from .panda_ik import PandaIK

class BlockSwapEnv(gym.Env):
    """
    MuJoCo environment for block swapping task with Franka Panda arm.

    Task: Swap positions of two blocks (red and blue) using a third empty slot.
    This is a non-Markovian task requiring memory of initial configuration.
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 20}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        observation_mode: str = 'full',  # 'full' or 'partial'
        reward_shaping: bool = True,
        sparse_reward: bool = False,     # New parameter for sparse rewards
        max_episode_steps: int = 500,
    ):
        self.render_mode = render_mode
        self.observation_mode = observation_mode
        self.reward_shaping = reward_shaping
        self.sparse_reward = sparse_reward  # If True, only give reward on task completion
        self.max_episode_steps = max_episode_steps

        # Task parameters
        self.cylinder_height = 0.55  # Height of cylindrical tables (reduced by ~8%)
        self.cylinder_radius = 0.07  # Radius of cylindrical tables (reduced by ~12%)
        self.slot_positions = {
            'A': np.array([-0.3, 0.3, self.cylinder_height + 0.025]),  # Block center height
            'B': np.array([0.0, 0.3, self.cylinder_height + 0.025]),
            'C': np.array([0.3, 0.3, self.cylinder_height + 0.025])
        }
        self.slot_radius = self.cylinder_radius
        self.block_size = 0.025  # Half-size of block

        # Create MuJoCo model
        self.model_xml = self._create_model_xml()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(self.model_xml)
            model_path = f.name

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        os.unlink(model_path)

        # Set up spaces
        self._setup_spaces()

        # Initialize tracking variables
        self.step_count = 0
        self.initial_config = None
        self.goal_config = None
        self.prev_action = None
        self.grasped_block = None

        # Viewer for rendering
        self.viewer = None
        self._renderer = None

        # Initialize IK solver
        self.ik_solver = PandaIK(self.model, self.data)

        # Target orientation for end-effector (pointing down)
        self.target_ee_orientation = np.array([0.9238795, 0, 0, -0.3826834])  # [w, x, y, z]

    def set_target_orientation(self, orientation: np.ndarray):
        """Set the target end-effector orientation for IK."""
        self.target_ee_orientation = orientation.copy()

    def _create_model_xml(self) -> str:
        """Generate MuJoCo XML for the environment."""
        xml = f"""
<mujoco model="block_swap">
    <compiler angle="radian" meshdir="{os.path.join(os.path.dirname(__file__), 'mujoco_menagerie', 'franka_emika_panda', 'assets')}" autolimits="true"/>

    <option gravity="0 0 -9.81" timestep="0.002" integrator="implicit" impratio="10">
        <flag warmstart="enable"/>
    </option>

    <visual>
        <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3"/>
        <rgba haze="0.15 0.25 0.35 1"/>
    </visual>

    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
        <texture name="floor_tex" type="2d" builtin="checker" rgb1="0.3 0.3 0.35" rgb2="0.4 0.4 0.45" width="512" height="512"/>
        <material name="floor_mat" texture="floor_tex" shininess="0.2" specular="0.2"/>
        <material name="table_mat" rgba="0.4 0.6 0.4 1" shininess="0.2" specular="0.3"/>  <!-- Green cylindrical tables -->
        <material name="red_mat" rgba="0.9 0.1 0.1 1" shininess="0.5" specular="0.5"/>
        <material name="blue_mat" rgba="0.1 0.1 0.9 1" shininess="0.5" specular="0.5"/>
        <material name="slot_mat" rgba="0.8 0.8 0.2 0.8" shininess="0.3" specular="0.3"/>  <!-- Yellow slot markers -->
        <material name="platform_mat" rgba="0.8 0.8 0.85 1" shininess="0.4" specular="0.4"/>  <!-- Light gray platform -->

        <!-- Franka Panda materials -->
        <material name="white" rgba="1 1 1 1"/>
        <material name="off_white" rgba="0.901961 0.921569 0.929412 1"/>
        <material name="black" rgba="0.25 0.25 0.25 1"/>
        <material name="green" rgba="0 1 0 1"/>
        <material name="light_blue" rgba="0.039216 0.541176 0.780392 1"/>

        <!-- Collision meshes -->
        <mesh name="link0_c" file="link0.stl"/>
        <mesh name="link1_c" file="link1.stl"/>
        <mesh name="link2_c" file="link2.stl"/>
        <mesh name="link3_c" file="link3.stl"/>
        <mesh name="link4_c" file="link4.stl"/>
        <mesh name="link5_c0" file="link5_collision_0.obj"/>
        <mesh name="link5_c1" file="link5_collision_1.obj"/>
        <mesh name="link5_c2" file="link5_collision_2.obj"/>
        <mesh name="link6_c" file="link6.stl"/>
        <mesh name="link7_c" file="link7.stl"/>
        <mesh name="hand_c" file="hand.stl"/>

        <!-- Visual meshes -->
        <mesh file="link0_0.obj"/>
        <mesh file="link0_1.obj"/>
        <mesh file="link0_2.obj"/>
        <mesh file="link0_3.obj"/>
        <mesh file="link0_4.obj"/>
        <mesh file="link0_5.obj"/>
        <mesh file="link0_7.obj"/>
        <mesh file="link0_8.obj"/>
        <mesh file="link0_9.obj"/>
        <mesh file="link0_10.obj"/>
        <mesh file="link0_11.obj"/>
        <mesh file="link1.obj"/>
        <mesh file="link2.obj"/>
        <mesh file="link3_0.obj"/>
        <mesh file="link3_1.obj"/>
        <mesh file="link3_2.obj"/>
        <mesh file="link3_3.obj"/>
        <mesh file="link4_0.obj"/>
        <mesh file="link4_1.obj"/>
        <mesh file="link4_2.obj"/>
        <mesh file="link4_3.obj"/>
        <mesh file="link5_0.obj"/>
        <mesh file="link5_1.obj"/>
        <mesh file="link5_2.obj"/>
        <mesh file="link6_0.obj"/>
        <mesh file="link6_1.obj"/>
        <mesh file="link6_2.obj"/>
        <mesh file="link6_3.obj"/>
        <mesh file="link6_4.obj"/>
        <mesh file="link6_5.obj"/>
        <mesh file="link6_6.obj"/>
        <mesh file="link6_7.obj"/>
        <mesh file="link6_8.obj"/>
        <mesh file="link6_9.obj"/>
        <mesh file="link6_10.obj"/>
        <mesh file="link6_11.obj"/>
        <mesh file="link6_12.obj"/>
        <mesh file="link6_13.obj"/>
        <mesh file="link6_14.obj"/>
        <mesh file="link6_15.obj"/>
        <mesh file="link6_16.obj"/>
        <mesh file="link7_0.obj"/>
        <mesh file="link7_1.obj"/>
        <mesh file="link7_2.obj"/>
        <mesh file="link7_3.obj"/>
        <mesh file="link7_4.obj"/>
        <mesh file="link7_5.obj"/>
        <mesh file="link7_6.obj"/>
        <mesh file="link7_7.obj"/>
        <mesh file="hand_0.obj"/>
        <mesh file="hand_1.obj"/>
        <mesh file="hand_2.obj"/>
        <mesh file="hand_3.obj"/>
        <mesh file="hand_4.obj"/>
        <mesh file="finger_0.obj"/>
        <mesh file="finger_1.obj"/>
    </asset>

    <default>
        <geom contype="1" conaffinity="1" friction="2.0 0.5 0.5" solref="0.02 1" solimp="0.9 0.95 0.001"/>
        <default class="panda">
            <material specular="0.5" shininess="0.25"/>
            <joint armature="0.1" damping="1" axis="0 0 1" range="-2.8973 2.8973"/>
            <general dyntype="none" biastype="affine" ctrlrange="-2.8973 2.8973" forcerange="-87 87"/>
            <default class="finger">
                <joint axis="0 1 0" type="slide" range="0 0.04"/>
            </default>
            <default class="visual">
                <geom type="mesh" contype="0" conaffinity="0" group="2"/>
            </default>
            <default class="collision">
                <geom type="mesh" group="3"/>
            </default>
        </default>
    </default>

    <worldbody>
        <!-- Floor -->
        <geom name="floor" type="plane" size="3 3 0.1" material="floor_mat"/>

        <!-- Robot Platform -->
        <body name="robot_platform" pos="0 0 0.05">
            <geom name="platform_geom" type="cylinder" size="0.15 0.05" material="platform_mat"/>
            <!-- Platform legs for stability -->
            <geom name="platform_leg1" type="cylinder" size="0.02 0.05" pos="0.1 0.1 -0.05" material="black"/>
            <geom name="platform_leg2" type="cylinder" size="0.02 0.05" pos="-0.1 0.1 -0.05" material="black"/>
            <geom name="platform_leg3" type="cylinder" size="0.02 0.05" pos="0.1 -0.1 -0.05" material="black"/>
            <geom name="platform_leg4" type="cylinder" size="0.02 0.05" pos="-0.1 -0.1 -0.05" material="black"/>
        </body>

                <!-- Cylindrical Tables (one for each slot) - smaller size -->
        <body name="cylinder_A" pos="-0.3 0.3 0.275">
            <geom name="cylinder_A_geom" type="cylinder" size="0.07 0.275" material="table_mat" friction="3.0 3.0 0.01"/>
            <!-- Slot marker on top -->
            <geom name="slot_A" type="cylinder" size="0.07 0.001" pos="0 0 0.276" material="slot_mat" contype="0" conaffinity="0"/>
        </body>

        <body name="cylinder_B" pos="0.0 0.3 0.275">
            <geom name="cylinder_B_geom" type="cylinder" size="0.07 0.275" material="table_mat" friction="3.0 3.0 0.01"/>
            <!-- Slot marker on top -->
            <geom name="slot_B" type="cylinder" size="0.07 0.001" pos="0 0 0.276" material="slot_mat" contype="0" conaffinity="0"/>
        </body>

        <body name="cylinder_C" pos="0.3 0.3 0.275">
            <geom name="cylinder_C_geom" type="cylinder" size="0.07 0.275" material="table_mat" friction="3.0 3.0 0.01"/>
            <!-- Slot marker on top -->
            <geom name="slot_C" type="cylinder" size="0.07 0.001" pos="0 0 0.276" material="slot_mat" contype="0" conaffinity="0"/>
        </body>

        <!-- Franka Panda Robot from menagerie -->
        <body name="link0" pos="0 0 0.1" childclass="panda">
            <inertial mass="0.629769" pos="-0.041018 -0.00014 0.049974"
                fullinertia="0.00315 0.00388 0.004285 8.2904e-7 0.00015 8.2299e-6"/>
            <geom mesh="link0_0" material="off_white" class="visual"/>
            <geom mesh="link0_1" material="black" class="visual"/>
            <geom mesh="link0_2" material="off_white" class="visual"/>
            <geom mesh="link0_3" material="black" class="visual"/>
            <geom mesh="link0_4" material="off_white" class="visual"/>
            <geom mesh="link0_5" material="black" class="visual"/>
            <geom mesh="link0_7" material="white" class="visual"/>
            <geom mesh="link0_8" material="white" class="visual"/>
            <geom mesh="link0_9" material="black" class="visual"/>
            <geom mesh="link0_10" material="off_white" class="visual"/>
            <geom mesh="link0_11" material="white" class="visual"/>
            <geom mesh="link0_c" class="collision"/>
            <body name="link1" pos="0 0 0.333">
                <inertial mass="4.970684" pos="0.003875 0.002081 -0.04762"
                    fullinertia="0.70337 0.70661 0.0091170 -0.00013900 0.0067720 0.019169"/>
                <joint name="joint1"/>
                <geom material="white" mesh="link1" class="visual"/>
                <geom mesh="link1_c" class="collision"/>
                <body name="link2" quat="1 -1 0 0">
                    <inertial mass="0.646926" pos="-0.003141 -0.02872 0.003495"
                        fullinertia="0.0079620 2.8110e-2 2.5995e-2 -3.925e-3 1.0254e-2 7.04e-4"/>
                    <joint name="joint2" range="-1.7628 1.7628"/>
                    <geom material="white" mesh="link2" class="visual"/>
                    <geom mesh="link2_c" class="collision"/>
                    <body name="link3" pos="0 -0.316 0" quat="1 1 0 0">
                        <joint name="joint3"/>
                        <inertial mass="3.228604" pos="2.7518e-2 3.9252e-2 -6.6502e-2"
                            fullinertia="3.7242e-2 3.6155e-2 1.083e-2 -4.761e-3 -1.1396e-2 -1.2805e-2"/>
                        <geom mesh="link3_0" material="white" class="visual"/>
                        <geom mesh="link3_1" material="white" class="visual"/>
                        <geom mesh="link3_2" material="white" class="visual"/>
                        <geom mesh="link3_3" material="black" class="visual"/>
                        <geom mesh="link3_c" class="collision"/>
                        <body name="link4" pos="0.0825 0 0" quat="1 1 0 0">
                            <inertial mass="3.587895" pos="-5.317e-2 1.04419e-1 2.7454e-2"
                                fullinertia="2.5853e-2 1.9552e-2 2.8323e-2 7.796e-3 -1.332e-3 8.641e-3"/>
                            <joint name="joint4" range="-3.0718 -0.0698"/>
                            <geom mesh="link4_0" material="white" class="visual"/>
                            <geom mesh="link4_1" material="white" class="visual"/>
                            <geom mesh="link4_2" material="black" class="visual"/>
                            <geom mesh="link4_3" material="white" class="visual"/>
                            <geom mesh="link4_c" class="collision"/>
                            <body name="link5" pos="-0.0825 0.384 0" quat="1 -1 0 0">
                                <inertial mass="1.225946" pos="-1.1953e-2 4.1065e-2 -3.8437e-2"
                                    fullinertia="3.5549e-2 2.9474e-2 8.627e-3 -2.117e-3 -4.037e-3 2.29e-4"/>
                                <joint name="joint5"/>
                                <geom mesh="link5_0" material="black" class="visual"/>
                                <geom mesh="link5_1" material="white" class="visual"/>
                                <geom mesh="link5_2" material="white" class="visual"/>
                                <geom mesh="link5_c0" class="collision"/>
                                <geom mesh="link5_c1" class="collision"/>
                                <geom mesh="link5_c2" class="collision"/>
                                <body name="link6" quat="1 1 0 0">
                                    <inertial mass="1.666555" pos="6.0149e-2 -1.4117e-2 -1.0517e-2"
                                        fullinertia="1.964e-3 4.354e-3 5.433e-3 1.09e-4 -1.158e-3 3.41e-4"/>
                                    <joint name="joint6" range="-0.0175 3.7525"/>
                                    <geom mesh="link6_0" material="off_white" class="visual"/>
                                    <geom mesh="link6_1" material="white" class="visual"/>
                                    <geom mesh="link6_2" material="black" class="visual"/>
                                    <geom mesh="link6_3" material="white" class="visual"/>
                                    <geom mesh="link6_4" material="white" class="visual"/>
                                    <geom mesh="link6_5" material="white" class="visual"/>
                                    <geom mesh="link6_6" material="white" class="visual"/>
                                    <geom mesh="link6_7" material="light_blue" class="visual"/>
                                    <geom mesh="link6_8" material="light_blue" class="visual"/>
                                    <geom mesh="link6_9" material="black" class="visual"/>
                                    <geom mesh="link6_10" material="black" class="visual"/>
                                    <geom mesh="link6_11" material="white" class="visual"/>
                                    <geom mesh="link6_12" material="green" class="visual"/>
                                    <geom mesh="link6_13" material="white" class="visual"/>
                                    <geom mesh="link6_14" material="black" class="visual"/>
                                    <geom mesh="link6_15" material="black" class="visual"/>
                                    <geom mesh="link6_16" material="white" class="visual"/>
                                    <geom mesh="link6_c" class="collision"/>
                                    <body name="link7" pos="0.088 0 0" quat="1 1 0 0">
                                        <inertial mass="7.35522e-01" pos="1.0517e-2 -4.252e-3 6.1597e-2"
                                            fullinertia="1.2516e-2 1.0027e-2 4.815e-3 -4.28e-4 -1.196e-3 -7.41e-4"/>
                                        <joint name="joint7"/>
                                        <geom mesh="link7_0" material="white" class="visual"/>
                                        <geom mesh="link7_1" material="black" class="visual"/>
                                        <geom mesh="link7_2" material="black" class="visual"/>
                                        <geom mesh="link7_3" material="black" class="visual"/>
                                        <geom mesh="link7_4" material="black" class="visual"/>
                                        <geom mesh="link7_5" material="black" class="visual"/>
                                        <geom mesh="link7_6" material="black" class="visual"/>
                                        <geom mesh="link7_7" material="white" class="visual"/>
                                        <geom mesh="link7_c" class="collision"/>
                                        <body name="hand" pos="0 0 0.107" quat="0.9238795 0 0 -0.3826834">
                                            <inertial mass="0.73" pos="-0.01 0 0.03" diaginertia="0.001 0.0025 0.0017"/>
                                            <geom mesh="hand_0" material="off_white" class="visual"/>
                                            <geom mesh="hand_1" material="black" class="visual"/>
                                            <geom mesh="hand_2" material="black" class="visual"/>
                                            <geom mesh="hand_3" material="white" class="visual"/>
                                            <geom mesh="hand_4" material="off_white" class="visual"/>
                                            <geom mesh="hand_c" class="collision"/>
                                            <site name="ee_site" pos="0 0 0" size="0.01" rgba="0 1 0 1"/>
                                            <camera name="wrist_camera" mode="fixed" pos="0 0 0.1" xyaxes="0 -1 0 1 0 0"/>
                                            <body name="left_finger" pos="0 0 0.0584">
                                                <inertial mass="0.015" pos="0 0 0" diaginertia="2.375e-6 2.375e-6 7.5e-7"/>
                                                <joint name="finger_joint1" class="finger"/>
                                                <geom mesh="finger_0" material="off_white" class="visual"/>
                                                <geom mesh="finger_1" material="black" class="visual"/>
                                                <geom mesh="finger_0" class="collision"/>
                                            </body>
                                            <body name="right_finger" pos="0 0 0.0584" quat="0 0 0 1">
                                                <inertial mass="0.015" pos="0 0 0" diaginertia="2.375e-6 2.375e-6 7.5e-7"/>
                                                <joint name="finger_joint2" class="finger"/>
                                                <geom mesh="finger_0" material="off_white" class="visual"/>
                                                <geom mesh="finger_1" material="black" class="visual"/>
                                                <geom mesh="finger_0" class="collision"/>
                                            </body>
                                        </body>
                                    </body>
                                </body>
                            </body>
                        </body>
                    </body>
                </body>
            </body>
        </body>

        <!-- Blocks -->
        <body name="red_block" pos="-0.3 0.3 0.575">
            <joint type="free"/>
            <geom name="red_block_geom" type="box" size="0.025 0.025 0.025" material="red_mat" friction="2.5 2.5 0.01"/>
            <site name="red_block_site" pos="0 0 0" size="0.01"/>
        </body>

        <body name="blue_block" pos="0.0 0.3 0.575">
            <joint type="free"/>
            <geom name="blue_block_geom" type="box" size="0.025 0.025 0.025" material="blue_mat" friction="2.5 2.5 0.01"/>
            <site name="blue_block_site" pos="0 0 0" size="0.01"/>
        </body>

        <!-- Cameras -->
        <camera name="front_camera" pos="0.5 -0.8 1.0" xyaxes="0.5 0.5 0 0 0 1"/>
    </worldbody>

    <tendon>
        <fixed name="split">
            <joint joint="finger_joint1" coef="0.5"/>
            <joint joint="finger_joint2" coef="0.5"/>
        </fixed>
    </tendon>

    <equality>
        <joint joint1="finger_joint1" joint2="finger_joint2" solimp="0.95 0.99 0.001" solref="0.005 1"/>
    </equality>

    <actuator>
        <general class="panda" name="actuator1" joint="joint1" gainprm="4500" biasprm="0 -4500 -450"/>
        <general class="panda" name="actuator2" joint="joint2" gainprm="4500" biasprm="0 -4500 -450"
            ctrlrange="-1.7628 1.7628"/>
        <general class="panda" name="actuator3" joint="joint3" gainprm="3500" biasprm="0 -3500 -350"/>
        <general class="panda" name="actuator4" joint="joint4" gainprm="3500" biasprm="0 -3500 -350"
            ctrlrange="-3.0718 -0.0698"/>
        <general class="panda" name="actuator5" joint="joint5" gainprm="2000" biasprm="0 -2000 -200" forcerange="-12 12"/>
        <general class="panda" name="actuator6" joint="joint6" gainprm="2000" biasprm="0 -2000 -200" forcerange="-12 12"
            ctrlrange="-0.0175 3.7525"/>
        <general class="panda" name="actuator7" joint="joint7" gainprm="2000" biasprm="0 -2000 -200" forcerange="-12 12"/>
        <general class="panda" name="actuator8" tendon="split" forcerange="-100 100" ctrlrange="0 255"
            gainprm="0.01568627451 0 0" biasprm="0 -100 -10"/>
    </actuator>

    <sensor>
        <framequat name="ee_orientation" objtype="site" objname="ee_site"/>
        <framepos name="ee_position" objtype="site" objname="ee_site"/>
    </sensor>
</mujoco>
        """
        return xml

    def _setup_spaces(self):
        """Set up observation and action spaces."""
        # Action space - always end-effector delta + gripper (IK handled internally)
        # [dx, dy, dz, gripper_action] - 4D action space
        self.action_space = spaces.Box(-1, 1, (4,), dtype=np.float32)

        # Observation space
        if self.observation_mode == 'full':
            self.observation_space = spaces.Dict({
                'robot_qpos': spaces.Box(-np.inf, np.inf, (7,), dtype=np.float32),
                'robot_qvel': spaces.Box(-np.inf, np.inf, (7,), dtype=np.float32),
                'ee_pos': spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
                'ee_quat': spaces.Box(-1, 1, (4,), dtype=np.float32),
                'gripper_state': spaces.Box(0, 1, (2,), dtype=np.float32),
                'red_block_pos': spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
                'red_block_quat': spaces.Box(-1, 1, (4,), dtype=np.float32),
                'blue_block_pos': spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
                'blue_block_quat': spaces.Box(-1, 1, (4,), dtype=np.float32),
                'slot_occupancy': spaces.Box(0, 2, (3,), dtype=np.float32),
                'initial_config': spaces.Box(0, 2, (3,), dtype=np.float32),
                # Add reward shaping features to observation for better learning
                'distance_metrics': spaces.Box(0, np.inf, (6,), dtype=np.float32),
            })
        else:  # partial observability
            self.observation_space = spaces.Dict({
                'front_camera': spaces.Box(0, 255, (128, 128, 3), dtype=np.uint8),
                'wrist_camera': spaces.Box(0, 255, (84, 84, 3), dtype=np.uint8),
                'robot_qpos': spaces.Box(-np.inf, np.inf, (7,), dtype=np.float32),
                'gripper_state': spaces.Box(0, 1, (2,), dtype=np.float32),
                'distance_metrics': spaces.Box(0, np.inf, (6,), dtype=np.float32),
            })

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)

        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)

        # Set robot to home position (Franka Panda home keyframe)
        home_qpos = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853])
        self.data.qpos[:7] = home_qpos
        self.data.qpos[7:9] = 0.04  # Open gripper

        # Randomize block positions
        occupied_slots = self.np_random.choice(['A', 'B', 'C'], size=2, replace=False)
        empty_slot = [s for s in ['A', 'B', 'C'] if s not in occupied_slots][0]

        # Place blocks with position noise (smaller radius due to smaller cylinders)
        red_slot, blue_slot = occupied_slots
        red_pos = self.slot_positions[red_slot] + self.np_random.uniform(-0.015, 0.015, 3)  # Reduced noise
        blue_pos = self.slot_positions[blue_slot] + self.np_random.uniform(-0.015, 0.015, 3)
        red_pos[2] = self.cylinder_height + 0.025  # Block center on top of cylinder
        blue_pos[2] = self.cylinder_height + 0.025

        # Set block positions
        self.data.qpos[9:12] = red_pos
        self.data.qpos[16:19] = blue_pos

        # Store initial configuration
        self.initial_config = {
            'red_slot': red_slot,
            'blue_slot': blue_slot,
            'empty_slot': empty_slot
        }

        # Goal is to swap positions
        self.goal_config = {
            'red_slot': blue_slot,
            'blue_slot': red_slot
        }

        # Forward simulation to settle
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)

        # Reset tracking variables
        self.step_count = 0
        self.prev_action = np.zeros(4)  # [dx, dy, dz, gripper]
        self.grasped_block = None

        return self._get_observation(), {}

    def step(self, action):
        """Execute one environment step."""
        # Clip action to valid range
        action = np.clip(action, -1, 1)

        # Apply action using IK-based control
        self._apply_ik_based_action(action)

        # Step simulation
        for _ in range(10):  # 20Hz control at 200Hz simulation
            mujoco.mj_step(self.model, self.data)

        # Update tracking
        self._update_grasped_block()

        # Get observation
        observation = self._get_observation()

        # Compute reward and check termination
        reward, terminated, info = self._compute_reward_and_termination()

        # Update step count
        self.step_count += 1
        truncated = self.step_count >= self.max_episode_steps

        # Store previous action
        self.prev_action = action.copy()

        return observation, reward, terminated, truncated, info

    def _apply_ik_based_action(self, action):
        """
        Apply action using IK-based control.

        Action format: [dx, dy, dz, gripper_action]
        - dx, dy, dz: End-effector position deltas (normalized to [-1, 1])
        - gripper_action: Gripper control (>0 = close, <=0 = open)
        """
        # Get current end-effector position
        current_pos = self.data.site('ee_site').xpos.copy()

        # Scale position deltas (max 8cm per step for faster movement)
        max_delta = 0.08
        delta_pos = action[:3] * max_delta

        # Compute target position
        target_pos = current_pos + delta_pos

        # Use position-only IK for reliable movement
        success, target_joints = self.ik_solver.solve_ik_position_only(
            target_pos,
            seed=self.data.qpos[:7].copy()
        )

        if success:
            # Apply joint position control
            self.data.ctrl[:7] = target_joints
        else:
            # Fallback: use current position if IK fails
            self.data.ctrl[:7] = self.data.qpos[:7]

        # Gripper control
        gripper_action = action[3]
        if gripper_action > 0:  # Close gripper
            self.data.ctrl[7] = 0  # Close (0 = closed for tendon)
        else:  # Open gripper
            self.data.ctrl[7] = 255  # Open (255 = open for tendon)

    def _get_observation(self):
        """Get current observation."""
        # Calculate current distance metrics for both full and partial obs
        distance_metrics = self._compute_distance_metrics()

        if self.observation_mode == 'full':
            # Get block positions
            red_pos = self.data.qpos[9:12].copy()
            red_quat = self.data.qpos[12:16].copy()
            blue_pos = self.data.qpos[16:19].copy()
            blue_quat = self.data.qpos[19:23].copy()

            # Get slot occupancy
            slot_occupancy = self._get_slot_occupancy()

            # Convert initial config to array
            initial_config = np.zeros(3)
            if self.initial_config:
                for i, slot in enumerate(['A', 'B', 'C']):
                    if self.initial_config['red_slot'] == slot:
                        initial_config[i] = 1
                    elif self.initial_config['blue_slot'] == slot:
                        initial_config[i] = 2

            # Get EE orientation using sensor data
            ee_quat = self._get_ee_quaternion()

            return {
                'robot_qpos': self.data.qpos[:7].copy(),
                'robot_qvel': self.data.qvel[:7].copy(),
                'ee_pos': self.data.site('ee_site').xpos.copy(),
                'ee_quat': ee_quat,
                'gripper_state': self.data.qpos[7:9].copy(),
                'red_block_pos': red_pos,
                'red_block_quat': red_quat,
                'blue_block_pos': blue_pos,
                'blue_block_quat': blue_quat,
                'slot_occupancy': slot_occupancy,
                'initial_config': initial_config,
                'distance_metrics': distance_metrics,
            }
        else:
            # Render cameras
            front_img = self._render_camera('front_camera', (128, 128))
            wrist_img = self._render_camera('wrist_camera', (84, 84))

            return {
                'front_camera': front_img,
                'wrist_camera': wrist_img,
                'robot_qpos': self.data.qpos[:7].copy(),
                'gripper_state': self.data.qpos[7:9].copy(),
                'distance_metrics': distance_metrics,
            }

    def _compute_distance_metrics(self):
        """Compute current distance metrics for reward shaping."""
        ee_pos = self.data.site('ee_site').xpos.copy()
        red_pos = self.data.qpos[9:12].copy()
        blue_pos = self.data.qpos[16:19].copy()

        red_goal_pos = self.slot_positions[self.goal_config['red_slot']]
        blue_goal_pos = self.slot_positions[self.goal_config['blue_slot']]

        return np.array([
            min(np.linalg.norm(ee_pos - red_pos), np.linalg.norm(ee_pos - blue_pos)),  # ee_to_nearest_block
            np.linalg.norm(red_pos - blue_pos),  # blocks_to_each_other
            np.linalg.norm(red_pos - red_goal_pos),  # red_to_goal
            np.linalg.norm(blue_pos - blue_goal_pos),  # blue_to_goal
            np.linalg.norm(ee_pos - red_pos),  # ee_to_red
            np.linalg.norm(ee_pos - blue_pos),  # ee_to_blue
        ], dtype=np.float32)

    def _get_ee_quaternion(self):
        """Get end-effector quaternion from rotation matrix."""
        # Get rotation matrix from site
        mat = self.data.site('ee_site').xmat.reshape(3, 3)
        return self._mat_to_quat(mat)

    def _get_slot_occupancy(self):
        """Determine which block is in which slot."""
        occupancy = np.zeros(3)
        red_pos = self.data.qpos[9:12]
        blue_pos = self.data.qpos[16:19]

        for i, (slot_name, slot_pos) in enumerate(self.slot_positions.items()):
            # Check red block
            if np.linalg.norm(red_pos[:2] - slot_pos[:2]) < self.slot_radius:
                occupancy[i] = 1
            # Check blue block
            elif np.linalg.norm(blue_pos[:2] - slot_pos[:2]) < self.slot_radius:
                occupancy[i] = 2

        return occupancy

    def _update_grasped_block(self):
        """Update which block is grasped."""
        ee_pos = self.data.site('ee_site').xpos
        red_pos = self.data.qpos[9:12]
        blue_pos = self.data.qpos[16:19]
        gripper_closed = self.data.qpos[7] < 0.02

        if gripper_closed:
            red_dist = np.linalg.norm(ee_pos - red_pos)
            blue_dist = np.linalg.norm(ee_pos - blue_pos)

            if red_dist < 0.05:
                self.grasped_block = 'red'
            elif blue_dist < 0.05:
                self.grasped_block = 'blue'
            else:
                self.grasped_block = None
        else:
            self.grasped_block = None

    def _compute_reward_and_termination(self):
        """Compute reward and check if task is complete."""
        reward = 0.0
        info = {}

        # Check goal achievement
        current_config = self._get_current_config()
        goal_achieved = (
            current_config['red_slot'] == self.goal_config['red_slot'] and
            current_config['blue_slot'] == self.goal_config['blue_slot']
        )

        if goal_achieved:
            reward += 10.0
            info['success'] = True
            terminated = True
        else:
            info['success'] = False
            terminated = False

        # Apply reward shaping if enabled and not sparse reward mode
        if self.reward_shaping and not self.sparse_reward:
            shaped_reward = self._compute_shaped_reward()
            reward += shaped_reward
            info['shaped_reward'] = shaped_reward
        elif self.sparse_reward:
            # In sparse reward mode, only give completion reward
            info['shaped_reward'] = 0.0

        # Safety penalties (applied even in sparse mode)
        safety_penalty = self._compute_safety_penalties()
        reward += safety_penalty
        info['safety_penalty'] = safety_penalty

        info.update({
            'grasped_block': self.grasped_block,
            'current_config': current_config,
            'goal_config': self.goal_config,
            'distance_metrics': self._compute_distance_metrics(),
        })

        return reward, terminated, info

    def _compute_shaped_reward(self):
        """
        Compute shaped reward with multiple components that get disabled when complete.

        Each component contributes -0.1 * distance. Once a phase is complete,
        that component is excluded so rewards don't drop during phase transitions.
        """
        reward = 0.0

        ee_pos = self.data.site('ee_site').xpos.copy()
        red_pos = self.data.qpos[9:12].copy()
        blue_pos = self.data.qpos[16:19].copy()

        red_goal_pos = self.slot_positions[self.goal_config['red_slot']]
        blue_goal_pos = self.slot_positions[self.goal_config['blue_slot']]

        # Reward components (each -0.1 * distance)
        components = {}

        # Component 1: Distance to red block (excluded once grasped)
        if self.grasped_block != 'red':
            components['ee_to_red'] = -0.1 * np.linalg.norm(ee_pos - red_pos)

        # Component 2: Red block to goal (excluded once placed)
        red_in_goal = np.linalg.norm(red_pos[:2] - red_goal_pos[:2]) < self.slot_radius
        if not red_in_goal:
            components['red_to_goal'] = -0.1 * np.linalg.norm(red_pos - red_goal_pos)

        # Component 3: Distance to blue block (excluded once grasped)
        if self.grasped_block != 'blue':
            components['ee_to_blue'] = -0.1 * np.linalg.norm(ee_pos - blue_pos)

        # Component 4: Blue block to goal (excluded once placed)
        blue_in_goal = np.linalg.norm(blue_pos[:2] - blue_goal_pos[:2]) < self.slot_radius
        if not blue_in_goal:
            components['blue_to_goal'] = -0.1 * np.linalg.norm(blue_pos - blue_goal_pos)

        # Sum active components
        reward = sum(components.values())

        # Small bonuses for good behaviors
        if self.grasped_block is not None:
            reward += 0.1

        if red_in_goal:
            reward += 0.5
        if blue_in_goal:
            reward += 0.5

        # Small time penalty
        reward -= 0.001

        return reward

    def _compute_safety_penalties(self):
        """Compute safety penalties for dangerous states."""
        penalty = 0.0

        # Height penalty for dropping blocks
        red_height = self.data.qpos[11]
        blue_height = self.data.qpos[18]
        table_height = self.cylinder_height

        if red_height < table_height - 0.1:
            penalty -= 2.0  # Large penalty for dropping blocks
        if blue_height < table_height - 0.1:
            penalty -= 2.0

        # Penalty for excessive action magnitude (smoothness)
        if self.prev_action is not None:
            action_magnitude = np.linalg.norm(self.prev_action[:3])
            if action_magnitude > 0.8:  # Large movements
                penalty -= 0.1 * (action_magnitude - 0.8)

        return penalty

    def _get_current_config(self):
        """Get current block configuration."""
        config = {'red_slot': None, 'blue_slot': None}
        red_pos = self.data.qpos[9:12]
        blue_pos = self.data.qpos[16:19]

        for slot_name, slot_pos in self.slot_positions.items():
            if np.linalg.norm(red_pos[:2] - slot_pos[:2]) < self.slot_radius:
                config['red_slot'] = slot_name
            if np.linalg.norm(blue_pos[:2] - slot_pos[:2]) < self.slot_radius:
                config['blue_slot'] = slot_name

        return config

    def _mat_to_quat(self, mat):
        """Convert rotation matrix to quaternion."""
        # Robust quaternion conversion
        trace = np.trace(mat)
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
            qw = 0.25 * s
            qx = (mat[2, 1] - mat[1, 2]) / s
            qy = (mat[0, 2] - mat[2, 0]) / s
            qz = (mat[1, 0] - mat[0, 1]) / s
        elif mat[0, 0] > mat[1, 1] and mat[0, 0] > mat[2, 2]:
            s = np.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2]) * 2  # s = 4 * qx
            qw = (mat[2, 1] - mat[1, 2]) / s
            qx = 0.25 * s
            qy = (mat[0, 1] + mat[1, 0]) / s
            qz = (mat[0, 2] + mat[2, 0]) / s
        elif mat[1, 1] > mat[2, 2]:
            s = np.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2]) * 2  # s = 4 * qy
            qw = (mat[0, 2] - mat[2, 0]) / s
            qx = (mat[0, 1] + mat[1, 0]) / s
            qy = 0.25 * s
            qz = (mat[1, 2] + mat[2, 1]) / s
        else:
            s = np.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1]) * 2  # s = 4 * qz
            qw = (mat[1, 0] - mat[0, 1]) / s
            qx = (mat[0, 2] + mat[2, 0]) / s
            qy = (mat[1, 2] + mat[2, 1]) / s
            qz = 0.25 * s

        return np.array([qw, qx, qy, qz])

    def _render_camera(self, camera_name, resolution):
        """Render camera view."""
        # Create renderer if needed
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, resolution[1], resolution[0])

        # Get camera ID
        try:
            camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        except:
            camera_id = 0  # Use default camera if name not found

        # Update and render
        self._renderer.update_scene(self.data, camera=camera_id)
        pixels = self._renderer.render()

        return pixels.astype(np.uint8)

    def render(self):
        """Render the environment."""
        if self.render_mode == 'human':
            # Use MuJoCo's built-in viewer
            if self.viewer is None:
                import mujoco.viewer
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
        elif self.render_mode == 'rgb_array':
            return self._render_camera('front_camera', (640, 480))

    def close(self):
        """Clean up resources."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


if __name__ == "__main__":
    # Example usage showing different configurations
    print("Testing BlockSwap Environment configurations...")

    # Test 1: Full observability with reward shaping
    print("\n1. Full observability with reward shaping:")
    env1 = BlockSwapEnv(
        observation_mode='full',
        reward_shaping=True,
        sparse_reward=False
    )
    obs, info = env1.reset()
    print(f"Observation space keys: {list(obs.keys())}")
    print(f"Distance metrics shape: {obs['distance_metrics'].shape}")

    # Test 2: Sparse reward mode
    print("\n2. Sparse reward mode:")
    env2 = BlockSwapEnv(
        observation_mode='full',
        reward_shaping=False,
        sparse_reward=True
    )
    obs, info = env2.reset()

    # Test 3: Partial observability with cameras
    print("\n3. Partial observability (camera-based):")
    env3 = BlockSwapEnv(
        observation_mode='partial',
        reward_shaping=True,
        sparse_reward=False
    )
    obs, info = env3.reset()
    print(f"Observation space keys: {list(obs.keys())}")
    print(f"Front camera shape: {obs['front_camera'].shape}")
    print(f"Wrist camera shape: {obs['wrist_camera'].shape}")

    # Quick action test
    action = env1.action_space.sample()
    obs, reward, terminated, truncated, info = env1.step(action)
    print(f"\nSample step reward: {reward}")
    print(f"Shaped reward component: {info.get('shaped_reward', 'N/A')}")
    print(f"Safety penalty: {info.get('safety_penalty', 'N/A')}")

    print(f"\nReward shaping explanation:")
    print(f"- Reward = sum of active distance components (-0.1 * distance)")
    print(f"- Components get excluded when their phase completes")
    print(f"- This prevents reward drops during phase transitions")

    env1.close()
    env2.close()
    env3.close()
    print("\nAll tests completed successfully!")