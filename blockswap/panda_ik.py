"""
Standalone Panda IK solver adapted from panda_robot repository.
Removes ROS dependencies and adapts for use with MuJoCo simulation.

Original work by Saif Sidhik <sxs1412@bham.ac.uk>
Adapted for MuJoCo by removing ROS dependencies and using XML-based URDF loading.
"""

import numpy as np
import mujoco
from typing import Optional, Tuple, Union
import os


class PandaIK:
    """
    Standalone Panda inverse kinematics solver using analytical approach.
    
    This implementation provides fast, analytical IK for the Franka Panda robot
    without requiring PyKDL or ROS dependencies.
    """
    
    def __init__(self, model, data):
        """
        Initialize the Panda IK solver.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
        """
        self.model = model
        self.data = data
        
        # Joint limits for Panda (radians)
        self.joint_limits = np.array([
            [-2.8973, 2.8973],   # joint1
            [-1.7628, 1.7628],   # joint2  
            [-2.8973, 2.8973],   # joint3
            [-3.0718, -0.0698],  # joint4
            [-2.8973, 2.8973],   # joint5
            [-0.0175, 3.7525],   # joint6
            [-2.8973, 2.8973]    # joint7
        ])
        
        # DH parameters for Franka Panda (from the URDF)
        self.d = np.array([0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.107])
        self.a = np.array([0.0, 0.0, 0.0, 0.0825, -0.0825, 0.0, 0.088])
        self.alpha = np.array([0.0, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2, np.pi/2])
        
        # End-effector site name
        self.ee_site_name = 'ee_site'
        
        # IK tolerances - relaxed for better convergence
        self.position_tolerance = 1e-3
        self.orientation_tolerance = 1e-2
        self.max_iterations = 50
        
        # Desired world direction for the gripper Z axis (+Z of hand frame)
        self.WORLD_DOWN = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        
    def forward_kinematics(self, joint_angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute forward kinematics for given joint angles.
        
        Args:
            joint_angles: 7-DOF joint angles
            
        Returns:
            Tuple of (position, quaternion) of end-effector
        """
        # Create a temporary data instance to avoid modifying the main simulation state
        temp_data = mujoco.MjData(self.model)
        temp_data.qpos[:] = self.data.qpos[:]
        temp_data.qvel[:] = self.data.qvel[:]
        
        # Set joint angles and compute forward kinematics
        temp_data.qpos[:7] = joint_angles
        mujoco.mj_fwdPosition(self.model, temp_data)
        
        # Get end-effector pose
        ee_pos = temp_data.site(self.ee_site_name).xpos.copy()
        ee_mat = temp_data.site(self.ee_site_name).xmat.reshape(3, 3)
        ee_quat = self._mat_to_quat(ee_mat)
        
        return ee_pos, ee_quat
        
    def inverse_kinematics(self, target_pos: np.ndarray, target_quat: Optional[np.ndarray] = None,
                          seed: Optional[np.ndarray] = None, position_only: bool = False) -> Tuple[bool, np.ndarray]:
        """
        Solve inverse kinematics using Newton-Raphson method.
        
        Args:
            target_pos: Target end-effector position [x, y, z]
            target_quat: Target end-effector quaternion [w, x, y, z] (optional)
            seed: Initial joint configuration (optional)
            position_only: If True, only solve for position (3-DOF)
            
        Returns:
            Tuple of (success, joint_angles)
        """
        if seed is None:
            q = self.data.qpos[:7].copy()
        else:
            q = seed.copy()
            
        # Clamp to joint limits
        q = self._clamp_joints(q)
        
        for iteration in range(self.max_iterations):
            # Compute current end-effector pose
            current_pos, current_quat = self.forward_kinematics(q)
            
            # Compute position error
            pos_error = target_pos - current_pos
            
            # Check position convergence
            if np.linalg.norm(pos_error) < self.position_tolerance:
                if position_only or target_quat is None:
                    return True, q
                    
                # Check orientation convergence if needed
                orient_error = self._orientation_error(current_quat, target_quat)
                if np.linalg.norm(orient_error) < self.orientation_tolerance:
                    return True, q
                    
                # Construct full error vector
                error = np.concatenate([pos_error, orient_error])
            else:
                if position_only or target_quat is None:
                    error = pos_error
                else:
                    orient_error = self._orientation_error(current_quat, target_quat)
                    error = np.concatenate([pos_error, orient_error])
            
            # Compute Jacobian
            jacobian = self._compute_jacobian(q, full_jacobian=(not position_only and target_quat is not None))
            
            # Solve for joint velocity using damped least squares
            damping = 1e-3  # Increased damping for stability
            if jacobian.shape[0] == 3:  # Position only
                J_inv = jacobian.T @ np.linalg.inv(jacobian @ jacobian.T + damping * np.eye(3))
            else:  # Full 6-DOF
                J_inv = jacobian.T @ np.linalg.inv(jacobian @ jacobian.T + damping * np.eye(6))
                
            delta_q = J_inv @ error
            
            # Adaptive step size based on error magnitude
            error_norm = np.linalg.norm(error)
            if error_norm > 0.1:
                step_size = 0.2  # Smaller steps for large errors
            elif error_norm > 0.01:
                step_size = 0.5  # Medium steps
            else:
                step_size = 0.8  # Larger steps for small errors
                
            q = q + step_size * delta_q
            q = self._clamp_joints(q)
            
        return False, q
        
    def _compute_jacobian(self, joint_angles: np.ndarray, full_jacobian: bool = True) -> np.ndarray:
        """
        Compute the Jacobian matrix using MuJoCo's built-in functions.
        
        Args:
            joint_angles: Current joint configuration
            full_jacobian: If True, return 6x7 Jacobian (pos+rot), else 3x7 (pos only)
            
        Returns:
            Jacobian matrix
        """
        # Create a temporary data instance to avoid modifying the main simulation state
        temp_data = mujoco.MjData(self.model)
        temp_data.qpos[:] = self.data.qpos[:]
        temp_data.qvel[:] = self.data.qvel[:]
        
        # Set joint angles
        temp_data.qpos[:7] = joint_angles
        mujoco.mj_fwdPosition(self.model, temp_data)
        
        # Compute Jacobian
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        
        site_id = self.model.site(self.ee_site_name).id
        mujoco.mj_jacSite(self.model, temp_data, jacp, jacr, site_id)
        
        if full_jacobian:
            return np.vstack([jacp[:, :7], jacr[:, :7]])
        else:
            return jacp[:, :7]
            
    def _orientation_error(self, current_quat: np.ndarray, target_quat: np.ndarray) -> np.ndarray:
        """
        Compute orientation error in axis-angle representation.
        
        Args:
            current_quat: Current quaternion [w, x, y, z]
            target_quat: Target quaternion [w, x, y, z]
            
        Returns:
            Orientation error as 3D vector
        """
        # Normalize quaternions
        current_quat = current_quat / np.linalg.norm(current_quat)
        target_quat = target_quat / np.linalg.norm(target_quat)
        
        # Compute quaternion error: q_error = q_target * q_current^-1
        current_quat_inv = np.array([current_quat[0], -current_quat[1], -current_quat[2], -current_quat[3]])
        
        # Quaternion multiplication
        qw = target_quat[0] * current_quat_inv[0] - target_quat[1] * current_quat_inv[1] - target_quat[2] * current_quat_inv[2] - target_quat[3] * current_quat_inv[3]
        qx = target_quat[0] * current_quat_inv[1] + target_quat[1] * current_quat_inv[0] + target_quat[2] * current_quat_inv[3] - target_quat[3] * current_quat_inv[2]
        qy = target_quat[0] * current_quat_inv[2] - target_quat[1] * current_quat_inv[3] + target_quat[2] * current_quat_inv[0] + target_quat[3] * current_quat_inv[1]
        qz = target_quat[0] * current_quat_inv[3] + target_quat[1] * current_quat_inv[2] - target_quat[2] * current_quat_inv[1] + target_quat[3] * current_quat_inv[0]
        
        q_error = np.array([qw, qx, qy, qz])
        
        # Ensure shortest path
        if q_error[0] < 0:
            q_error = -q_error
            
        # Convert to axis-angle
        if abs(q_error[0]) > 1.0:
            q_error[0] = np.sign(q_error[0])
            
        angle = 2.0 * np.arccos(abs(q_error[0]))
        if angle > 1e-6:
            axis = q_error[1:4] / np.sin(angle / 2.0)
            return axis * angle
        else:
            return np.zeros(3)
            
    def _clamp_joints(self, joint_angles: np.ndarray) -> np.ndarray:
        """Clamp joint angles to limits."""
        clamped = joint_angles.copy()
        for i in range(7):
            clamped[i] = np.clip(clamped[i], self.joint_limits[i, 0], self.joint_limits[i, 1])
        return clamped
        
    def _mat_to_quat(self, mat: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to quaternion [w, x, y, z]."""
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
        
    def _quat_to_mat(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix."""
        w, x, y, z = q
        return np.array([
            [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
            [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
            [2*x*z - 2*y*w,         2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
        ], dtype=np.float64)
        
    def _axis_error(
        self,
        current_R: np.ndarray,
        desired_axis_world: np.ndarray,
        body_axis_idx: int = 2,           # 0:+X, 1:+Y, 2:+Z
    ) -> np.ndarray:
        """Torque-like vector driving body-axis → desired world axis."""
        cur = current_R[:, body_axis_idx]
        cur /= np.linalg.norm(cur)
        des = desired_axis_world / np.linalg.norm(desired_axis_world)
        return np.cross(cur, des)        # 3-vector
        
    def solve_ik_position_only(self, target_pos: np.ndarray, seed: Optional[np.ndarray] = None) -> Tuple[bool, np.ndarray]:
        """
        Solve IK for position only (3-DOF problem for 7-DOF robot).
        
        Args:
            target_pos: Target position [x, y, z]
            seed: Initial joint configuration
            
        Returns:
            Tuple of (success, joint_angles)
        """
        return self.inverse_kinematics(target_pos, None, seed, position_only=True)
        
    def solve_ik_full(self, target_pos: np.ndarray, target_quat: np.ndarray, 
                     seed: Optional[np.ndarray] = None) -> Tuple[bool, np.ndarray]:
        """
        Solve IK for full 6-DOF pose.
        
        Args:
            target_pos: Target position [x, y, z]
            target_quat: Target quaternion [w, x, y, z]
            seed: Initial joint configuration
            
        Returns:
            Tuple of (success, joint_angles)
        """
        return self.inverse_kinematics(target_pos, target_quat, seed, position_only=False)
        
    def get_current_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current end-effector pose.
        
        Returns:
            Tuple of (position, quaternion)
        """
        return self.forward_kinematics(self.data.qpos[:7])
        
    def compute_cartesian_path(self, waypoints: list, target_orientations: Optional[list] = None,
                              max_step: float = 0.01) -> Tuple[bool, list]:
        """
        Plan a Cartesian path through waypoints.
        
        Args:
            waypoints: List of target positions
            target_orientations: List of target orientations (optional)
            max_step: Maximum step size between waypoints
            
        Returns:
            Tuple of (success, joint_trajectories)
        """
        if target_orientations is None:
            target_orientations = [None] * len(waypoints)
            
        trajectories = []
        current_q = self.data.qpos[:7].copy()
        
        for i, (waypoint, orientation) in enumerate(zip(waypoints, target_orientations)):
            # Interpolate if waypoint is far from current position
            current_pos, _ = self.forward_kinematics(current_q)
            distance = np.linalg.norm(waypoint - current_pos)
            
            if distance > max_step:
                # Create intermediate waypoints
                num_steps = int(np.ceil(distance / max_step))
                for step in range(1, num_steps + 1):
                    alpha = step / num_steps
                    intermediate_pos = current_pos + alpha * (waypoint - current_pos)
                    
                    if orientation is not None:
                        # Simple spherical interpolation for orientation
                        intermediate_quat = orientation  # Simplified - could use slerp
                    else:
                        intermediate_quat = None
                        
                    success, q_solution = self.inverse_kinematics(intermediate_pos, intermediate_quat, current_q)
                    
                    if not success:
                        print(f"IK failed for intermediate waypoint {step}/{num_steps} of waypoint {i}")
                        return False, trajectories
                        
                    trajectories.append(q_solution)
                    current_q = q_solution
            else:
                # Direct IK solve
                success, q_solution = self.inverse_kinematics(waypoint, orientation, current_q)
                
                if not success:
                    print(f"IK failed for waypoint {i}")
                    return False, trajectories
                    
                trajectories.append(q_solution)
                current_q = q_solution
                
        return True, trajectories
        
    def solve_ik_point_down(
        self,
        target_pos: np.ndarray,
        seed: np.ndarray | None = None,
        position_tolerance: float | None = None,
        orientation_tolerance: float | None = None,
        max_iterations: int | None = None,
    ) -> Tuple[bool, np.ndarray]:
        """Like solve_ik_full, but only constrains tool Z to world −Z."""
        pos_tol = position_tolerance or self.position_tolerance
        ang_tol = orientation_tolerance or self.orientation_tolerance
        iters   = max_iterations or self.max_iterations

        q = self.data.qpos[:7].copy() if seed is None else seed.copy()
        q = self._clamp_joints(q)

        for _ in range(iters):
            cur_pos, cur_quat = self.forward_kinematics(q)
            pos_err = target_pos - cur_pos

            # orientation error that ignores yaw
            cur_R      = self._quat_to_mat(cur_quat)
            orient_err = self._axis_error(cur_R, self.WORLD_DOWN)

            if (np.linalg.norm(pos_err) < pos_tol and
                    np.linalg.norm(orient_err) < ang_tol):
                return True, q

            # full 6×7 Jacobian (pos+rot)
            J = self._compute_jacobian(q, full_jacobian=True)
            # build error vector (size 6)
            err = np.concatenate([pos_err, orient_err])

            # damped least squares
            λ2 = 1e-4
            Jt = J.T
            delta_q = Jt @ np.linalg.inv(J @ Jt + λ2 * np.eye(6)) @ err

            # modest step size
            q = self._clamp_joints(q + 0.5 * delta_q)

        return False, q  # failed