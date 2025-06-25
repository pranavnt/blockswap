"""
Keyboard-teleoperation script for collecting BlockSwap demonstrations.

Now **macOS‑friendly**: the script automatically falls back to the cross‑platform
`pynput` backend when `keyboard` is unavailable or crashes on macOS.

=====================================================================
Usage (from project root):
---------------------------------------------------------------------
$ python -m blockswap.collect_demos \
        --episodes 10 \
        --out-dir demos/ \
        --obs-mode full

Controls (while viewer window is focused):
---------------------------------------------------------------------
  w / s               : +Y / –Y translation (forward / back)
  a / d               : –X / +X translation (left / right)
  r / f               : +Z / –Z translation (up / down)
  space               : toggle gripper (open ⇄ close)
  enter               : finish current episode & start a new one
  esc                 : abort recording immediately
  p                   : pause / un‑pause simulation (helpful for breaks)

Saved data format
---------------------------------------------------------------------
Each episode is stored as a compressed *.npz* file in *out‑dir* with arrays:
    observations  (T, obs_dim)  float32
    actions       (T, 4)        float32  (dx, dy, dz, gripper)
    rewards       (T,)          float32
    dones         (T,)          bool

Additional scalars (observation_mode, control_frequency, …) are embedded.
"""
from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from pathlib import Path
from typing import List, Tuple, Callable

import numpy as np

# ---------------------------------------------------------------------------
# Cross‑platform key‑state helper
# ---------------------------------------------------------------------------

# We expose a single callable `is_pressed(key:str) -> bool`
if platform.system() == "Darwin":
    # ---------------------------------------------------------------------
    # macOS path – use pynput (does NOT require root / Accessibility by default)
    # ---------------------------------------------------------------------
    try:
        from pynput import keyboard as _pyn_kb  # type: ignore
    except ImportError as e:  # pragma: no cover
        print("Missing dependency: pip install pynput  (needed on macOS)")
        raise e

    _pressed: set[str] = set()

    def _to_str(key: _pyn_kb.Key | _pyn_kb.KeyCode) -> str | None:
        if isinstance(key, _pyn_kb.KeyCode) and key.char is not None:
            return key.char.lower()
        # Named keys we actually need
        mapping = {
            _pyn_kb.Key.space: "space",
            _pyn_kb.Key.enter: "enter",
            _pyn_kb.Key.esc: "esc",
            _pyn_kb.Key.shift: "shift",
            _pyn_kb.Key.cmd: "cmd",
            _pyn_kb.Key.ctrl: "ctrl",
            _pyn_kb.Key.alt: "alt",
            _pyn_kb.Key.tab: "tab",
        }
        return mapping.get(key)

    def _on_press(key):
        k = _to_str(key)
        if k:
            _pressed.add(k)

    def _on_release(key):
        k = _to_str(key)
        if k and k in _pressed:
            _pressed.remove(k)

    _listener = _pyn_kb.Listener(on_press=_on_press, on_release=_on_release)
    _listener.daemon = True
    _listener.start()

    def is_pressed(key: str) -> bool:  # noqa: N802  (simple alias)
        return key.lower() in _pressed
else:
    # ---------------------------------------------------------------------
    # Windows / Linux path – prefer the very low‑latency `keyboard` library
    # (requires sudo on some Linux distros). Fall back gracefully to pynput.
    # ---------------------------------------------------------------------
    try:
        import keyboard  # type: ignore  # noqa: WPS433

        def is_pressed(key: str) -> bool:  # noqa: N802
            return keyboard.is_pressed(key)

    except Exception:  # pragma: no cover – any import / runtime failure
        from pynput import keyboard as _pyn_kb  # type: ignore

        _pressed: set[str] = set()

        def _to_str(key: _pyn_kb.Key | _pyn_kb.KeyCode) -> str | None:
            if isinstance(key, _pyn_kb.KeyCode) and key.char is not None:
                return key.char.lower()
            mapping = {
                _pyn_kb.Key.space: "space",
                _pyn_kb.Key.enter: "enter",
                _pyn_kb.Key.esc: "esc",
                _pyn_kb.Key.ctrl_l: "ctrl",
                _pyn_kb.Key.ctrl_r: "ctrl",
            }
            return mapping.get(key)

        def _on_press(key):
            k = _to_str(key)
            if k:
                _pressed.add(k)

        def _on_release(key):
            k = _to_str(key)
            if k and k in _pressed:
                _pressed.remove(k)

        _listener = _pyn_kb.Listener(on_press=_on_press, on_release=_on_release)
        _listener.daemon = True
        _listener.start()

        def is_pressed(key: str) -> bool:  # noqa: N802
            return key.lower() in _pressed

# ---------------------------------------------------------------------------
# Environment import
# ---------------------------------------------------------------------------
try:
    from blockswap.blockswap_env import BlockSwapEnv
except ModuleNotFoundError:
    from blockswap_env import BlockSwapEnv  # type: ignore  # local fallback

# ---------------------------------------------------------------------------
# Key → action mapping helpers (same as before)
# ---------------------------------------------------------------------------

_TRANSLATION_HOTKEYS = {
    "w": (1, +1),
    "s": (1, -1),
    "a": (0, -1),
    "d": (0, +1),
    "r": (2, +1),
    "f": (2, -1),
}
_GRIPPER_TOGGLE_KEY = "space"
_NEXT_EPISODE_KEY = "enter"
_ABORT_KEY = "esc"
_PAUSE_KEY = "p"
_STEP_MAG = 1.0  # value in [-1, 1]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _poll_keyboard() -> Tuple[np.ndarray, bool, bool, bool]:
    """Return action vector and control flags based on current key state."""
    action = np.zeros(4, dtype=np.float32)

    # Cartesian translation
    for key, (axis, sign) in _TRANSLATION_HOTKEYS.items():
        if is_pressed(key):
            action[axis] += sign * _STEP_MAG

    reset = is_pressed(_NEXT_EPISODE_KEY)
    abort = is_pressed(_ABORT_KEY)
    pause = is_pressed(_PAUSE_KEY)
    return action, reset, abort, pause


# ---------------------------------------------------------------------------
# Main data‑collection routine (unchanged apart from is_pressed backend)
# ---------------------------------------------------------------------------

def collect_demos(
    num_episodes: int,
    out_dir: Path,
    observation_mode: str = "full",
    max_steps: int | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    env = BlockSwapEnv(render_mode="human", observation_mode=observation_mode)

    grip_state = -1.0
    last_toggle_state = False
    paused = False

    for ep in range(num_episodes):
        obs, _ = env.reset()
        episode_obs: List[np.ndarray] = []
        episode_act: List[np.ndarray] = []
        episode_rew: List[float] = []
        episode_done: List[bool] = []

        t = 0
        print(
            f"\n>>> Recording episode {ep+1}/{num_episodes}. "
            "Press <enter> to end, <esc> to abort."
        )

        while True:
            if not paused:
                action_vec, reset_flag, abort_flag, pause_flag = _poll_keyboard()

                # Gripper toggle on rising edge
                toggle_pressed = is_pressed(_GRIPPER_TOGGLE_KEY)
                if toggle_pressed and not last_toggle_state:
                    grip_state *= -1
                last_toggle_state = toggle_pressed
                action_vec[3] = grip_state

                next_obs, reward, terminated, truncated, _ = env.step(action_vec)
                env.render()

                episode_obs.append(obs)
                episode_act.append(action_vec)
                episode_rew.append(float(reward))
                episode_done.append(bool(terminated or truncated))

                obs = next_obs
                t += 1

                if max_steps and t >= max_steps:
                    reset_flag = True

                if abort_flag:
                    print("Aborting … saving current episode first.")
                    _save_episode(out_dir, ep, episode_obs, episode_act, episode_rew, episode_done, observation_mode)
                    env.close()
                    sys.exit(0)
                if reset_flag or terminated or truncated:
                    print(f"Episode finished after {t} steps. Saving …")
                    _save_episode(out_dir, ep, episode_obs, episode_act, episode_rew, episode_done, observation_mode)
                    break
            else:
                _, _, abort_flag, pause_flag = _poll_keyboard()
                if pause_flag:
                    paused = not paused
                    print("Resumed." if not paused else "Paused.")
                if abort_flag:
                    print("Aborting from pause state.")
                    env.close()
                    sys.exit(0)

            # Handle pause toggle when running
            if not paused and _poll_keyboard()[3]:
                paused = True
                print("Simulation paused. Press 'p' to resume.")

            time.sleep(0.05)  # ≈20 Hz

    env.close()
    print(f"\nAll {num_episodes} demonstrations saved to {out_dir.resolve()}.")


# ---------------------------------------------------------------------------
# Helper to persist an episode
# ---------------------------------------------------------------------------

def _save_episode(
    out_dir: Path,
    idx: int,
    obs: List[np.ndarray],
    act: List[np.ndarray],
    rew: List[float],
    done: List[bool],
    obs_mode: str,
) -> None:
    obs_arr = np.asarray(obs, dtype=np.float32)
    act_arr = np.asarray(act, dtype=np.float32)
    rew_arr = np.asarray(rew, dtype=np.float32)
    done_arr = np.asarray(done, dtype=bool)

    filename = out_dir / f"demo_{idx:03d}.npz"
    np.savez_compressed(
        filename,
        observations=obs_arr,
        actions=act_arr,
        rewards=rew_arr,
        dones=done_arr,
        observation_mode=np.string_(obs_mode),
        control_frequency=np.float32(20.0),
    )
    print(f"Saved → {filename}  (T = {len(obs_arr)})")


# ---------------------------------------------------------------------------
# CLI entry‑point
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Collect BlockSwap demos via keyboard teleop.")
    parser.add_argument("--episodes", type=int, default=5, help="number of episodes to record")
    parser.add_argument("--out-dir", type=Path, default=Path("demos"), help="directory to save .npz files")
    parser.add_argument("--obs-mode", choices=["full", "partial"], default="full", help="environment observation mode")
    parser.add_argument("--max-steps", type=int, default=None, help="optional hard cap on steps per episode")
    args = parser.parse_args(argv)

    collect_demos(args.episodes, args.out_dir, args.obs_mode, args.max_steps)


if __name__ == "__main__":  # pragma: no cover
    main()
