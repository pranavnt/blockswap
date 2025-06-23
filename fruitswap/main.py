"""
Main entry point for the FruitSwap package.
"""

from .blockswap_env import BlockSwapEnv

def main():
    """
    Demo function showing basic usage of the BlockSwapEnv.
    """
    print("Welcome to FruitSwap!")
    print("Creating BlockSwapEnv...")

    try:
        env = BlockSwapEnv(render_mode='human')
        print(f"Environment created successfully!")
        print(f"Action space: {env.action_space}")
        print(f"Observation space: {env.observation_space}")

        # Reset environment
        obs, info = env.reset()
        if hasattr(obs, 'shape'):
            print(f"Environment reset. Observation shape: {obs.shape}")
        else:
            print(f"Environment reset. Observation type: {type(obs)}")

        # Take a few random actions
        for step in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step {step + 1}: reward={reward:.3f}, terminated={terminated}, truncated={truncated}")

            if terminated or truncated:
                break

        env.close()
        print("Demo completed successfully!")

    except Exception as e:
        print(f"Error running demo: {e}")
        print("Make sure you have mujoco and all dependencies installed.")

if __name__ == "__main__":
    main()