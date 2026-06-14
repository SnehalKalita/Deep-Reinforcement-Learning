import gymnasium as gym
from stable_baselines3 import SAC
import numpy as np

def evaluate_policy_reward(model_path, env_id="Humanoid-v4", num_episodes=10):
    env = gym.make(env_id)

    print(f"Loading model from {model_path}...")
    model = SAC.load(model_path, env=env)

    episode_rewards = []

    print("\nStarting evaluation rollouts...")
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1} completed. Total Reward: {total_reward:.2f}")

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    print(f"\n{'='*30}")
    print(f" FINAL EVALUATION RESULTS ")
    print(f"{'='*30}")
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    return mean_reward, std_reward

if __name__ == "__main__":
    model_file = "sac_humanoid.zip"
    evaluate_policy_reward(model_file)