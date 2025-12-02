import gymnasium as gym
import myosuite
from stable_baselines3 import PPO

# 1. Create the Environment
env_id = "myoChallengeSoccerP1-v0"
print(f"Initializing {env_id} for training...")
env = gym.make(env_id)

# 2. Train with Progress Bar
# 'progress_bar=True' enables the tqdm bar with ETA
print("Training agent with ETA...")
model = PPO("MlpPolicy", env, verbose=1)

# INCREASED steps to 10,000 so you actually have time to see the bar move
model.learn(total_timesteps=5000000, progress_bar=True) 

# 3. Save
model_name = "Soccer_policy1"
model.save(model_name)
print(f"SUCCESS: Saved compatible '{model_name}.zip'.")

env.close()