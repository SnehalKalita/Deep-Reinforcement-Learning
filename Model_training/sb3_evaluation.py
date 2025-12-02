import numpy as np
import gymnasium as gym
import myosuite
from stable_baselines3 import PPO
from tqdm import tqdm
import skvideo.io

# 1. Configuration
# This ID is confirmed to exist now.
env_id = "myoChallengeSoccerP1-v0" 
model_name = "Soccer_policy1"  # Matches the name in your training notebook

print(f"Initializing {env_id}...")

# 2. Native Gymnasium Load
# No wrappers, no Shimmy. Just raw Gymnasium.
try:
    env = gym.make(env_id)
except gym.error.NameNotFound:
    print(f"CRITICAL ERROR: Even though verification said True, {env_id} failed to load.")
    exit()

# 3. Load the Model
print(f"Loading model: {model_name}...")
try:
    # We pass 'env' so SB3 can infer the observation/action space automatically
    model = PPO.load(model_name, env=env)
except FileNotFoundError:
    print(f"ERROR: Could not find '{model_name}.zip'. Did you move the file?")
    exit()
except Exception as e:
    print(f"Model load error: {e}")
    exit()

frames = [] 
all_rewards = [] 
num_episodes = 5

print(f"Starting evaluation over {num_episodes} episodes...")

for n_episode in tqdm(range(num_episodes)):
    # Gymnasium API: reset returns (obs, info)
    obs, info = env.reset()
    ep_rewards = [] 
    
    terminated = False
    truncated = False
    
    while not (terminated or truncated):
        # Predict action (deterministic=True is better for evaluation)
        action, _ = model.predict(obs, deterministic=True)
        
        # Gymnasium API: step returns 5 values
        obs, r, terminated, truncated, info = env.step(action)
        
        ep_rewards.append(r)

        # Render only the last episode to save memory
        if n_episode == num_episodes - 1:
            try:
                # Access the internal MuJoCo renderer directly for offscreen recording
                # This bypasses any standard gym render limitations
                img = env.unwrapped.sim.renderer.render_offscreen(
                    width=640, height=480, camera_id=1
                )
                frames.append(img)
            except AttributeError:
                # Fallback if internal structure is different
                frames.append(env.render())

    all_rewards.append(np.sum(ep_rewards))

env.close()

print(f"Average reward: {np.mean(all_rewards):.2f} over {num_episodes} episodes")

# 4. Save Video
if len(frames) > 0:
    print(f"Saving video with {len(frames)} frames...")
    try:
        skvideo.io.vwrite('myoChallengeSoccer_baseline.mp4', np.asarray(frames), 
                          inputdict={'-r': '100'}, 
                          outputdict={"-pix_fmt": "yuv420p"})
        print("Video saved successfully: myoChallengeSoccer_baseline.mp4")
    except Exception as e:
        print(f"Video save failed. Check FFMPEG install. Error: {e}")
else:
    print("No frames were captured. Check your loop logic.")