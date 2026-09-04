import gymnasium as gym
import imageio
from stable_baselines3 import SAC

print("--- Phase 1: Training SAC on Walker2d ---")
env = gym.make("Walker2d-v4")

model = SAC("MlpPolicy", env, verbose=1, device='cuda') 

model.learn(total_timesteps=2_00_000) 

model.save("sac_walker2d")
print("Training complete and model saved!")

print("Loading environment and model...")

env = gym.make("Walker2d-v4", render_mode="rgb_array")

model = SAC.load("sac_walker2d", env=env, device="cpu")

obs, info = env.reset()
frames = []

frames.append(env.render())

print("Recording video frames...")

# (We set a max of 1000 steps so the video doesn't get massively huge)
for step in range(1000):

    action, _states = model.predict(obs, deterministic=True)
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    frames.append(env.render())
    
    if terminated or truncated:
        print(f"Episode finished after {step} steps.")
        break

env.close()

video_filename = "sac_walker2d.mp4"

print(f"Encoding and saving video to {video_filename}...")

imageio.mimsave(video_filename, frames, fps=30)

print("Video saved successfully!")