# import gymnasium as gym
# import numpy as np
# from stable_baselines3 import SAC
# import imageio  

# # 1. Load the environment with 'rgb_array' to capture pixels quietly
# print("Starting evaluation and recording video...")
# env = gym.make("Humanoid-v4", render_mode="rgb_array")

# # 2. Load the trained brain
# model = SAC.load("X:\TD-MPC2\Humanoid\sac_humanoid.zip", env=env, device="cpu") 

# obs, _ = env.reset()
# done = False

# total_error = 0
# steps = 0
# frames = [] 

# # Capture the very first starting frame
# frames.append(env.render())

# # 3. The Cyborg Evaluation Loop
# while not done:
#     # A. Ask the Neural Network what it wants to do
#     action, _ = model.predict(obs, deterministic=True)
    
#     # B. Calculate what our pure Math Equation wants to do for Joint 0
#     # Mapping the new SAC features: x2=obs[19], x3=obs[62], x4=obs[16]
#     val_x2 = obs[19]
#     val_x3 = obs[62]
#     val_x4 = obs[16]
    
#     # Complexity 15 Equation:
#     # y = ((x₂ / -1.2573) - (cos((cos(x₄) * x₃) + 0.35951) - -0.7402)) * -0.11045
#     math_activation = ((val_x2 / -1.2573) - (np.cos((np.cos(val_x4) * val_x3) + 0.35951) + 0.7402)) * -0.11045
    
#     # C. Record the difference
#     nn_activation = action[0]
#     total_error += abs(nn_activation - math_activation)
#     steps += 1
    
#     # D. THE CYBORG OVERRIDE
#     action[0] = math_activation
    
#     # E. Step the simulation
#     obs, reward, terminated, truncated, info = env.step(action)
    
#     # F. Capture the new frame and add it to our movie!
#     frames.append(env.render())
    
#     if terminated or truncated:
#         break

# env.close()

# # 4. Save the Video
# video_filename = "humanoid_eval_sac.mp4"
# print(f"\nEncoding and saving video to {video_filename}...")
# imageio.mimsave(video_filename, frames, fps=30)

# # 5. Print the final results
# print(f"\n--- Evaluation Complete ---")
# print(f"Total Steps Survived: {steps}")
# print(f"Average absolute difference between NN and Math per step: {total_error / steps:.5f}")

import gymnasium as gym
import numpy as np
import imageio  

from symbolic_policy_new import symbolic_policy

print("Starting evaluation using ONLY the Symbolic Math Policy...")
env = gym.make("Humanoid-v4", render_mode="rgb_array")

obs, _ = env.reset()
done = False

steps = 0
frames = [] 
frames.append(env.render())

while not done:
    # The Math Equation directly calculates all 17 joint actions
    action = symbolic_policy(obs)
    
    obs, reward, terminated, truncated, info = env.step(action)
    steps += 1
    
    frames.append(env.render())
    
    if terminated or truncated:
        break

env.close()

video_filename = "humanoid_pure_math_eval_new.mp4"
print(f"\nEncoding and saving video to {video_filename}...")
imageio.mimsave(video_filename, frames, fps=30)

print(f"\n--- Evaluation Complete ---")
print(f"Total Steps Survived: {steps}")