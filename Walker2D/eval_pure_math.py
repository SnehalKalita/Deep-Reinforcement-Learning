import re

print("Reading walker2d_equations.txt...")

try:
    with open("walker2d_equations.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
except FileNotFoundError:
    print("Error: walker2d_equations.txt not found. Did the distillation script finish?")
    exit()

# Setup the new Python file structure
output_code = [
    "import numpy as np",
    "",
    "def symbolic_policy(obs):",
    "    # --- Math Operations ---",
    "    sin = np.sin",
    "    cos = np.cos",
    "    exp = np.exp", # CRITICAL FIX: Added exponential mapping
    "    ",
    "    # --- Joint Equations ---"
]

current_features = []
joint_count = 0

for line in lines:
    line = line.strip()
    
    # 1. Grab the feature array for the current joint
    if "Features" in line:
        bracket_content = line[line.find("[")+1 : line.find("]")]
        current_features = [int(x) for x in re.findall(r'\d+', bracket_content)]
        
    # 2. Grab the equation and translate it
    elif "Equation:" in line:
        raw_equation = line.split("Equation:")[1].strip()
        
        # Normalize Julia subscripts (₀, ₁) to standard numbers (0, 1)
        subscripts = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
        raw_equation = raw_equation.translate(subscripts)
        raw_equation = raw_equation.replace("x_", "x")
        
        # CRITICAL FIX: Regex word boundaries guarantee x1 doesn't match inside x10
        for i in range(len(current_features)):
            global_index = current_features[i]
            # \b signifies a word boundary. It strictly matches 'x1' but ignores 'x10'
            raw_equation = re.sub(rf'\bx{i}\b', f"obs[{global_index}]", raw_equation)
            
        output_code.append(f"    a{joint_count} = {raw_equation}")
        joint_count += 1

# 3. Assemble the final action array
output_code.append("    ")
action_array_str = ", ".join([f"a{i}" for i in range(joint_count)])
output_code.append(f"    action = np.array([{action_array_str}], dtype=np.float32)")
output_code.append("    action = np.nan_to_num(action)")
output_code.append("    action = np.clip(action, -1.0, 1.0)")
output_code.append("    return action")

# Write to the new file
with open("symbolic_policy_walker2d.py", "w", encoding="utf-8") as f:
    f.write("\n".join(output_code))

print(f"Success! Translated {joint_count} equations into symbolic_policy_walker2d.py")


import gymnasium as gym
import numpy as np
import imageio  

from symbolic_policy_walker2d import symbolic_policy

print("Starting evaluation using ONLY the Symbolic Math Policy...")
env = gym.make("Walker2d-v4", render_mode="rgb_array")

obs, _ = env.reset()
done = False

steps = 0
frames = [] 
frames.append(env.render())

while not done:
    # The Math Equation directly calculates all 6 joint actions for Walker2d
    action = symbolic_policy(obs)
    
    obs, reward, terminated, truncated, info = env.step(action)
    steps += 1
    
    frames.append(env.render())
    
    if terminated or truncated:
        break

env.close()

video_filename = "walker2d_pure_math_eval.mp4"
print(f"\nEncoding and saving video to {video_filename}...")
imageio.mimsave(video_filename, frames, fps=30)

print(f"\n--- Evaluation Complete ---")
print(f"Total Steps Survived: {steps}")