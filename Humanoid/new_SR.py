import os
os.environ['JULIA_PKG_SERVER'] = ""
import warnings
warnings.filterwarnings('ignore')

from pysr import PySRRegressor
import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC

# ==========================================
# Phase 1: Loading the Pre-Trained SAC Policy
# ==========================================
print("--- Phase 1: Loading Pre-trained SAC on Humanoid ---")
env = gym.make("Humanoid-v4")
model = SAC.load("sac_humanoid.zip", env=env, device='cpu')
print("Model loaded successfully!")

# ==========================================
# Phase 2: Collect State-Action Data
# ==========================================
print("\n--- Phase 2: Collecting Trajectory Data ---")
obs, _ = env.reset()
observations = []
actions = []

# Gathering 2000 steps of expert data
for _ in range(2000):
    action, _ = model.predict(obs, deterministic=True)
    observations.append(obs)
    actions.append(action)
    
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()

X_data = np.array(observations)
Y_data = np.array(actions)

# =========================================================
# Phase 3: The DeepSHAP Global Feature Mask
# =========================================================
# This replaces the Lasso algorithm entirely. We force PySR to use 
# only the biomechanically relevant features proven by the SAC expert.
top_30_features = [
    29, 333, 198, 62, 206, 81, 22, 200, 37, 82, 
    194, 284, 285, 351, 60, 328, 280, 7, 283, 229, 
    275, 151, 193, 260, 33, 61, 131, 9, 258, 331
]

print(f"\n--- Phase 3: Applying SHAP Mask ---")
print(f"Original observation space: {X_data.shape}")

# Physically blindfolding PySR to the 346 irrelevant features
X_pruned = X_data[:, top_30_features]

print(f"Masked observation space: {X_pruned.shape}")

# =========================================================
# Phase 4: Loop Through All 17 Joints (Hard-Masked PySR)
# =========================================================
TOTAL_JOINTS = 17
all_equations_log = []

for target_joint in range(TOTAL_JOINTS):
    print(f"\n{'='*50}")
    print(f" DISTILLING JOINT {target_joint} / {TOTAL_JOINTS - 1} ")
    print(f"{'='*50}")

    y_target = Y_data[:, target_joint]

    # Initialize PySR with the patched hyperparameters
    pysr_model = PySRRegressor(
        niterations=60, 
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos", "exp"],
        # extra_sympy_mappings={"exp": lambda x: np.exp(x)},
        loss="loss(prediction, target) = (prediction - target)^2",
        maxsize=15,
        verbosity=0 
    )
    
    print("Searching for equation...")
    pysr_model.fit(X_pruned, y_target)
    
    # Grab the pure string equation directly from PySR
    equation_string = str(pysr_model.sympy())
    
    # CRITICAL: We pass the top_30_features list into the log so your 
    # regex parser knows exactly which variables x0 through x29 map to.
    log_entry = f"Joint {target_joint}:\n  Features (x0 to x{len(top_30_features)-1}): {top_30_features}\n  Equation: {equation_string}\n"
    all_equations_log.append(log_entry)
    
    print(f"=> Joint {target_joint} distilled successfully: {equation_string}")

# =========================================================
# Phase 5: Saving all the Equations in a File
# =========================================================
print("\n--- All 17 Joints Distilled! ---")
output_file = "humanoid_equations_new.txt"

with open(output_file, "w") as f:
    f.write("=== HUMANOID SYMBOLIC POLICY ===\n\n")
    for log in all_equations_log:
        f.write(log + "\n")

print(f"Equations saved successfully to: {output_file}")