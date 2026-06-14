import os
os.environ['JULIA_PKG_SERVER'] = ""
import warnings
warnings.filterwarnings('ignore')

from pysr import PySRRegressor
import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from sklearn.linear_model import Lasso

# ==========================================
# Phase 1: Loading the Pre-Trained SAC Policy
# ==========================================
print("--- Phase 1: Loading Pre-trained SAC on Humanoid ---")
env = gym.make("Humanoid-v4")
model = SAC.load("X:\TD-MPC2\Humanoid\sac_humanoid.zip", env=env, device='cpu')
print("Model loaded successfully!")

# ==========================================
# Phase 2: Collect State-Action Data
# ==========================================
print("\n--- Phase 2: Collecting Trajectory Data ---")
obs, _ = env.reset()
observations = []
actions = []

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
# Phase 3 & 4: Loop Through All 17 Joints (Pure PySR)
# =========================================================
TOTAL_JOINTS = 17
all_equations_log = []

for target_joint in range(TOTAL_JOINTS):
    print(f"\n{'='*50}")
    print(f" DISTILLING JOINT {target_joint} / {TOTAL_JOINTS - 1} ")
    print(f"{'='*50}")

    y_target = Y_data[:, target_joint]

    # --- Feature Pruning ---
    lasso = Lasso(alpha=0.01)
    lasso.fit(X_data, y_target)
    important_features = np.where(lasso.coef_ != 0)[0]

    if len(important_features) == 0:
        important_features = np.argsort(np.abs(lasso.coef_))[-5:]
    elif len(important_features) > 5:
        important_features = important_features[np.argsort(np.abs(lasso.coef_[important_features]))[-5:]]

    print(f"Selected top {len(important_features)} observation indices: {important_features}")
    X_pruned = X_data[:, important_features]

    # --- Native PySR Distillation ---
    pysr_model = PySRRegressor(
        niterations=30,
        binary_operators=["+", "*", "-", "/"],
        unary_operators=["sin", "cos"], 
        maxsize=15,
        verbosity=0 
    )
    
    print("Searching for equation...")
    pysr_model.fit(X_pruned, y_target)
    
    # Grabing the pure string equation directly from PySR!
    equation_string = str(pysr_model.sympy())
    
    log_entry = f"Joint {target_joint}:\n  Features (x0 to x{len(important_features)-1}): {important_features}\n  Equation: {equation_string}\n"
    all_equations_log.append(log_entry)
    
    print(f"=> Joint {target_joint} distilled successfully: {equation_string}")

# =========================================================
# Phase 5: Saving all the Equations in a File
# =========================================================
print("\n--- All 17 Joints Distilled! ---")
output_file = "humanoid_equations.txt"

with open(output_file, "w") as f:
    f.write("=== HUMANOID SYMBOLIC POLICY ===\n\n")
    for log in all_equations_log:
        f.write(log + "\n")

print(f"Equations saved successfully to: {output_file}")