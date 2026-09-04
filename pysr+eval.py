import os
import re
import warnings
import numpy as np
import sympy as sp
import gymnasium as gym
from stable_baselines3 import SAC
from sklearn.linear_model import Lasso
from pysr import PySRRegressor

os.environ['JULIA_PKG_SERVER'] = ""
warnings.filterwarnings('ignore')

# ==========================================
# 1. Neural Policy Evaluation
# ==========================================
def evaluate_neural_policy(model_path, env_id="Walker2d-v4", num_episodes=10):
    env = gym.make(env_id)
    print(f"Loading SAC model from {model_path}...")
    model = SAC.load(model_path, env=env, device='cpu')

    episode_rewards = []
    print("\nStarting evaluation rollouts for SAC Expert...")
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
    
    print(f"\n{'='*30}\n NEURAL POLICY RESULTS \n{'='*30}")
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}\n")
    return model

# ==========================================
# 2. Physics-Aware Symbolic Distillation
# ==========================================
def distill_policy(model, env_id="Walker2d-v4", output_file="walker2d_equations.txt"):
    env = gym.make(env_id)
    print("--- Collecting Trajectory Data ---")
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

    TOTAL_JOINTS = 6  # Walker2d action space
    DYNAMIC_INDICES = list(range(8, 17)) # Observation indices for velocities/dynamics
    all_equations_log = []

    for target_joint in range(TOTAL_JOINTS):
        print(f"\n{'='*50}\n DISTILLING JOINT {target_joint} / {TOTAL_JOINTS - 1} \n{'='*50}")
        y_target = Y_data[:, target_joint]

        # Physics-Aware Feature Pruning
        lasso = Lasso(alpha=0.01)
        lasso.fit(X_data, y_target)
        
        dynamic_weights = np.abs(lasso.coef_[DYNAMIC_INDICES])
        top_dynamic = [DYNAMIC_INDICES[i] for i in np.argsort(dynamic_weights)[-2:]]
        standard_features = np.argsort(np.abs(lasso.coef_))[-3:]
        
        important_features = np.unique(np.concatenate((standard_features, top_dynamic)))
        print(f"Selected observation indices (Physics-Aware): {important_features}")
        
        X_pruned = X_data[:, important_features]

        pysr_model = PySRRegressor(
            niterations=30,
            binary_operators=["+", "*", "-", "/"],
            unary_operators=["sin", "cos"], 
            maxsize=15,
            verbosity=0 
        )
        
        print("Searching for equation...")
        pysr_model.fit(X_pruned, y_target)
        equation_string = str(pysr_model.sympy())
        
        log_entry = f"Joint {target_joint}:\n  Features (x0 to x{len(important_features)-1}): {important_features}\n  Equation: {equation_string}\n"
        all_equations_log.append(log_entry)
        print(f"=> Joint {target_joint} distilled successfully: {equation_string}")

    with open(output_file, "w") as f:
        f.write("=== WALKER2D SYMBOLIC POLICY ===\n\n")
        for log in all_equations_log:
            f.write(log + "\n")
    print(f"\nEquations saved successfully to: {output_file}")

# ==========================================
# 3. Load and Evaluate Symbolic Policy
# ==========================================
def load_symbolic_policy(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    blocks = content.split('Joint ')[1:]
    policy = []
    
    for block in blocks:
        feat_match = re.search(r'Features.*?\[(.*?)\]', block)
        indices_str = feat_match.group(1).replace('\n', ' ')
        indices = [int(idx) for idx in indices_str.split()]
        
        eq_match = re.search(r'Equation:\s*(.*)', block)
        eq_str = eq_match.group(1).strip()
        
        symbols = sp.symbols(f'x0:{len(indices)}')
        expr = sp.sympify(eq_str)
        func = sp.lambdify(symbols, expr, 'numpy')
        
        policy.append({
            'indices': indices,
            'func': func
        })
    return policy

def evaluate_symbolic_policy(policy, env_id="Walker2d-v4", num_episodes=10):
    env = gym.make(env_id)
    episode_rewards = []

    print("\nStarting evaluation rollouts for Symbolic Policy...")
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = np.zeros(6) # Updated for Walker2d action space
            
            for i, joint in enumerate(policy):
                inputs = obs[joint['indices']]
                action_val = joint['func'](*inputs)
                action[i] = action_val
            
            action = np.clip(action, -1.0, 1.0)
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1} completed. Total Reward: {total_reward:.2f}")

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    print(f"\n{'='*30}\n SYMBOLIC POLICY RESULTS \n{'='*30}")
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}\n")

# ==========================================
# Execution Pipeline
# ==========================================
if __name__ == "__main__":
    # Updated to point to your saved SAC model
    model_file = "sac_walker2d"
    equation_file = "walker2d_equations.txt"
    
    # 1. Evaluate the trained SAC Expert
    expert_model = evaluate_neural_policy(model_file)
    
    # 2. Distill into equations with physics-aware constraint
    distill_policy(expert_model, output_file=equation_file)
    
    # 3. Parse equations and evaluate the Symbolic Policy
    print("Parsing equations and compiling math functions...")
    symbolic_policy = load_symbolic_policy(equation_file)
    evaluate_symbolic_policy(symbolic_policy)