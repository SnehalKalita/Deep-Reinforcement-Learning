import gymnasium as gym
from stable_baselines3 import SAC
import torch
import torch.nn as nn
import shap
import numpy as np

class DeterministicActorWrapper(nn.Module):
    def __init__(self, sac_model):
        super().__init__()
        self.actor = sac_model.policy.actor

    def forward(self, obs):
        features = self.actor.features_extractor(obs)
        latent_pi = self.actor.latent_pi(features)
        mean_actions = self.actor.mu(latent_pi)
        deterministic_actions = torch.tanh(mean_actions)
        return deterministic_actions

def generate_master_feature_list():
    env_id = "Humanoid-v4" 
    env = gym.make(env_id)
    
    # Load your trained model
    model_path = "sac_humanoid.zip"
    print(f"Loading expert policy from {model_path}...")
    model = SAC.load(model_path, env=env)
    
    wrapped_actor = DeterministicActorWrapper(model)
    wrapped_actor.eval() 
    device = model.device
    wrapped_actor.to(device)

    # 1. Gather Background Data
    print("Gathering background data (300 states)...")
    background_states = []
    obs, _ = env.reset()
    for _ in range(300):
        background_states.append(obs)
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
             
    background_tensor = torch.tensor(np.vstack(background_states), dtype=torch.float32).to(device)

    # 2. Gather Test Data
    print("Gathering test states to explain (10 states)...")
    test_states = []
    obs, _ = env.reset()
    for _ in range(10):
        test_states.append(obs)
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        
    test_tensor = torch.tensor(np.vstack(test_states), dtype=torch.float32).to(device)

    # 3. Calculate SHAP Values
    print("Calculating SHAP values...")
    explainer = shap.DeepExplainer(wrapped_actor, background_tensor)
    shap_values = explainer.shap_values(test_tensor, check_additivity=False)

    # 4. Aggregation Pipeline
    print("\nAggregating global feature importance...")
    
    # DeepSHAP returns a list of arrays (one for each of the 17 joints)
    # We stack them into a single 3D array: [samples, features, joints]
    if isinstance(shap_values, list):
        shap_tensor = np.stack(shap_values, axis=2)
    else:
        shap_tensor = shap_values

    # Step A: Convert everything to absolute magnitudes
    abs_shap = np.abs(shap_tensor)

    # Step B: Average across the 10 test samples (collapsing axis 0)
    # Resulting shape: [features, joints]
    mean_across_samples = np.mean(abs_shap, axis=0)

    # Step C: Average across the 17 joints (collapsing axis 1)
    # Resulting shape: [features] - This is the global importance score!
    global_importance = np.mean(mean_across_samples, axis=1)

    # 5. Extract the Top Features
    # Sort indices by importance (descending order)
    top_indices = np.argsort(global_importance)[::-1]
    
    # Let's grab the top 30 most critical features for PySR masking
    top_30_features = top_indices[:30].tolist()
    
    print(f"\n{'='*50}")
    print(" MASTER FEATURE LIST (TOP 30) ")
    print(f"{'='*50}")
    print(f"Top Indices Array for PySR Masking:\n{top_30_features}\n")
    
    # Print a quick breakdown of the top 5 just so you can see the scores
    print("Top 5 Breakdown (Index -> Importance Score):")
    for i in range(5):
        idx = top_indices[i]
        score = global_importance[idx]
        print(f"  Feature {idx:3d} : {score:.5f}")
        
    return top_30_features

if __name__ == "__main__":
    generate_master_feature_list()