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

def run_comprehensive_shap_analysis():
    env_id = "Walker2d-v4" 
    env = gym.make(env_id)
    
    model_path = "sac_walker2d"
    print(f"Loading expert policy from {model_path}...")
    model = SAC.load(model_path, env=env)
    
    wrapped_actor = DeterministicActorWrapper(model)
    wrapped_actor.eval() 
    device = model.device
    wrapped_actor.to(device)

    # 1. Gather Data
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

    print("Gathering test states to explain (10 states)...")
    test_states = []
    obs, _ = env.reset()
    for _ in range(10):
        test_states.append(obs)
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        
    test_tensor = torch.tensor(np.vstack(test_states), dtype=torch.float32).to(device)

    # 2. Calculate SHAP Values (Executed once for both outputs)
    print("Calculating SHAP values (this may take a moment)...")
    explainer = shap.DeepExplainer(wrapped_actor, background_tensor)
    shap_values = explainer.shap_values(test_tensor, check_additivity=False)

    # 3. Global Feature Aggregation
    print("\n--- Part 1: Aggregating Global Feature Importance ---")
    if isinstance(shap_values, list):
        shap_tensor = np.stack(shap_values, axis=2)
    else:
        shap_tensor = shap_values

    abs_shap = np.abs(shap_tensor)
    mean_across_samples = np.mean(abs_shap, axis=0)
    global_importance = np.mean(mean_across_samples, axis=1)

    top_indices = np.argsort(global_importance)[::-1]
    top_10_features = top_indices[:10].tolist()
    
    print(f"\nMASTER FEATURE LIST (TOP 10)")
    print(f"Top Indices Array for PySR Masking:\n{top_10_features}\n")
    
    print("Top 5 Breakdown (Index -> Importance Score):")
    for i in range(5):
        idx = top_indices[i]
        score = global_importance[idx]
        print(f"  Feature {idx:3d} : {score:.5f}")

    # 4. Summary Plot Visualization
    print("\n--- Part 2: Generating Summary Plot ---")
    action_index = 0 # Adjust this to check different joints (0 to 5 for Walker2d)
    feature_names = [f"Feature_{i}" for i in range(background_tensor.shape[1])]
    
    if isinstance(shap_values, list):
        plot_values = shap_values[action_index]
    else:
        plot_values = shap_values[:, :, action_index] if shap_values.ndim == 3 else shap_values

    shap.summary_plot(
        plot_values, 
        test_tensor.cpu().numpy(), 
        feature_names=feature_names,
        show=True
    )
    
    return top_10_features

if __name__ == "__main__":
    run_comprehensive_shap_analysis()