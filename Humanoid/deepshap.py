import gymnasium as gym
from stable_baselines3 import SAC
import torch
import torch.nn as nn
import shap
import numpy as np

class DeterministicActorWrapper(nn.Module):
    """
    Wraps the SB3 SAC actor to return only the deterministic mean action.
    DeepSHAP requires a single tensor output, not a probability distribution.
    """
    def __init__(self, sac_model):
        super().__init__()
        self.actor = sac_model.policy.actor

    def forward(self, obs):
        # Passing observation directly through the feature extractor module
        features = self.actor.features_extractor(obs)
        
        # Getting the latent representation
        latent_pi = self.actor.latent_pi(features)
        
        # Extracting the mean action
        mean_actions = self.actor.mu(latent_pi)
        
        # SAC squashes the output with a Tanh to bound actions between [-1, 1]
        deterministic_actions = torch.tanh(mean_actions)
        return deterministic_actions

def main():
    env_id = "Humanoid-v4" 
    env = gym.make(env_id)
    
    # Loading trained SAC expert policy
    model_path = "sac_humanoid.zip"
    print(f"Loading model from {model_path}...")
    model = SAC.load(model_path, env=env)
    
    # Initializing the wrapper
    wrapped_actor = DeterministicActorWrapper(model)
    wrapped_actor.eval() 

    device = model.device
    wrapped_actor.to(device)

    print("Gathering background data for DeepSHAP...")
    background_states = []
    obs, _ = env.reset()
    
    for _ in range(300):
        background_states.append(obs)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            obs, _ = env.reset()
             
    # Using vstack to handle DummyVecEnv batch dimensions.
    background_tensor = torch.tensor(np.vstack(background_states), dtype=torch.float32).to(device)

    # Gathering Test Data
    print("Gathering test states to explain...")
    test_states = []
    obs, _ = env.reset()
    
    for _ in range(10):
        test_states.append(obs)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
    test_tensor = torch.tensor(np.vstack(test_states), dtype=torch.float32).to(device)

    # Initializing DeepSHAP and Calculating Values
    print("Calculating SHAP values (this may take a moment)...")
    explainer = shap.DeepExplainer(wrapped_actor, background_tensor)
    shap_values = explainer.shap_values(test_tensor, check_additivity=False)

    # Visualizing the Results
    print("Generating summary plot...")
    action_index = 0 # Adjust this to check different joints (e.g., knee, ankle)
    
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

if __name__ == "__main__":
    main()