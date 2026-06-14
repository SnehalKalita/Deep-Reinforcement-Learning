import gymnasium as gym
import numpy as np
import sympy as sp
import re

def load_symbolic_policy(filepath):
    """
    Parses the text file and compiles raw string equations into fast numpy functions.
    """
    with open(filepath, 'r') as f:
        content = f.read()

    # Spliting the file into blocks per joint
    blocks = content.split('Joint ')[1:]
    
    policy = []
    for block in blocks:

        # Extracting the array of feature indices
        feat_match = re.search(r'Features.*?\[(.*?)\]', block)

        # Handle formatting quirks like double spaces or newlines inside brackets
        indices_str = feat_match.group(1).replace('\n', ' ')
        indices = [int(idx) for idx in indices_str.split()]
        
        # Extracting the actual equation string
        eq_match = re.search(r'Equation:\s*(.*)', block)
        eq_str = eq_match.group(1).strip()
        
        # Creating SymPy symbols dynamically (x0, x1, x2, etc.) matching the number of features
        symbols = sp.symbols(f'x0:{len(indices)}')
        
        # Parsing the string into a SymPy mathematical expression
        expr = sp.sympify(eq_str)
        
        # Compiling the expression into a fast, executable numpy function
        func = sp.lambdify(symbols, expr, 'numpy')
        
        policy.append({
            'indices': indices,
            'func': func
        })
        
    return policy

def evaluate_symbolic_policy(policy, env_id="Humanoid-v4", num_episodes=10):
    env = gym.make(env_id)
    episode_rewards = []

    print("\nStarting evaluation rollouts for Symbolic Policy...")
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = np.zeros(17)
            
            # Calculating the torque for each joint independently
            for i, joint in enumerate(policy):
                # Extracting only the specific observation variables this equation cares about
                inputs = obs[joint['indices']]
                
                # Evaluating the mathematical equation
                # We use *inputs to unpack the array and pass variables as x0, x1...
                action_val = joint['func'](*inputs)
                action[i] = action_val
            
            # CRITICAL: The SAC actor network physically bounds its output between [-1, 1] using a Tanh function. 
            # Our algebraic equations have no such boundaries and can mathematically explode. 
            # We must clip the calculated forces to respect the physical limits of the humanoid's motors.
            action = np.clip(action, -1.0, 1.0)
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1} completed. Total Reward: {total_reward:.2f}")

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    print(f"\n{'='*30}")
    print(f" SYMBOLIC POLICY RESULTS ")
    print(f"{'='*30}")
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

if __name__ == "__main__":
    # Loading the symbolic equations from the text file and evaluating them in the environment
    filepath = "humanoid_equations_new.txt"
    print("Parsing equations and compiling math functions...")
    
    symbolic_policy = load_symbolic_policy(filepath)
    evaluate_symbolic_policy(symbolic_policy)