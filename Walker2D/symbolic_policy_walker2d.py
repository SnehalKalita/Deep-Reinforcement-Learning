import numpy as np

def symbolic_policy(obs):
    # --- Math Operations ---
    sin = np.sin
    cos = np.cos
    exp = np.exp
    
    # --- Joint Equations ---
    a0 = cos(sin(sin(obs[5] - 0.17322578*obs[9])) - 0.77178293)
    a1 = obs[4]*obs[4]
    a2 = cos((obs[7] - 9.677887)/obs[0])
    a3 = sin(obs[4] + 2.2441854)
    a4 = sin(obs[6]) - 1*(-0.5089335)
    a5 = sin(obs[7] + cos(9.573923/obs[0]))
    
    action = np.array([a0, a1, a2, a3, a4, a5], dtype=np.float32)
    action = np.nan_to_num(action)
    action = np.clip(action, -1.0, 1.0)
    return action