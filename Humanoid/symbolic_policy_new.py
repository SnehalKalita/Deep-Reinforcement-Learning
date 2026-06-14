import numpy as np

def symbolic_policy(obs):
    # --- Math Operations ---
    sin = np.sin
    cos = np.cos
    exp = np.exp
    
    # --- Joint Equations ---
    a0 = obs[331]*(-0.0008924647)
    a1 = obs[193]*0.06419729 + obs[131]
    a2 = obs[285]*(-0.0013432663)
    a3 = obs[333]*(-0.0003892351)
    a4 = (obs[333]*0.0014990132 + obs[82])*(-0.20264655)
    a5 = (obs[275] - (obs[333] + obs[351]))*(-0.00022186937)
    a6 = 0.34808254 + obs[333]*(-0.00048735822)
    a7 = obs[82]*(obs[283]*(-0.0055495803) - 0.14799878)
    a8 = (obs[151] + obs[131])*0.5640424 + 0.09035276
    a9 = obs[351]*0.00040872753 + obs[200]*(-0.13595791)
    a10 = obs[60]*obs[151] + exp(obs[351]*(-0.0071339114))*0.14756848
    a11 = obs[131]*(-0.7146032)
    a12 = obs[351]*0.00020620391 - 1*(-0.10070938)
    a13 = obs[229]*0.021164792 + obs[275]*(-0.002598168)
    a14 = (obs[62] + obs[200])*(-0.1451643)
    a15 = (obs[131] - (obs[22] + obs[200]*0.31795198))*0.31978342
    a16 = (obs[285] - 68.86225)*0.0018979372
    
    action = np.array([a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16], dtype=np.float32)
    action = np.nan_to_num(action)
    action = np.clip(action, -1.0, 1.0)
    return action