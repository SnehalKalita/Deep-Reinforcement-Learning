# kick_demo_fixed.py
# Run with: python kick_demo_fixed.py
import time
import numpy as np
import gymnasium as gym
import ReflexCtrInterface
import mujoco

def safe_reset(env):
    res = env.reset()
    if isinstance(res, tuple) and len(res) in (2,3):
        return res[0], (res[1] if len(res) > 1 else {})
    return res, {}

def safe_step(env, action):
    res = env.step(action)
    if len(res) == 5:
        obs, rew, term, trunc, info = res
        done = term or trunc
        return obs, rew, done, info
    if len(res) == 4:
        obs, rew, done, info = res
        return obs, rew, done, info
    raise RuntimeError("Unknown env.step() signature")

def print_model_names(sim):
    """
    Robustly print body/actuator/site/geom names for different MuJoCo wrappers:
    - prefer sim.model.body_names / sim.model.actuator_names if present
    - else try sim.model._model with mujoco.mj_id2name
    - else print model counts + dir(sim.model) to help debug
    """
    print("\n--- Model info (robust) ---")
    model = getattr(sim, "model", None)
    if model is None:
        print("No sim.model available.")
        return

    # 1) direct name arrays (common in some wrappers)
    printed = False
    try:
        body_names = getattr(model, "body_names", None)
        if body_names:
            print("Bodies (index : name) from model.body_names:")
            for i, nm in enumerate(body_names):
                if isinstance(nm, bytes): nm = nm.decode()
                print(i, nm)
            printed = True
    except Exception:
        pass

    try:
        act_names = getattr(model, "actuator_names", None)
        if act_names:
            print("\nActuators (index : name) from model.actuator_names:")
            for i, nm in enumerate(act_names):
                if isinstance(nm, bytes): nm = nm.decode()
                print(i, nm)
            printed = True
    except Exception:
        pass

    try:
        site_names = getattr(model, "site_names", None)
        if site_names:
            print("\nSites (index : name) from model.site_names:")
            for i, nm in enumerate(site_names):
                if isinstance(nm, bytes): nm = nm.decode()
                print(i, nm)
            printed = True
    except Exception:
        pass

    # 2) try low-level ._model usable with mujoco.mj_id2name
    if not printed:
        low = getattr(model, "_model", None) or getattr(model, "ptr", None)
        if low is not None:
            try:
                nbody = getattr(low, "nbody", getattr(model, "nbody", None))
                print("\nBodies (index : name) via low-level model/mujoco.mj_id2name:")
                for i in range(int(nbody)):
                    try:
                        nm = mujoco.mj_id2name(low, int(mujoco.mjtObj.mjOBJ_BODY), i)
                    except Exception:
                        # some mujoco versions expect integer 'type'
                        nm = mujoco.mj_id2name(low, int(1), i)
                    print(i, nm)
                na = getattr(low, "na", getattr(model, "na", None))
                print("\nActuators (index : name):")
                for i in range(int(na)):
                    try:
                        nm = mujoco.mj_id2name(low, int(mujoco.mjtObj.mjOBJ_ACTUATOR), i)
                    except Exception:
                        nm = mujoco.mj_id2name(low, int(5), i)  # 5 may be ACTUATOR id in some builds
                    print(i, nm)
                ns = getattr(low, "nsite", getattr(model, "nsite", 0))
                print("\nSites (index : name):")
                for i in range(int(ns)):
                    try:
                        nm = mujoco.mj_id2name(low, int(mujoco.mjtObj.mjOBJ_SITE), i)
                    except Exception:
                        nm = mujoco.mj_id2name(low, int(3), i)
                    print(i, nm)
                printed = True
            except Exception as e:
                print("Low-level mj_id2name attempt failed:", e)

    # 3) final fallback: print counts and dir to help you debug manually
    if not printed:
        print("Couldn't list names via the above methods. Showing available attributes and counts:")
        for attr in ("nbody", "nq", "nv", "na", "ngeom", "nsite"):
            if hasattr(model, attr):
                print(attr, "=", getattr(model, attr))
        print("\nSome attributes on sim.model (dir() partial):")
        attrs = [a for a in dir(model) if not a.startswith("_")]
        print(attrs[:80])
    print("--- end model info ---\n")

def get_ball_pos_from_sim(sim):
    # try common low-level name lookups for 'ball' - robust heuristics
    for name in ('ball', 'Ball', 'ball_site', 'ball_body'):
        try:
            idx = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, name)
        except Exception:
            try:
                idx = mujoco.mj_name2id(getattr(sim.model, "_model", sim.model), int(mujoco.mjtObj.mjOBJ_BODY), name)
            except Exception:
                idx = -1
        if idx >= 0:
            try:
                return sim.data.body_xpos[idx].copy()
            except Exception:
                pass
    # fallback: try to find any geom name containing 'ball'
    try:
        for i in range(getattr(sim.model, "ngeom", 0)):
            try:
                nm = mujoco.mj_id2name(getattr(sim.model, "_model", sim.model), int(mujoco.mjtObj.mjOBJ_GEOM), i)
            except Exception:
                nm = None
            if nm and 'ball' in nm.lower():
                return sim.data.geom_xpos[i].copy()
    except Exception:
        pass
    return None

def choose_kick_actuators(sim, substrings=('hip','knee','ankle','leg','thigh','foot')):
    picks = []
    # try model.actuator_names first
    names = []
    act_names = getattr(sim.model, "actuator_names", None) or getattr(sim.model, "actuator_name", None)
    if act_names:
        names = [n.decode() if isinstance(n, bytes) else n for n in act_names]
    else:
        # fallback: attempt to read via low-level counts and mj_id2name if possible
        try:
            low = getattr(sim.model, "_model", sim.model)
            na = int(getattr(low, "na", getattr(sim.model, "na", 0)))
            for i in range(na):
                try:
                    nm = mujoco.mj_id2name(low, int(mujoco.mjtObj.mjOBJ_ACTUATOR), i)
                except Exception:
                    nm = f"act_{i}"
                names.append(nm)
        except Exception:
            names = [f"act_{i}" for i in range(int(getattr(sim.model, "na", 0)) or 0)]

    for i, nm in enumerate(names):
        lname = (nm or "").lower()
        for sub in substrings:
            if sub in lname:
                picks.append(i)
                break

    print("Found actuator candidates for kick (index:name):")
    for i, nm in enumerate(names):
        mark = '*' if i in picks else ' '
        print(f"{mark} {i:3d} : {nm}")
    return picks

def main():
    try:
        env = ReflexCtrInterface.MyoLegReflex().env
    except Exception as e:
        print("Wrapper failed; trying gym.make fallback. Error:", e)
        candidates = ['myoSoccer-v0', 'myoSoccerFixed-v0', 'myoSoccer-v1']
        env = None
        for cand in candidates:
            try:
                env = gym.make(cand)
                print("Made env:", cand)
                break
            except Exception:
                pass
        if env is None:
            raise RuntimeError("Cannot find soccer env automatically. Please edit the script with correct env id or show me ReflexCtrInterface header.")

    obs, info = safe_reset(env)
    sim = env.unwrapped.sim
    print("Environment reset OK. Simulation dt (approx):", getattr(env.unwrapped, 'dt', None))

    print_model_names(sim)

    kick_act_idxs = choose_kick_actuators(sim)
    if not kick_act_idxs:
        print("No actuator matched; will fallback to first two actuators (if present).")
        kick_act_idxs = [29, 37, 38, 39, 30] if getattr(sim.model, "na", 0) >= 2 else list(range(getattr(sim.model, "na", 0)))

    action_dim = env.action_space.shape[0]
    baseline_action = np.zeros(action_dim, dtype=np.float32)

    kick_strength = 3.0
    kick_duration = 6   
    kick_time = 40
    ball_trigger_dist = 0.30

    max_steps = 1000
    kicked = False

    # try to find foot site/body
    foot_pos = None
    for name in ('foot','r_foot','l_foot','right_foot','left_foot','foot_site'):
        try:
            idx = mujoco.mj_name2id(getattr(sim.model, "_model", sim.model), int(mujoco.mjtObj.mjOBJ_BODY), name)
            if idx >= 0:
                foot_pos = ('body', idx); break
        except Exception:
            try:
                idx = mujoco.mj_name2id(getattr(sim.model, "_model", sim.model), int(mujoco.mjtObj.mjOBJ_SITE), name)
                if idx >= 0:
                    foot_pos = ('site', idx); break
            except Exception:
                pass
    if foot_pos:
        print("Found foot:", foot_pos)
    else:
        print("Foot not found automatically. Ball-distance trigger will use ball position alone.")

    for t in range(max_steps):
        try:
            env.render()
        except Exception:
            try:
                env.mj_render()
            except Exception:
                pass

        ball_pos = None
        # try obs first
        if isinstance(obs, dict):
            for key in ('ball_pos','ball_position','ball'):
                if key in obs:
                    ball_pos = np.array(obs[key]); break
        if ball_pos is None:
            ball_pos = get_ball_pos_from_sim(sim)

        fpos = None
        if foot_pos and ball_pos is not None:
            typ, idx = foot_pos
            try:
                fpos = sim.data.body_xpos[idx].copy() if typ == 'body' else sim.data.site_xpos[idx].copy()
            except Exception:
                fpos = None

        do_kick = False
        if not kicked:
            if ball_pos is not None and fpos is not None:
                dist = np.linalg.norm(ball_pos - fpos)
                print(f"t={t} ball_pos={np.round(ball_pos,3)} foot_pos={np.round(fpos,3)} dist={dist:.3f}")
                if dist < ball_trigger_dist:
                    do_kick = True
                    print("Triggering kick by proximity (dist)", dist)
            elif t == kick_time:
                do_kick = True
                print("Triggering kick by fixed time", t)

        if do_kick and not kicked:
            impulse = baseline_action.copy()
            for idx in kick_act_idxs:
                if idx < action_dim:
                    impulse[idx] += kick_strength
            # clip to action_space if bounded
            try:
                impulse = np.minimum(np.maximum(impulse, env.action_space.low), env.action_space.high)
            except Exception:
                pass
            print("Applying kick impulse to actuators:", kick_act_idxs, "strength:", kick_strength)
            for k in range(kick_duration):
                obs, rew, done, info = safe_step(env, impulse)
            kicked = True
            continue
        else:
            obs, rew, done, info = safe_step(env, baseline_action)

        if done:
            print("Episode ended at step", t)
            break

    print("Done. Close env.")
    env.close()

if __name__ == "__main__":
    main()
