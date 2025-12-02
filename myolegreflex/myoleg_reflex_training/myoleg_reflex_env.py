# myoleg_reflex_env.py
import os
import gymnasium as gym
import numpy as np
from gymnasium import spaces
# IMPORTANT: ReflexCtrInterface.py must be in PYTHONPATH or same folder.
import ReflexCtrInterface

class MyoLegReflexKickEnv(gym.Env):
    """
    High-level Gym env that uses the Reflex controller as low-level.
    Action: additive modulation to baseline reflex params (shape = N_params), values in [-1,1].
    Observation: pelvis position, pelvis velocity, ball pos/vel, foot contacts, optionally prev_action.
    Reward: forward (x) displacement of the ball per env.step (encourages shooting).
    """

    def __init__(self,
                 baseline_params_path="baseline_params.txt",
                 control_dt=0.02,         # seconds per env step
                 sim_dt=0.001,            # must match your MuJoCo option timestep
                 delta_scale=0.1,        # max relative change to baseline params (10%)
                 include_prev_action=True,
                 max_episode_seconds=8.0):
        super().__init__()

        # load reflex interface
        self.reflex = ReflexCtrInterface.MyoLegReflex()
        self.env = self.reflex.env  # underlying mujoco env

        # load baseline params
        if not os.path.exists(baseline_params_path):
            raise FileNotFoundError(f"baseline params not found: {baseline_params_path}")
        self.baseline = np.loadtxt(baseline_params_path).astype(np.float32)
        self.N_params = self.baseline.size

        # action: per-parameter additive multiplier in [-1,1]; final param = baseline*(1 + action*delta_scale)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.N_params,), dtype=np.float32)

        # observation: compose a compact vector
        # We'll expose pelvis pos (3), pelvis linear vel (3), ball pos (3), ball vel (3), foot touch sensors (4)
        # plus prev_action if requested.
        obs_len = 3 + 3 + 3 + 3 + 4
        if include_prev_action:
            obs_len += self.N_params
        self.include_prev_action = include_prev_action

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_len,), dtype=np.float32)

        # control timing
        self.sim_dt = sim_dt
        self.control_dt = control_dt
        self.control_steps = int(round(control_dt / sim_dt))
        if self.control_steps < 1:
            self.control_steps = 1

        self.delta_scale = float(delta_scale)
        self.max_steps = int(max_episode_seconds / control_dt)
        self._step_count = 0

        # For bookkeeping
        self.prev_action = np.zeros(self.N_params, dtype=np.float32)

        # done threshold for "goal" (ball x position beyond this)
        self.goal_x = 3.0  # adjust to your soccer pitch coordinates

    def reset(self):
        # reset reflex env and reflex controller
        self.reflex.reset()
        # set baseline params initially
        self.reflex.set_control_params(self.baseline.copy())
        self.prev_action[:] = 0.0
        self._step_count = 0

        return self._get_obs()

    def step(self, action):
        # clip action
        action = np.array(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        # compute modulated params
        params = self.baseline * (1.0 + (action * self.delta_scale))
        self.reflex.set_control_params(params)

        # run the reflex for control_steps (each run_reflex_step runs one sim step and computes reflex)
        for _ in range(self.control_steps):
            # note: run_reflex_step performs one simulation step and renders if called; it uses internal env
            self.reflex.run_reflex_step()

        # compute reward: encourage forward (x) displacement of the ball
        ball_pos = self._get_ball_pos()
        ball_vel = self._get_ball_vel()

        # reward signal: forward velocity of ball (x component) times control_dt
        reward = float(ball_vel[0]) * self.control_dt

        # small penalty on magnitude of action (to regularize)
        reward -= 0.001 * float(np.sum(np.square(action)))

        obs = self._get_obs()
        self.prev_action = action.copy()

        self._step_count += 1

        # termination conditions: goal reached or falling or max steps
        done = False
        info = {}

        if ball_pos[0] > self.goal_x:
            done = True
            info['success'] = True
            reward += 10.0   # bonus for scoring / shooting far
        elif self._step_count >= self.max_steps:
            done = True
            info['success'] = False

        # you can add other failure conditions (e.g., pelvis below threshold)
        pelvis_z = self._get_pelvis_pos()[2]
        if pelvis_z < 0.4:
            # fell
            done = True
            info['fell'] = True
            reward -= 2.0

        return obs, reward, done, info

    def _get_pelvis_pos(self):
        # rely on the underlying env to return body pos: typical myosuite env has model and data
        try:
            # many myosuite envs have self.env.sim (mujoco) with model/body id
            # find pelvis body
            # robust: query by site or body names expected by MyoLegReflex
            b_id = self.env.sim.model.body_name2id('pelvis')
            pos = self.env.sim.data.body_xpos[b_id].copy()
            return pos
        except Exception:
            # fallback: zeros
            return np.zeros(3, dtype=np.float32)

    def _get_pelvis_vel(self):
        try:
            b_id = self.env.sim.model.body_name2id('pelvis')
            vel = self.env.sim.data.body_xvelp[b_id].copy()
            return vel
        except Exception:
            return np.zeros(3, dtype=np.float32)

    def _get_ball_pos(self):
        # many MyoChallenge soccer envs expose a "ball" body or geom named 'ball'. Try a few names.
        try:
            mid = self.env.sim.model.body_name2id('ball')
            return self.env.sim.data.body_xpos[mid].copy()
        except Exception:
            # try geom
            try:
                gid = self.env.sim.model.geom_name2id('ball')
                return self.env.sim.data.geom_xpos[gid].copy()
            except Exception:
                return np.zeros(3, dtype=np.float32)

    def _get_ball_vel(self):
        try:
            mid = self.env.sim.model.body_name2id('ball')
            return self.env.sim.data.body_xvelp[mid].copy()
        except Exception:
            try:
                gid = self.env.sim.model.geom_name2id('ball')
                # approximate by zero if not available
                return np.zeros(3, dtype=np.float32)
            except Exception:
                return np.zeros(3, dtype=np.float32)

    def _get_foot_contacts(self):
        # the reflex model has touch sensors named r_foot, l_foot, r_toes, l_toes
        vals = []
        for sname in ['r_foot', 'r_toes', 'l_foot', 'l_toes']:
            try:
                idx = self.env.sim.model.sensor_name2id(sname)
                vals.append(float(self.env.sim.data.sensordata[idx]))
            except Exception:
                vals.append(0.0)
        return np.array(vals, dtype=np.float32)

    def _get_obs(self):
        pelvis_pos = self._get_pelvis_pos()
        pelvis_vel = self._get_pelvis_vel()
        ball_pos = self._get_ball_pos()
        ball_vel = self._get_ball_vel()
        touches = self._get_foot_contacts()  # 4 floats

        parts = [pelvis_pos, pelvis_vel, ball_pos, ball_vel, touches]
        if self.include_prev_action:
            parts.append(self.prev_action)
        obs = np.concatenate([p.reshape(-1) for p in parts]).astype(np.float32)
        return obs

    def render(self, mode='human'):
        # delegate to underlying env render
        return self.env.mj_render()

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass
