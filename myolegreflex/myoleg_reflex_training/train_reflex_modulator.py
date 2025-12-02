# train_reflex_modulator.py
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from myoleg_reflex_env import MyoLegReflexKickEnv

def make_env():
    return MyoLegReflexKickEnv(
        baseline_params_path="baseline_params.txt",
        control_dt=0.02,
        sim_dt=0.001,
        delta_scale=0.12,   # allow ±12% param mod
        include_prev_action=True,
        max_episode_seconds=8.0
    )

if __name__ == "__main__":
    n_envs = 4
    env = DummyVecEnv([make_env for _ in range(n_envs)])
    # Optional: normalize observations and rewards
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    logdir = "./logs_reflex"
    os.makedirs(logdir, exist_ok=True)

    # callbacks
    checkpoint_cb = CheckpointCallback(save_freq=5000, save_path=logdir, name_prefix='rl_model')
    eval_env = make_env()
    eval_cb = EvalCallback(eval_env, best_model_save_path=logdir+"/best",
                           log_path=logdir+"/eval", eval_freq=5000, deterministic=True, render=False)

    model = PPO("MlpPolicy",
                env,
                verbose=1,
                learning_rate=3e-4,
                n_steps=2048//n_envs,
                batch_size=256,
                n_epochs=10,
                gamma=0.995,
                gae_lambda=0.95,
                ent_coef=1e-4,
                tensorboard_log=logdir + "/tb",
                )

    total_timesteps = 2_000_000  # adjust
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_cb, eval_cb])
    model.save(logdir + "/final_model")
    print("Training finished. Model saved to", logdir + "/final_model")
