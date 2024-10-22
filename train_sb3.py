import json
import os
from argparse import ArgumentParser

from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)
from utils.env_containers import EnvContainer
from utils.ppo import MyPPO
from utils.vecenv import MyVecEnv

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def main(args, remaining_args):
    env_name = "halfcheetah"
    backend = "mjx"
    batch_size = 1024
    episode_length = 256
    train_env_container = EnvContainer(env_name, backend, batch_size, True, episode_length)
    train_vecenv = MyVecEnv(train_env_container, seed=0)
    eval_env_container = EnvContainer(env_name, backend, 16, False, episode_length)
    eval_vecenv = MyVecEnv(eval_env_container, seed=0)

    n_iters = args.n_iters
    n_saves = 10
    save_and_eval_every = (n_iters * episode_length) // n_saves
    total_timesteps = batch_size * episode_length * n_iters

    outdir = os.path.join("out", args.run_name, f"{args.seed:02d}")
    checkpoint_callback = CheckpointCallback(
        save_freq=save_and_eval_every,
        save_path=outdir,
        name_prefix="rl_model",
    )
    # Evaluation callback to monitor progress; for analysis, it's better to use the evaluation script on snapshots saved
    eval_callback = EvalCallback(eval_vecenv, n_eval_episodes=1, eval_freq=save_and_eval_every)

    ppo = MyPPO(
        "MlpPolicy",
        train_vecenv,
        policy_kwargs={"log_std_init": -2, "net_arch": [64, 64]},
        learning_rate=3e-4,
        max_grad_norm=0.1,
        batch_size=16384,
        n_epochs=10,
        n_steps=episode_length,
    )
    ppo.save(f"{outdir}/rl_model_{ppo.num_timesteps}.zip")
    with open(f"{outdir}/args.json", "w") as f:
        json.dump(args.__dict__, f)
    ppo.learn(
        total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        tb_log_name="",
        progress_bar=True,
    )
    ppo.save(f"{outdir}/rl_model_{ppo.num_timesteps}.zip")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n_iters", type=int, default=100)
    parser.add_argument("--episode_length", type=int, default=256)
    parser.add_argument("--log_std_init", type=float, default=-2.0)
    args, remaining_args = parser.parse_known_args()
    main(args, remaining_args)
