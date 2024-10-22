import glob
from argparse import ArgumentParser
from stable_baselines3 import PPO
from tqdm import tqdm
from utils.env_containers import EnvContainer
from utils.vecenv import MyVecEnv
import json
import numpy as np
import os
import pandas as pd

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def main(args, remaining_args):
    with open(f"{args.run_dir}/args.json", "r") as f:
        saved_args = json.load(f)
    outdir = os.path.join("out", saved_args["run_name"], f"{saved_args['seed']:02d}")

    env_name = "halfcheetah"
    backend = "mjx"
    batch_size = 16
    episode_length = 256
    eval_env_container = EnvContainer(env_name, backend, batch_size, False, episode_length)
    eval_vecenv = MyVecEnv(eval_env_container, seed=0)
    policy_paths = sorted(glob.glob(f"{args.run_dir}/rl_model_*.zip"))
    for policy_path in policy_paths:
        ppo = PPO.load(policy_path, eval_vecenv)

        obs = eval_vecenv.reset()
        rewards = []
        for _ in tqdm(range(episode_length)):
            action, _ = ppo.predict(obs, deterministic=True)
            obs, reward, _, _ = eval_vecenv.step(action)
            rewards.append(reward)
        rewards = np.stack(rewards, axis=-1)
        mean_eval_reward = np.mean(rewards)

        df = pd.DataFrame()
        df["run_name"] = [saved_args["run_name"]]
        df["seed"] = [saved_args["seed"]]
        df["num_timesteps"] = [ppo.num_timesteps]
        df["mean_eval_reward"] = [mean_eval_reward]

        df.to_csv(f"{outdir}/eval_{ppo.num_timesteps}.csv", index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args, remaining_args = parser.parse_known_args()
    main(args, remaining_args)
