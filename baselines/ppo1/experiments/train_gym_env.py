#!/usr/bin/env python
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import os.path as osp
import gym, logging
from baselines import logger
import sys
import tensorflow as tf
from argparse import ArgumentParser
import os

def train(exp_name, env_id, max_iters, save_step, seed):
    from baselines.ppo1 import mlp_policy, pposgd_simple
    sess = U.make_session(num_cpu=1)
    sess.__enter__()
#    logger.session().__enter__()
    set_global_seeds(seed)
    env = gym.make(env_id)
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    hid_size=64, num_hid_layers=2)
#    env = bench.Monitor(env, osp.join(logger.get_dir(), "monitor.json"))
    env.seed(seed)
    gym.logger.setLevel(logging.WARN)

    scriptpath = os.path.dirname(os.path.abspath( __file__ ))
    directory = os.path.join(scriptpath, exp_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, "")

    pposgd_simple.learn(env, policy_fn,
            max_iters=max_iters,
            filepath=filepath,
            save_step=save_step,
            timesteps_per_actorbatch=4000,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95,
        )
    env.close()

def main():
    parser = ArgumentParser()
    parser.add_argument('exp_name', type=str, help='Required experiment name to avoid confusion.')
    parser.add_argument('env', type=str,
                        help="Gym environment id.")
    parser.add_argument('--max_iters', type=int, default=20001,
                        help='Number of PPO1 algo iterations before experiment finished')
    parser.add_argument('--save_step', type=int, default=100,
                        help='Indicates every how many episodes the model is to be saved.')
    parser.add_argument('--seed', type=int, default=1,
                        help='Global RNG seed.')
    args = parser.parse_args()
    train(args.exp_name, args.env, args.max_iters, args.save_step, seed=args.seed)

if __name__ == "__main__":
    main()
