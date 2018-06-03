# !/usr/bin/env python
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import os.path as osp
import gym, logging
from baselines import logger
import sys
import tensorflow as tf
from argparse import ArgumentParser

def combine_with_duplicate(root, rel_path):
    rs = root.split("/")
    rps = rel_path.split("/")
    popped = False
    for v in rs:
        if v == rps[0]:
            rps.pop(0)
            popped = True
        elif popped:
            break

    return "/".join(rs+rps)

import os

def enjoy(filepath, env_id, episodes, seed, analyze, render_mode):
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
    obs = env.reset()
    env.seed(seed)
    gym.logger.setLevel(logging.WARN)
    pi = policy_fn('pi', env.observation_space, env.action_space)

    script_path = os.path.dirname(os.path.abspath( __file__ ))
    filepath = combine_with_duplicate(script_path, filepath)
    tf.train.Saver().restore(sess, filepath)
    episode = 0
    if analyze:
        goals_reached = 0
        ep_steps_goal_reached = []
        dist_mins = []
        diff_angles = []

    while episode < episodes:
        action = pi.act(True, obs)[0]
        obs, reward, done, info = env.step(action)
        env.render(render_mode)

        if done:
            if analyze:
                if info["goal"] == True:
                    goals_reached += 1
                    ep_steps_goal_reached.append(env.steps)
                    diff_angles.append(env.get_diff_angle(degree=True))
                else:
                    dist_mins.append(env.dist_min)

            env.reset()
            episode += 1

    if analyze:
        eps = 1e-10
        print("Goals reached {:3d}/{:3d}".format(goals_reached, episodes))
#        print("Goals reached after retries {:3d}/{:3d}".format(goals_reached+goals_reached_after_retries, total_episodes))
        print("Average steps needed to reach goal: {:5.1f}.".format(sum(ep_steps_goal_reached)/(len(ep_steps_goal_reached)+eps)))
        print("Average min distance from goal within one episode, when goal not reached: {:5.2f}mm.".format(1000*sum(dist_mins)/(len(dist_mins)+eps)))
        print("Average angle difference when goal reached: {:5.2f}°.".format(sum(diff_angles)/(len(diff_angles)+eps)))

def main():
    parser = ArgumentParser()
    parser.add_argument('filepath', type=str, help='Relative path to tensorflow saved model.')
    parser.add_argument('env', type=str,
                        help='Gym environment id string to be used. Has to be in accordance to filepath.')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of episodes to be simulated.')
    parser.add_argument('--analyze', type=int, default=0,
                        help='True value indicates that rollouts are to be analyzed')
    parser.add_argument('--render_mode', type=str, default='',
                        help='"human" for 3D plot \n \
                        "string"  for simple console output of distance \
                        "" for no rendering.')
    parser.add_argument('--seed', type=int, default=1,
                        help='Global RNG seed.')
    args = parser.parse_args()
    enjoy(args.filepath, args.env, args.episodes, args.seed, args.analyze, args.render_mode)


if __name__ == '__main__':
    main()
