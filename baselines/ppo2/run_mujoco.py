#!/usr/bin/env python3
import argparse
from baselines.common.cmd_util import mujoco_arg_parser
from baselines import bench, logger

def train(env_id, num_timesteps, seed, policy, r_ex_coef, r_in_coef, lr_alpha, lr_beta, reward_freq):
    from baselines.common import set_global_seeds
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.ppo2 import ppo2
    from baselines.ppo2.policies import MlpPolicy, MlpPolicyIntrinsicReward
    import gym
    import tensorflow as tf
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True
    tf.Session(config=config).__enter__()
    def make_env():
        env = gym.make(env_id)
        env = bench.Monitor(env, logger.get_dir())
        return env
    env = DummyVecEnv([make_env])
    env = VecNormalize(env)

    set_global_seeds(seed)
    if policy == 'mlp':
        policy = MlpPolicy
    elif policy == 'mlp_int':
        policy = MlpPolicyIntrinsicReward
    else:
        raise NotImplementedError
    ppo2.learn(policy=policy, env=env, nsteps=2048, nminibatches=32,
        lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
        ent_coef=0.0,
        lr_alpha=lr_alpha,
        cliprange=0.2,
        total_timesteps=num_timesteps,
        r_ex_coef=r_ex_coef,
        r_in_coef=r_in_coef,
        lr_beta=lr_beta,
        reward_freq=reward_freq)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='Environment ID', default='Walker2d-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--policy', help='Policy architecture', choices=['mlp', 'mlp_int'], default='mlp_int')
    parser.add_argument('--num-timesteps', type=int, default=int(1E6))
    parser.add_argument('--r-ex-coef', type=float, default=0)
    parser.add_argument('--r-in-coef', type=float, default=1)
    parser.add_argument('--lr-alpha', type=float, default=3E-4)
    parser.add_argument('--lr-beta', type=float, default=1E-4)
    parser.add_argument('--reward-freq', type=int, default=20)
    args = parser.parse_args()
    logger.configure()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed, policy=args.policy,
          r_ex_coef=args.r_ex_coef, r_in_coef=args.r_in_coef,
          lr_alpha=args.lr_alpha, lr_beta=args.lr_beta,
          reward_freq=args.reward_freq)


if __name__ == '__main__':
    main()
