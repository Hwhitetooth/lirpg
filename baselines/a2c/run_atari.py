#!/usr/bin/env python3

from baselines import logger
from baselines.common.cmd_util import make_atari_env, atari_arg_parser
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.a2c.a2c import learn
from baselines.a2c.policies import CnnPolicy, LstmPolicy, LnLstmPolicy, CnnPolicyIntrinsicReward

def train(env_id, num_timesteps, seed, policy, lrschedule, num_env,
          v_ex_coef, r_ex_coef, r_in_coef, lr_alpha, lr_beta):
    if policy == 'cnn':
        policy_fn = CnnPolicy
    elif policy == 'lstm':
        policy_fn = LstmPolicy
    elif policy == 'lnlstm':
        policy_fn = LnLstmPolicy
    elif policy == 'cnn_int':
        policy_fn = CnnPolicyIntrinsicReward
    else:
        raise NotImplementedError
    env = VecFrameStack(make_atari_env(env_id, num_env, seed), 4)
    learn(policy_fn, env, seed, total_timesteps=int(num_timesteps * 1.01), lrschedule=lrschedule,
          v_ex_coef=v_ex_coef, r_ex_coef=r_ex_coef, r_in_coef=r_in_coef,
          lr_alpha=lr_alpha, lr_beta=lr_beta)
    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='Environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm', 'cnn_int'], default='cnn_int')
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='linear')
    parser.add_argument('--num-timesteps', type=int, default=int(50E6))
    parser.add_argument('--v-ex-coef', type=float, default=0.1)
    parser.add_argument('--r-ex-coef', type=float, default=1)
    parser.add_argument('--r-in-coef', type=float, default=0.01)
    parser.add_argument('--lr-alpha', type=float, default=7E-4)
    parser.add_argument('--lr-beta', type=float, default=7E-4)
    args = parser.parse_args()
    logger.configure()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
          policy=args.policy, lrschedule=args.lrschedule, num_env=16,
          v_ex_coef=args.v_ex_coef, r_ex_coef=args.r_ex_coef, r_in_coef=args.r_in_coef,
          lr_alpha=args.lr_alpha, lr_beta=args.lr_beta)

if __name__ == '__main__':
    main()
