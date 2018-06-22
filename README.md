# Learning Intrinsic Rewards for Policy Gradient

This repository is an implementation of [On Learning Intrinsic Rewards for Policy Gradient Methods](https://arxiv.org/abs/1804.06459).
```
@article{zheng2018learning,
  title={On Learning Intrinsic Rewards for Policy Gradient Methods},
  author={Zheng, Zeyu and Oh, Junhyuk and Singh, Satinder},
  journal={arXiv preprint arXiv:1804.06459},
  year={2018}
}
```

## Dependencies
This code is based on [OpenAI baselines](https://github.com/openai/baselines). In addtion, it requires the following:
- Python 3.*
- TensorFlow 1.7.0+

## Training
To run `A2C+LIRPG` on Atari games:
```angular2html
python -m baselines.a2c.run_atari --env BreakoutNoFrameskip-v4
```

To run `PPO+LIRPG` on delayed Mujoco tasks:
```angular2html
python -m baselines.ppo2.run_mujoco --env Hopper-v2 --reward-freq 20
```
