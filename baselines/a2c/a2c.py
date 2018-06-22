import os.path as osp
import gym
import time
import joblib
import logging
import numpy as np
import tensorflow as tf
from baselines import logger

from baselines.common import set_global_seeds, explained_variance
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.atari_wrappers import wrap_deepmind
from baselines.common import tf_util

from baselines.a2c.utils import discount_with_dones
from baselines.a2c.utils import Scheduler, make_path, find_trainable_variables
from baselines.a2c.utils import cat_entropy, mse

from collections import deque

class Model(object):

    def __init__(self, policy, ob_space, ac_space, nenvs, nsteps,
            ent_coef=0.01, v_mix_coef=0.5, max_grad_norm=0.5, lr_alpha=7e-4, lr_beta=7e-4,
            alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear',
            r_ex_coef=1.0, r_in_coef=0.0, v_ex_coef=1.0):

        sess = tf_util.make_session()
        nact = ac_space.n
        nbatch = nenvs*nsteps

        A = tf.placeholder(tf.int32, [nbatch], 'A')
        R_EX = tf.placeholder(tf.float32, [nbatch], 'R_EX')
        ADV_EX = tf.placeholder(tf.float32, [nbatch], 'ADV_EX')
        RET_EX = tf.placeholder(tf.float32, [nbatch], 'RET_EX')
        V_MIX = tf.placeholder(tf.float32, [nbatch], 'V_MIX')
        DIS_V_MIX_LAST = tf.placeholder(tf.float32, [nbatch], 'DIS_V_MIX_LAST')
        COEF_MAT = tf.placeholder(tf.float32, [nbatch, nbatch], 'COEF_MAT')
        LR_ALPHA = tf.placeholder(tf.float32, [], 'LR_ALPHA')
        LR_BETA = tf.placeholder(tf.float32, [], 'LR_BETA')

        step_model = policy(sess, ob_space, ac_space, nenvs, 1, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nenvs*nsteps, nsteps, reuse=True)

        r_mix = r_ex_coef * R_EX + r_in_coef * tf.reduce_sum(train_model.r_in * tf.one_hot(A, nact), axis=1)
        ret_mix = tf.squeeze(tf.matmul(COEF_MAT, tf.reshape(r_mix, [nbatch, 1])), [1]) + DIS_V_MIX_LAST
        adv_mix = ret_mix - V_MIX

        neglogpac = train_model.pd.neglogp(A)
        pg_mix_loss = tf.reduce_mean(adv_mix * neglogpac)
        v_mix_loss = tf.reduce_mean(mse(tf.squeeze(train_model.v_mix), ret_mix))
        entropy = tf.reduce_mean(cat_entropy(train_model.pi))
        policy_loss = pg_mix_loss - ent_coef * entropy + v_mix_coef * v_mix_loss

        policy_params = train_model.policy_params
        policy_grads = tf.gradients(policy_loss, policy_params)
        if max_grad_norm is not None:
            policy_grads, policy_grad_norm = tf.clip_by_global_norm(policy_grads, max_grad_norm)
        policy_grads_and_vars = list(zip(policy_grads, policy_params))
        policy_trainer = tf.train.RMSPropOptimizer(learning_rate=LR_ALPHA, decay=alpha, epsilon=epsilon)
        policy_train = policy_trainer.apply_gradients(policy_grads_and_vars)

        rmss = [policy_trainer.get_slot(var, 'rms') for var in policy_params]
        policy_params_new = {}
        for grad, rms, var in zip(policy_grads, rmss, policy_params):
            ms = rms + (tf.square(grad) - rms) * (1 - alpha)
            policy_params_new[var.name] = var - LR_ALPHA * grad / tf.sqrt(ms + epsilon)
        policy_new = train_model.policy_new_fn(policy_params_new, ob_space, ac_space, nbatch, nsteps)

        neglogpac_new = policy_new.pd.neglogp(A)
        ratio_new = tf.exp(tf.stop_gradient(neglogpac) - neglogpac_new)
        pg_ex_loss = tf.reduce_mean(-ADV_EX * ratio_new)
        v_ex_loss = tf.reduce_mean(mse(tf.squeeze(train_model.v_ex), RET_EX))
        intrinsic_loss = pg_ex_loss + v_ex_coef * v_ex_loss

        intrinsic_params = train_model.intrinsic_params
        intrinsic_grads = tf.gradients(intrinsic_loss, intrinsic_params)
        if max_grad_norm is not None:
            intrinsic_grads, intrinsic_grad_norm = tf.clip_by_global_norm(intrinsic_grads, max_grad_norm)
        intrinsic_grads_and_vars = list(zip(intrinsic_grads, intrinsic_params))
        intrinsic_trainer = tf.train.RMSPropOptimizer(learning_rate=LR_BETA, decay=alpha, epsilon=epsilon)
        intrinsic_train = intrinsic_trainer.apply_gradients(intrinsic_grads_and_vars)

        lr_alpha = Scheduler(v=lr_alpha, nvalues=total_timesteps, schedule=lrschedule)
        lr_beta = Scheduler(v=lr_beta, nvalues=total_timesteps, schedule=lrschedule)

        all_params = tf.global_variables()

        def train(obs, policy_states, masks, actions, r_ex, ret_ex, v_ex, v_mix, dis_v_mix_last, coef_mat):
            advs_ex = ret_ex - v_ex
            for step in range(len(obs)):
                cur_lr_alpha = lr_alpha.value()
                cur_lr_beta= lr_beta.value()
            td_map = {train_model.X:obs, policy_new.X:obs, A:actions, R_EX:r_ex, ADV_EX:advs_ex, RET_EX:ret_ex,
                      V_MIX:v_mix, DIS_V_MIX_LAST:dis_v_mix_last, COEF_MAT:coef_mat,
                      LR_ALPHA:cur_lr_alpha, LR_BETA:cur_lr_beta}
            if policy_states is not None:
                td_map[train_model.PS] = policy_states
                td_map[train_model.M] = masks
            return sess.run(
                [entropy, policy_train, intrinsic_train],
                td_map
            )[0]

        def save(save_path):
            ps = sess.run(all_params)
            make_path(osp.dirname(save_path))
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(all_params, loaded_params):
                restores.append(p.assign(loaded_p))
            ps = sess.run(restores)

        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.intrinsic_reward = step_model.intrinsic_reward
        self.init_policy_state = step_model.init_policy_state
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess)

class Runner(object):

    def __init__(self, env, model, nsteps=5, gamma=0.99, r_ex_coef=1.0, r_in_coef=0.0):
        self.env = env
        self.model = model
        nenv = env.num_envs
        self.batch_ob_shape = (nenv*nsteps,) + env.observation_space.shape
        self.obs = env.reset()
        self.gamma = gamma
        self.nsteps = nsteps
        self.policy_states = model.init_policy_state
        self.dones = [False for _ in range(nenv)]
        self.r_ex_coef = r_ex_coef
        self.r_in_coef = r_in_coef
        self.ep_r_in = np.zeros([nenv])
        self.ep_r_ex = np.zeros([nenv])
        self.ep_len = np.zeros([nenv])

    def run(self):
        mb_obs, mb_r_ex, mb_r_in, mb_ac, mb_v_ex, mb_v_mix, mb_dones = [],[],[],[],[],[],[]
        mb_policy_states = []
        ep_info, ep_r_ex, ep_r_in, ep_len = [],[],[],[]
        for n in range(self.nsteps):
            mb_policy_states.append(self.policy_states)
            ac, v_ex, v_mix, policy_states, _ = self.model.step(self.obs, self.policy_states, self.dones)
            mb_obs.append(np.copy(self.obs))
            mb_ac.append(ac)
            mb_v_ex.append(v_ex)
            mb_v_mix.append(v_mix)
            mb_dones.append(self.dones)
            obs, r_ex, dones, infos = self.env.step(ac)
            r_in = self.model.intrinsic_reward(self.obs, ac)
            mb_r_ex.append(r_ex)
            mb_r_in.append(r_in)
            self.policy_states = policy_states
            self.dones = dones
            self.ep_r_ex += r_ex
            self.ep_r_in += r_in
            self.ep_len += 1
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo:
                    ep_info.append(maybeepinfo)
            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n]*0
                    ep_r_ex.append(self.ep_r_ex[n])
                    ep_r_in.append(self.ep_r_in[n])
                    ep_len.append(self.ep_len[n])
                    self.ep_r_ex[n], self.ep_r_in[n], self.ep_len[n] = 0,0,0
            self.obs = obs
        mb_dones.append(self.dones)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.float32).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_r_ex = np.asarray(mb_r_ex, dtype=np.float32).swapaxes(1, 0)
        mb_r_in = np.asarray(mb_r_in, dtype=np.float32).swapaxes(1, 0)
        mb_r_mix = self.r_ex_coef * mb_r_ex + self.r_in_coef * mb_r_in
        mb_ac = np.asarray(mb_ac, dtype=np.int32).swapaxes(1, 0)
        mb_v_ex = np.asarray(mb_v_ex, dtype=np.float32).swapaxes(1, 0)
        mb_v_mix = np.asarray(mb_v_mix, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        last_v_ex, last_v_mix = self.model.value(self.obs, self.policy_states, self.dones)
        last_v_ex, last_v_mix = last_v_ex.tolist(), last_v_mix.tolist()
        #discount/bootstrap off value fn
        mb_ret_ex, mb_ret_mix = np.zeros(mb_r_ex.shape), np.zeros(mb_r_mix.shape)
        for n, (r_ex, r_mix, dones, v_ex, v_mix) in enumerate(zip(mb_r_ex, mb_r_mix, mb_dones, last_v_ex, last_v_mix)):
            r_ex, r_mix = r_ex.tolist(), r_mix.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                ret_ex = discount_with_dones(r_ex+[v_ex], dones+[0], self.gamma)[:-1]
                ret_mix = discount_with_dones(r_mix+[v_mix], dones+[0], self.gamma)[:-1]
            else:
                ret_ex = discount_with_dones(r_ex, dones, self.gamma)
                ret_mix = discount_with_dones(r_mix, dones, self.gamma)
            mb_ret_ex[n], mb_ret_mix[n] = ret_ex, ret_mix
        mb_r_ex = mb_r_ex.flatten()
        mb_r_in = mb_r_in.flatten()
        mb_ret_ex = mb_ret_ex.flatten()
        mb_ret_mix = mb_ret_mix.flatten()
        mb_ac = mb_ac.flatten()
        mb_v_ex = mb_v_ex.flatten()
        mb_v_mix = mb_v_mix.flatten()
        mb_masks = mb_masks.flatten()
        mb_dones = mb_dones.flatten()
        return mb_obs, mb_ac, mb_policy_states, mb_r_in, mb_r_ex, mb_ret_ex, mb_ret_mix,\
               mb_v_ex, mb_v_mix, last_v_ex, last_v_mix, mb_masks, mb_dones,\
               ep_info, ep_r_ex, ep_r_in, ep_len

def learn(policy, env, seed, nsteps=5, total_timesteps=int(80e6), v_mix_coef=0.5, ent_coef=0.01, max_grad_norm=0.5,
          lr_alpha=7e-4, lr_beta=7e-4, lrschedule='linear', epsilon=1e-5, alpha=0.99, gamma=0.99, log_interval=100,
          v_ex_coef=1.0, r_ex_coef=0.0, r_in_coef=1.0):
    tf.reset_default_graph()
    set_global_seeds(seed)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps, ent_coef=ent_coef,
                  v_ex_coef=v_ex_coef, max_grad_norm=max_grad_norm, lr_alpha=lr_alpha, lr_beta=lr_beta,
                  alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule,
                  v_mix_coef=v_mix_coef, r_ex_coef=r_ex_coef, r_in_coef=r_in_coef)
    runner = Runner(env, model, nsteps=nsteps, gamma=gamma, r_ex_coef=r_ex_coef, r_in_coef=r_in_coef)

    nbatch = nenvs*nsteps
    tstart = time.time()
    epinfobuf = deque(maxlen=100)
    eprexbuf = deque(maxlen=100)
    eprinbuf = deque(maxlen=100)
    eplenbuf = deque(maxlen=100)
    for update in range(1, total_timesteps//nbatch+1):
        obs, ac, policy_states, r_in, r_ex, ret_ex, ret_mix, \
        v_ex, v_mix, last_v_ex, last_v_mix, masks, dones, \
        epinfo, ep_r_ex, ep_r_in, ep_len = runner.run()
        dis_v_mix_last = np.zeros([nbatch], np.float32)
        coef_mat = np.zeros([nbatch, nbatch], np.float32)
        for i in range(nbatch):
            dis_v_mix_last[i] = gamma ** (nsteps - i % nsteps) * last_v_mix[i // nsteps]
            coef = 1.0
            for j in range(i, nbatch):
                if j > i and j % nsteps == 0:
                    break
                coef_mat[i][j] = coef
                coef *= gamma
                if dones[j]:
                    dis_v_mix_last[i] = 0
                    break
        entropy = model.train(obs, policy_states[0], masks, ac, r_ex, ret_ex, v_ex, v_mix, dis_v_mix_last, coef_mat)

        nseconds = time.time()-tstart
        fps = int((update*nbatch)/nseconds)
        epinfobuf.extend(epinfo)
        eprexbuf.extend(ep_r_ex)
        eprinbuf.extend(ep_r_in)
        eplenbuf.extend(ep_len)
        if update % log_interval == 0 or update == 1:
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update*nbatch)
            logger.record_tabular("fps", fps)
            logger.record_tabular("entropy", float(entropy))
            v_ex_ev = explained_variance(v_ex, ret_ex)
            logger.record_tabular("v_ex_ev", float(v_ex_ev))
            v_mix_ev = explained_variance(v_mix, ret_mix)
            logger.record_tabular("v_mix_ev", float(v_mix_ev))
            logger.record_tabular("gamescoremean", safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.record_tabular("gamelenmean", safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.dump_tabular()
    env.close()

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)
