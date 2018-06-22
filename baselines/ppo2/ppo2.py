import os
import time
import joblib
import numpy as np
import os.path as osp
import tensorflow as tf
from baselines import logger
from collections import deque
from baselines.common import explained_variance

class Model(object):
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm, r_ex_coef, r_in_coef):
        sess = tf.get_default_session()
        nbatch = nbatch_act * nsteps

        act_model = policy(sess, ob_space, ac_space, nbatch_act, 1, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nbatch_train, nsteps, reuse=True)

        A = train_model.pdtype.sample_placeholder([nbatch_train], "A")
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [nbatch_train], "OLDNEGLOGPAC")
        R_EX = tf.placeholder(tf.float32, [nbatch], "R_EX")
        ADV_EX = tf.placeholder(tf.float32, [nbatch_train], "ADV_EX")
        RET_EX = tf.placeholder(tf.float32, [nbatch_train], "RET_EX")
        OLDV_EX = tf.placeholder(tf.float32, [nbatch_train], "OLDV_EX")
        OLDV_MIX = tf.placeholder(tf.float32, [nbatch_train], "OLDV_MIX")
        TD_MIX = tf.placeholder(tf.float32, [nbatch], "TD_MIX")
        COEF_MAT = tf.placeholder(tf.float32, [nbatch_train, nbatch], "COEF_MAT")
        CLIPRANGE = tf.placeholder(tf.float32, [])
        LR_ALPHA = tf.placeholder(tf.float32, [], "LR_ALPHA")
        LR_BETA = tf.placeholder(tf.float32, [], "LR_BETA")

        # Simulate GAE.
        delta_mix = r_in_coef * train_model.r_in + r_ex_coef * R_EX + TD_MIX
        adv_mix = tf.squeeze(tf.matmul(COEF_MAT, tf.reshape(delta_mix, [nbatch, 1])), [1])
        ret_mix = adv_mix + OLDV_MIX
        adv_mix_mean, adv_mix_var = tf.nn.moments(adv_mix, axes=0)
        adv_mix = (adv_mix - adv_mix_mean) / (tf.sqrt(adv_mix_var) + 1E-8)

        neglogpac = train_model.pd.neglogp(A)
        entropy = tf.reduce_mean(train_model.pd.entropy())

        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        pg_mix_loss1 = -adv_mix * ratio
        pg_mix_loss2 = -adv_mix * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_mix_loss = tf.reduce_mean(tf.maximum(pg_mix_loss1, pg_mix_loss2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio- 1.0), CLIPRANGE)))
        v_mix = train_model.v_mix
        v_mix_clipped = OLDV_MIX + tf.clip_by_value(v_mix - OLDV_MIX, - CLIPRANGE, CLIPRANGE)
        v_mix_loss1 = tf.square(v_mix - ret_mix)
        v_mix_loss2 = tf.square(v_mix_clipped - ret_mix)
        v_mix_loss = .5 * tf.reduce_mean(tf.maximum(v_mix_loss1, v_mix_loss2))
        policy_loss = pg_mix_loss - entropy * ent_coef + v_mix_loss * vf_coef
        policy_params = tf.trainable_variables("policy")
        policy_grads = tf.gradients(policy_loss, policy_params)
        if max_grad_norm is not None:
            policy_grads, policy_grad_norm = tf.clip_by_global_norm(policy_grads, max_grad_norm)
        policy_grads_and_vars = list(zip(policy_grads, policy_params))
        policy_trainer = tf.train.AdamOptimizer(learning_rate=LR_ALPHA, epsilon=1e-5)
        policy_train = policy_trainer.apply_gradients(policy_grads_and_vars)

        beta1_power, beta2_power = policy_trainer._get_beta_accumulators()
        policy_params_new = {}
        for var, grad in zip(policy_params, policy_grads):
            lr_ = LR_ALPHA * tf.sqrt(1 - beta2_power) / (1 - beta1_power)
            m, v = policy_trainer.get_slot(var, 'm'), policy_trainer.get_slot(var, 'v')
            m = m + (grad - m) * (1 - .9)
            v = v + (tf.square(tf.stop_gradient(grad)) - v) * (1 - .999)
            policy_params_new[var.name] = var - m * lr_ / (tf.sqrt(v) + 1E-5)
        policy_new = train_model.policy_new_fn(policy_params_new, ob_space, ac_space, nbatch_train, nsteps)

        neglogpac_new = policy_new.pd.neglogp(A)
        ratio_new = tf.exp(OLDNEGLOGPAC - neglogpac_new)
        pg_ex_loss1 = -ADV_EX * ratio_new
        pg_ex_loss2 = -ADV_EX * tf.clip_by_value(ratio_new, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_ex_loss = tf.reduce_mean(tf.maximum(pg_ex_loss1, pg_ex_loss2))
        v_ex = train_model.v_ex
        v_ex_clipped = OLDV_EX + tf.clip_by_value(v_ex - OLDV_EX, - CLIPRANGE, CLIPRANGE)
        v_ex_loss1 = tf.square(v_ex - RET_EX)
        v_ex_loss2 = tf.square(v_ex_clipped - RET_EX)
        v_ex_loss = .5 * tf.reduce_mean(tf.maximum(v_ex_loss1, v_ex_loss2))
        intrinsic_loss = pg_ex_loss + vf_coef * v_ex_loss
        intrinsic_params = tf.trainable_variables("intrinsic")
        intrinsic_grads = tf.gradients(intrinsic_loss, intrinsic_params)
        if max_grad_norm is not None:
            intrinsic_grads, intrinsic_grad_norm = tf.clip_by_global_norm(intrinsic_grads, max_grad_norm)
        intrinsic_grads_and_vars = list(zip(intrinsic_grads, intrinsic_params))
        intrinsic_trainer = tf.train.AdamOptimizer(learning_rate=LR_BETA, epsilon=1E-5)
        intrinsic_train = intrinsic_trainer.apply_gradients(intrinsic_grads_and_vars)

        all_params = tf.global_variables()

        def train(obs, obs_all, actions, actions_all, neglogpacs, states, masks,
                  r_ex, ret_ex, v_ex, td_mix, v_mix, coef_mat, lr_alpha, lr_beta, cliprange):
            adv_ex = ret_ex - v_ex
            adv_ex = (adv_ex - adv_ex.mean()) / (adv_ex.std() + 1e-8)
            td_map = {train_model.X:obs, train_model.X_ALL:obs_all, policy_new.X:obs,
                      A:actions, train_model.A_ALL:actions_all, OLDNEGLOGPAC:neglogpacs,
                      R_EX:r_ex, ADV_EX:adv_ex, RET_EX:ret_ex, OLDV_EX:v_ex, OLDV_MIX:v_mix, TD_MIX:td_mix,
                      COEF_MAT:coef_mat, CLIPRANGE:cliprange, LR_ALPHA:lr_alpha, LR_BETA:lr_beta}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            return sess.run(
                [entropy, approxkl, clipfrac, policy_train, intrinsic_train],
                td_map
            )[:-2]

        def save(save_path):
            ps = sess.run(all_params)
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(all_params, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)
            # If you want to load weights, also save/load observation scaling inside VecNormalize

        self.train = train
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.intrinsic_reward = act_model.intrinsic_reward
        self.initial_state = act_model.initial_state
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess) #pylint: disable=E1101

class Runner(object):

    def __init__(self, *, env, model, nsteps, gamma, lam, r_ex_coef, r_in_coef, reward_freq):
        self.env = env
        self.model = model
        nenv = env.num_envs
        self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=model.train_model.X.dtype.name)
        self.obs[:] = env.reset()
        self.gamma = gamma
        self.lam = lam
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]
        self.r_ex_coef = r_ex_coef
        self.r_in_coef = r_in_coef
        self.reward_freq = reward_freq
        self.delay_r_ex = np.zeros([nenv])
        self.delay_step = np.zeros([nenv])
        self.ep_r_in = np.zeros([nenv])
        self.ep_r_ex = np.zeros([nenv])
        self.ep_len = np.zeros([nenv])

    def run(self):
        mb_obs, mb_r_ex, mb_r_in, mb_ac, mb_v_ex, mb_v_mix, mb_dones, mb_neglogpacs = [],[],[],[],[],[],[],[]
        mb_states = self.states
        epinfos, ep_r_ex, ep_r_in, ep_len = [],[],[],[]
        for _ in range(self.nsteps):
            ac, v_ex, v_mix, self.states, neglogpacs = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(self.obs.copy())
            mb_ac.append(ac)
            mb_v_ex.append(v_ex)
            mb_v_mix.append(v_mix)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            obs, r_ex, self.dones, infos = self.env.step(ac)
            self.delay_r_ex += r_ex
            self.delay_step += 1
            for n, done in enumerate(self.dones):
                if done or self.delay_step[n] == self.reward_freq:
                    r_ex[n] = self.delay_r_ex[n]
                    self.delay_r_ex[n] = self.delay_step[n] = 0
                else:
                    r_ex[n] = 0
            mb_r_ex.append(r_ex)
            r_in = self.model.intrinsic_reward(self.obs, ac)
            mb_r_in.append(r_in)
            self.ep_r_ex += r_ex
            self.ep_r_in += r_in
            self.ep_len += 1
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            for n, done in enumerate(self.dones):
                if done:
                    self.obs[n] = self.obs[n]*0
                    ep_r_ex.append(self.ep_r_ex[n])
                    ep_r_in.append(self.ep_r_in[n])
                    ep_len.append(self.ep_len[n])
                    self.ep_r_ex[n], self.ep_r_in[n], self.ep_len[n] = 0,0,0
            self.obs = obs
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_ac = np.asarray(mb_ac)
        mb_r_ex = np.asarray(mb_r_ex, dtype=np.float32)
        mb_r_in = np.asarray(mb_r_in, dtype=np.float32)
        mb_r_mix = self.r_ex_coef * mb_r_ex + self.r_in_coef * mb_r_in
        mb_v_ex = np.asarray(mb_v_ex, dtype=np.float32)
        mb_v_mix = np.asarray(mb_v_mix, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_v_ex, last_v_mix = self.model.value(self.obs, self.states, self.dones)
        mb_v_mix_next = np.zeros_like(mb_v_mix)
        mb_v_mix_next[:-1] = mb_v_mix[1:] * (1.0 - mb_dones[1:])
        mb_v_mix_next[-1] = last_v_mix * (1.0 - self.dones)
        td_mix = self.gamma * mb_v_mix_next - mb_v_mix
        #discount/bootstrap off value fn
        mb_adv_ex = np.zeros_like(mb_r_ex)
        mb_adv_mix = np.zeros_like(mb_r_mix)
        lastgaelam_ex, lastgaelam_mix = 0,0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextv_ex = last_v_ex
                nextv_mix = last_v_mix
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextv_ex = mb_v_ex[t+1]
                nextv_mix = mb_v_mix[t+1]
            delta_ex = mb_r_ex[t] + self.gamma * nextv_ex * nextnonterminal - mb_v_ex[t]
            delta_mix = mb_r_mix[t] + self.gamma * nextv_mix * nextnonterminal - mb_v_mix[t]
            mb_adv_ex[t] = lastgaelam_ex = delta_ex + self.gamma * self.lam * nextnonterminal * lastgaelam_ex
            mb_adv_mix[t] = lastgaelam_mix = delta_mix + self.gamma * self.lam * nextnonterminal * lastgaelam_mix
        mb_ret_ex = mb_adv_ex + mb_v_ex
        mb_ret_mix = mb_adv_mix + mb_v_mix
        return (*map(sf01, (mb_obs, mb_dones, mb_ac, mb_neglogpacs,
                            mb_r_ex, mb_r_in, mb_ret_ex, mb_ret_mix, mb_v_ex, mb_v_mix, td_mix)),
            mb_states, epinfos, ep_r_ex, ep_r_in, ep_len)

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

def constfn(val):
    def f(_):
        return val
    return f

def learn(*, policy, env, nsteps, total_timesteps, ent_coef, lr_alpha,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=0, r_ex_coef=0, r_in_coef=1, lr_beta=1E-4, reward_freq=1):

    if isinstance(lr_alpha, float): lr_alpha = constfn(lr_alpha)
    else: assert callable(lr_alpha)
    if isinstance(lr_beta, float): lr_beta = constfn(lr_beta)
    else: assert callable(lr_beta)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches

    make_model = lambda : Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm, r_ex_coef=r_ex_coef, r_in_coef=r_in_coef)
    if save_interval and logger.get_dir():
        import cloudpickle
        with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
            fh.write(cloudpickle.dumps(make_model))
    model = make_model()
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam,
                    r_ex_coef=r_ex_coef, r_in_coef=r_in_coef, reward_freq=reward_freq)

    epinfobuf = deque(maxlen=100)
    eprinbuf = deque(maxlen=100)
    tfirststart = time.time()

    nupdates = total_timesteps//nbatch
    for update in range(1, nupdates+1):
        assert nbatch % nminibatches == 0
        nbatch_train = nbatch // nminibatches
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        cur_lr_alpha = lr_alpha(frac)
        cur_lr_beta = lr_beta(frac)
        cur_cliprange = cliprange(frac)
        obs, masks, actions, neglogpacs, r_ex, r_in, ret_ex, ret_mix, v_ex, v_mix, td_mix, states,\
        epinfos, ep_r_ex, ep_r_in, ep_len = runner.run() #pylint: disable=E0632
        epinfobuf.extend(epinfos)
        eprinbuf.extend(ep_r_in)
        mblossvals = []
        if states is None: # nonrecurrent version
            inds = np.arange(nbatch)
            for _ in range(noptepochs):
                np.random.shuffle(inds)
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    coef_mat = np.zeros([nbatch_train, nbatch], "float32")
                    for i in range(nbatch_train):
                        coef = 1.0
                        for j in range(mbinds[i], nbatch):
                            if j > mbinds[i] and (masks[j] or j % nsteps == 0):
                                break
                            coef_mat[i][j] = coef
                            coef *= gamma * lam
                    entropy, approxkl, clipfrac = model.train(obs[mbinds], obs, actions[mbinds], actions, neglogpacs[mbinds],
                                None, masks[mbinds], r_ex, ret_ex[mbinds], v_ex[mbinds], td_mix,
                                v_mix[mbinds], coef_mat, cur_lr_alpha, cur_lr_beta, cur_cliprange)
        else: # recurrent version
            raise NotImplementedError

        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.time()
        fps = int(nbatch / (tnow - tstart))
        if update % log_interval == 0 or update == 1:
            logger.logkv("serial_timesteps", update*nsteps)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update*nbatch)
            logger.logkv("fps", fps)
            logger.logkv('gamescoremean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('gamelenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.logkv('time_elapsed', tnow - tfirststart)
            v_ex_ev = explained_variance(v_ex, ret_ex)
            logger.logkv("v_ex_ev", float(v_ex_ev))
            v_mix_ev = explained_variance(v_mix, ret_mix)
            logger.logkv("v_mix_ev", float(v_mix_ev))
            logger.logkv("entropy", float(entropy))
            logger.logkv("approxkl", float(approxkl))
            logger.logkv("clipfrac", float(clipfrac))
            logger.dumpkvs()
        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
            checkdir = osp.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, '%.5i'%update)
            print('Saving to', savepath)
            model.save(savepath)
    env.close()

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)
