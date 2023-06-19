import argparse
import os
import sys
base_path = os.getcwd()
sys.path.append(base_path)
from configs.config import *
import copy
import torch as th
import rlkit.torch.pytorch_util as ptu
import numpy as np
import warnings
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

from Stage2.envs import Truss
from models import MLP, TanhGaussianPolicy, MakeDeterministic, TRANSFORMEREMBED, GNNEMBED, TRANSFORMEREMBED_policy, GNNEMBED_policy

class SizeEnvReplayBuffer(EnvReplayBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def random_batch(self, batch_size):
        indices = np.random.choice(self._size, size=batch_size, replace=self._replace or self._size < batch_size)
        if not self._replace and self._size < batch_size:
            warnings.warn('Replace was set to false, but is temporarily set to true because batch size is larger than current size of replay.')
        batch = dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
        )
        for key in self._env_info_keys:
            assert key not in batch.keys()
            batch[key] = self._env_infos[key][indices]
        return batch

if __name__ == '__main__':
    parser = get_base_config()
    args = parser.parse_known_args(sys.argv[1:])[0]
    config = make_config(args.config)
    for k, v in config.get("base", {}).items():
        if f"--{k}" not in args:
            setattr(args, k, v)
    print(args)
    print(args.save_model_path)
    if not os.path.exists('saved_models'):
        os.mkdir('saved_models')
    if not os.path.exists(args.save_model_path):
        os.mkdir(args.save_model_path)
    args.save_model_path = os.path.join(args.save_model_path, args.run_id)
    if not os.path.exists(args.save_model_path):
        os.mkdir(args.save_model_path)
    if th.cuda.is_available():
        ptu.set_gpu_mode(True)
    env = NormalizedBoxEnv(Truss(args, args.maxp, os.path.join(args.input_path_2, args.run_id),
                                 args.coordinate_range, args.area_range,
                                 args.coordinate_delta_range, args.area_delta_range,
                                 args.fixed_points, args.variable_edges,
                                 args.max_refine_steps, dimension=args.env_dims, reward_lambda=args.reward_lambda))

    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size

    #qf1 = MLP(obs_dim + action_dim, args.hidden_dims)
    #qf2 = MLP(obs_dim + action_dim, args.hidden_dims)
    if (args.EmbeddingBackbone == 'Transformer'):
        qf1 = TRANSFORMEREMBED(args.prev_dims, args.hidden_dims, args.env_dims, args.env_mode, src_mask=True)
        qf2 = TRANSFORMEREMBED(args.prev_dims, args.hidden_dims, args.env_dims, args.env_mode, src_mask=True)
    elif (args.EmbeddingBackbone == 'GNN'):
        qf1 = GNNEMBED(args.prev_dims, args.hidden_dims, args.env_dims, args.env_mode, args.maxp)
        qf2 = GNNEMBED(args.prev_dims, args.hidden_dims, args.env_dims, args.env_mode, args.maxp)

    target_qf1 = copy.deepcopy(qf1)
    target_qf2 = copy.deepcopy(qf2)
    
    if (args.EmbeddingBackbone == 'Transformer'):
        expl_policy = TRANSFORMEREMBED_policy(args.prev_dims, action_dim, args.hidden_dims, args.env_dims, args.env_mode, num_point=args.maxp, src_mask=True)
    elif (args.EmbeddingBackbone == 'GNN'):
        expl_policy = GNNEMBED_policy(args.prev_dims, action_dim, args.hidden_dims, args.env_dims, args.env_mode, num_point=args.maxp)

    #expl_policy = TanhGaussianPolicy(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=args.hidden_dims)
    
    if args.finetune:
        print("load pretrain")
        expl_policy.load_state_dict(th.load("{}/policy.th".format(args.finetune_model_path)))
        qf1.load_state_dict(th.load("{}/qf1.th".format(args.finetune_model_path)))
        qf2.load_state_dict(th.load("{}/qf2.th".format(args.finetune_model_path)))
        target_qf1.load_state_dict(th.load("{}/target_qf1.th".format(args.finetune_model_path)))
        target_qf2.load_state_dict(th.load("{}/target_qf2.th".format(args.finetune_model_path)))

    eval_policy = MakeDeterministic(expl_policy)

    if args.eval:
        expl_policy = MakeDeterministic(expl_policy)
        trainer = SACTrainer(env=env, policy=expl_policy, qf1=qf1, qf2=qf2, target_qf1=target_qf1, target_qf2=target_qf2,
                         soft_target_tau=0.005, reward_scale=1, policy_lr=0.0000, qf_lr=0.0000)
    else: trainer = SACTrainer(env=env, policy=expl_policy, qf1=qf1, qf2=qf2, target_qf1=target_qf1, target_qf2=target_qf2,
                         soft_target_tau=0.005, reward_scale=1, policy_lr=0.0003, qf_lr=0.0003)

    expl_path_collector = MdpPathCollector(env, expl_policy)
    eval_path_collector = MdpPathCollector(env, eval_policy)
    replay_buffer = EnvReplayBuffer(args.buffer_size, env)
    algorithm = TorchBatchRLAlgorithm(trainer=trainer, exploration_env=env, evaluation_env=env,
                                      exploration_data_collector=expl_path_collector,
                                      evaluation_data_collector=eval_path_collector,
                                      replay_buffer=replay_buffer,
                                      num_epochs=args.epoch,
                                      num_eval_steps_per_epoch=2000,
                                      num_trains_per_train_loop=args.num_trains_per_train_loop,
                                      num_train_loops_per_epoch=args.num_train_loops_per_epoch,
                                      num_expl_steps_per_train_loop=1000,
                                      min_num_steps_before_training=1000,
                                      max_path_length=500,
                                      batch_size=args.batch_size, eval = args.eval)
    algorithm.to(ptu.device)
    algorithm.train()

    trained_network = algorithm.trainer.networks
    th.save(trained_network[0].state_dict(), "{}/policy.th".format(args.save_model_path))
    th.save(trained_network[1].state_dict(), "{}/qf1.th".format(args.save_model_path))
    th.save(trained_network[2].state_dict(), "{}/qf2.th".format(args.save_model_path))
    th.save(trained_network[3].state_dict(), "{}/target_qf1.th".format(args.save_model_path))
    th.save(trained_network[4].state_dict(), "{}/target_qf2.th".format(args.save_model_path))
