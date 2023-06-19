import argparse
import copy
import os
import torch as th
import rlkit.torch.pytorch_util as ptu
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

from envs import Truss
from models import MLP, TanhGaussianPolicy, MakeDeterministic, TRANSFORMEREMBED, EmbedTanhGaussianPolicy

import gtimer as gt

def get_args():
    parser = argparse.ArgumentParser(description='Alpha Truss')
    # Training args
    parser.add_argument('--num_trains_per_train_loop', type=int, default=5, help='for sac training')
    parser.add_argument('--num_train_loops_per_epoch', type=int, default=5, help='for sac training')
    parser.add_argument('--hidden_dims', type=list, default=[256, 1024], help='hidden layer dimensions')
    parser.add_argument('--buffer_size', type=int, default=1000000, help='buffer size')
    parser.add_argument('--epoch', type=int, default=50, help='epoch')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--evaluation', action='store_true', default=False)
    parser.add_argument('--save_model_path', type=str, default='.')
    parser.add_argument('--prev_dims', type=list, default=[8, 32], help='input dims for TransformerEmbed')
    parser.add_argument('--post_dims', type=list, default=[256, 128], help='hidden dims for TransformerEmbed')
    # Env args
    parser.add_argument('--num_points', type=int, default=9, help='number of nodes')
    parser.add_argument('--initial_state_files', type=str, default='best_results/TrainMax9p_1/', help='input file for refine')
    parser.add_argument('--coordinate_range', type=list, default=[(0.0, 18.288), (0.0, 9.144)], help='nodes\' coordinate range')
    parser.add_argument('--area_range', type=list, default=(6.452e-05, 0.04), help='edges\' area range')
    parser.add_argument('--coordinate_delta_range', type=list, default=[(-0.5715, 0.5715), (-0.5715, 0.5715)], help='nodes\' coordinate delta range')
    parser.add_argument('--area_delta_range', type=list, default=(-0.0005, 0.0005), help='edges\' area delta range')
    parser.add_argument('--fixed_points', type=int, default=4, help='number of fixed nodes')
    parser.add_argument('--variable_edges', type=int, default=-1, help='number of variable edges, -1 if all is variable')
    parser.add_argument('--max_refine_steps', type=int, default=200, help='maximum timesteps of an episode')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    point_num = args.num_points
    #args.save_model_path = args.initial_state_files.replace('best_results', 'saved_models')
    #args.save_model_path = args.save_model_path[:-1]
    args.save_model_path = 'saved_models/TrainMax9p_1'
    #args.num_points = int(point_num)
    if not os.path.exists(args.save_model_path):
        os.mkdir(args.save_model_path)
    UseEvalModel = False
    if UseEvalModel:
        args.save_model_path = 'saved_models/Noise6789p_2'
    if th.cuda.is_available():
        ptu.set_gpu_mode(True)
    env = NormalizedBoxEnv(Truss(args.num_points, args.initial_state_files,
                                 args.coordinate_range, args.area_range,
                                 args.coordinate_delta_range, args.area_delta_range,
                                 args.fixed_points, args.variable_edges,
                                 args.max_refine_steps))

    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size
    print(obs_dim, action_dim)
    #qf1 = MLP(obs_dim + action_dim, args.hidden_dims)
    #qf2 = MLP(obs_dim + action_dim, args.hidden_dims)
    qf1 = TRANSFORMEREMBED(args.prev_dims, args.hidden_dims)
    qf2 = TRANSFORMEREMBED(args.prev_dims, args.hidden_dims)
    target_qf1 = copy.deepcopy(qf1)
    target_qf2 = copy.deepcopy(qf2)
    expl_policy = EmbedTanhGaussianPolicy(obs_dim=args.prev_dims[-1], action_dim=action_dim, hidden_sizes=args.hidden_dims, input_dims=args.prev_dims)
    eval_policy = MakeDeterministic(expl_policy)
    if os.path.exists("{}/qf1.th".format(args.save_model_path)):
        args.evaluation = True
    if args.evaluation:
        print("load pretrain")
        expl_policy.load_state_dict(th.load("{}/policy.th".format(args.save_model_path)))
        qf1.load_state_dict(th.load("{}/qf1.th".format(args.save_model_path)))
        qf2.load_state_dict(th.load("{}/qf2.th".format(args.save_model_path)))
        target_qf1.load_state_dict(th.load("{}/target_qf1.th".format(args.save_model_path)))
        target_qf2.load_state_dict(th.load("{}/target_qf2.th".format(args.save_model_path)))


    expl_path_collector = MdpPathCollector(env, expl_policy)
    eval_path_collector = MdpPathCollector(env, eval_policy)
    replay_buffer = EnvReplayBuffer(args.buffer_size, env)
    trainer = SACTrainer(env=env, policy=expl_policy, qf1=qf1, qf2=qf2, target_qf1=target_qf1, target_qf2=target_qf2,
                         soft_target_tau=0.005, reward_scale=10., policy_lr=0.0003, qf_lr=0.0003)
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
                                      batch_size=args.batch_size)
    if not UseEvalModel:
        gt.reset_root()
    algorithm.to(ptu.device)
    algorithm.train()

    if not UseEvalModel:
        trained_network = algorithm.trainer.networks
        th.save(trained_network[0].state_dict(), "{}/policy.th".format(args.save_model_path))
        th.save(trained_network[1].state_dict(), "{}/qf1.th".format(args.save_model_path))
        th.save(trained_network[2].state_dict(), "{}/qf2.th".format(args.save_model_path))
        th.save(trained_network[3].state_dict(), "{}/target_qf1.th".format(args.save_model_path))
        th.save(trained_network[4].state_dict(), "{}/target_qf2.th".format(args.save_model_path))
