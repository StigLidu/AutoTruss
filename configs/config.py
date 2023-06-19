from typing import Dict
import argparse
import yaml

def get_base_config():
    parser = argparse.ArgumentParser(
        description='RL_truss_layout',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--config', type = str)
# args for UCTs:
    parser.add_argument("--c", default = [30.0, 30.0, 30.0])
    parser.add_argument("--rate2", default = 0.9)
    parser.add_argument("--prob", default = 1.0)
    parser.add_argument("--alpha", default = 0.3)
    parser.add_argument("--rate", default = 0)
    parser.add_argument("--pickalpha", default = 0)
    parser.add_argument("--sgm1", default = 0.0005)
    parser.add_argument("--sgm2", default = 0.5)
    parser.add_argument("--maxnum", default = 25)
    parser.add_argument("--maxson", default = 200)
    parser.add_argument("--UCT-maxiter", type = int, default = 300000)
    parser.add_argument("--UCT_extra_iter_for_point_pos", type = int, default = 250000)
    parser.add_argument("--initson", default = 25)
    parser.add_argument("--USE-VALUE-NETWORK", type = bool, default = 0)

# args for Env:
    parser.add_argument("--bad-attempt-limit", default = 5)
    parser.add_argument("--maxp", default = 10)
    parser.add_argument("--env-dims", default = 3, type = int)
    parser.add_argument("--env-mode", default='DT', choices=['Area', 'DT'])
    parser.add_argument("--useIntersect", default = True)
    parser.add_argument('--coordinate_range', type=list, default=[(0.0, 4.634), (-0.483, 0.7725), (-0.5, 1.0)], help='points\' range')
    parser.add_argument('--area_range', type=list, default=(0.0001, 0.003028), help='edges\' area range')
    parser.add_argument("--len_range", type=list, default = (0.03, 5.0), help='edges\' length range')    
    parser.add_argument('--area_delta_range', type=list, default=(-0.0005, 0.0005), help='edges\' area delta range')
    parser.add_argument('--coordinate_delta_range', type=list, default=[(-0.5715, 0.5715), (-0.5715, 0.5715), (-0.5715, 0.5715)], help='nodes\' coordinate delta range')
    parser.add_argument('--d-range', type=list, default=(0.025, 0.12))
    parser.add_argument('--t-range', type=list, default=(0.0015, 0.005))
    parser.add_argument('--d-delta-range', type=list, default=(-0.05, 0.05))
    parser.add_argument('--t-delta-range', type=list, default=(-0.005, 0.005))
    parser.add_argument("--usePlist", default = False)
    parser.add_argument("--Plist-path", type = str, default = None)
    parser.add_argument("--useAlist", default = True)
    parser.add_argument("--Alist-path", type = str, default = 'input/sectionList3.txt')
    parser.add_argument("--input-path", type = str, default = 'input/kr-sundial-newinput.txt')
    parser.add_argument("--ratio_ring", default = 0.0)
    parser.add_argument('--fixed_points', type=int, default=4, help='number of fixed nodes')
    parser.add_argument('--variable_edges', type=int, default=-1, help='number of variable edges, -1 if all is variable')
    parser.add_argument('--symmetry-build', type=int, default=0)

# args for dynamics:
    parser.add_argument('--E', type = float, default = 1.93*10**11)
    parser.add_argument('--pho', type = float, default = 8.0*10**3)
    parser.add_argument('--sigma-T', type = float, default = 123.0*10**6)
    parser.add_argument('--sigma-C', type = float, default = 123.0*10**6)
    parser.add_argument('--slenderness_ratio_c', type = float, default = 180.0)
    parser.add_argument('--slenderness_ratio_t', type = float, default = 220.0)
    parser.add_argument('--dislimit', type = float, default = 0.002)

    parser.add_argument('--CONSTRAINT-CROSS-EDGE', type = int, default = 1)
    parser.add_argument('--CONSTRAINT-STRESS', type = int, default = 1)
    parser.add_argument('--CONSTRAINT-DIS', type = int, default = 1)
    parser.add_argument('--CONSTRAINT-BUCKLE', type = int, default = 1)
    parser.add_argument('--CONSTRAINT-SLENDERNESS', type = int, default = 1)
    parser.add_argument("--CONSTRAINT-MAX-LENGTH", type = int, default = 1)
    parser.add_argument("--CONSTRAINT-MIN-LENGTH", type = int, default = 1)
    parser.add_argument("--CONSTRAINT-SELF-WEIGHT", type = int, default = 1)
    parser.add_argument("--NEW_CONST....", type = int, default = 1)

# args for save and load:
    parser.add_argument("--save-KR", default = False)
    parser.add_argument("--save-diversity", type = bool, default = True)
    parser.add_argument("--save-path", type = str, default = './results_3d/')
    parser.add_argument("--input-path-2", type = str, default = './PostResults/')
    parser.add_argument("--save-model-path", type = str, default = './saved_models/')
    parser.add_argument("--finetune-model-path", type = str, default = './saved_models/')
    parser.add_argument("--OUTPUT_ALL_THRESHOLD", type = float, default = 4000)
    parser.add_argument("--MASS_OUTPUT_ALL_THRESHOLD", type = float, default = 4000)
    parser.add_argument("--save-invalid-factor", type = int, default = 0)
    parser.add_argument("--run-id", type = str, default = '.')
    parser.add_argument("--logfile-stage1", type = str, default = 'log_stage1.log')
    parser.add_argument("--logfile-stage2", type = str, default = 'log_stage2.log')
    parser.add_argument("--transfer-filefold", type = str, default = 'DIVERSITY_TOPO_result')

# args for Reward:
    parser.add_argument("--reward_lambda", default = 10 * 50 * 50)

# args for RL:    
    parser.add_argument('--initial_state_files', type=str, default='PostResults/', help='input file for refine')
    parser.add_argument('--num_trains_per_train_loop', type=int, default=5, help='for sac training')
    parser.add_argument('--num_train_loops_per_epoch', type=int, default=5, help='for sac training')
    parser.add_argument('--hidden-dims', type=list, default=[256, 512], help='hidden layer dimensions')
    parser.add_argument('--buffer-size', type=int, default=1000000, help='buffer size')
    parser.add_argument('--epoch', type=int, default=40, help='epoch')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--finetune', action='store_true', default=False)
    parser.add_argument('--only-position', type=bool, default = True)
    parser.add_argument('--greedy-upd', type=bool, default = True)
    parser.add_argument('--prev-dims', type=list, default=[128, 256], help='input dims for TransformerEmbed')
    parser.add_argument('--post-dims', type=list, default=[256, 128], help='hidden dims for TransformerEmbed')
    parser.add_argument('--max-refine-steps', type=int, default=20, help='maximum timesteps of an episode')
    parser.add_argument('--EmbeddingBackbone', type = str, default = 'Transformer', help = 'Transformer or GNN')
    parser.add_argument('--max_num_topo_truss', type = int, default = 5)

# args for check:
    parser.add_argument('--check-file', type=str, default=None)

# args for draw:
    parser.add_argument('--draw-file', type=str, default=None)

# args for transfer
    parser.add_argument('--trans-folder-name', type=str, default="")
    return parser

ALL_CONFIGS = {
    # 3D
    'kr_sundial': "configs/input_kr_sundial.yaml",
    # 2D
    'without_buckle_case1': "configs/input_without_buckle_case1.yaml",
    'without_buckle_case2': "configs/input_without_buckle_case2.yaml",
    '17_bar_case': "configs/input_17_bar_case.yaml"
}

def make_config(type_) -> Dict:
    with open(ALL_CONFIGS[type_]) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config