import matplotlib.pyplot as plt
import numpy as np
import sys, os
base_path = os.getcwd()
sys.path.append(base_path)
from configs.config import get_base_config, make_config
from utils.utils import *
if __name__ == '__main__':
    parser = get_base_config()
    args = parser.parse_known_args(sys.argv[1:])[0]
    from truss_envs.reward import Envs_init, reward_fun
    config = make_config(args.config)
    for k, v in config.get("base", {}).items():
        if f"--{k}" not in args:
            setattr(args, k, v)
    Envs_init(args)
    log_file1 = np.loadtxt(os.path.join(args.save_path, args.run_id, args.logfile_stage1))
    print(log_file1.shape)
    begin_index = 0
    while (log_file1[begin_index][2] > 1000000): begin_index += 1
    plt.plot(log_file1[begin_index:, 0], log_file1[begin_index:, 2])
    plt.savefig(os.path.join(args.save_path, args.run_id, 'graph_stage1.jpg'), dpi = 1000)
    plt.clf()
    log_file2 = np.loadtxt(os.path.join(args.input_path_2, args.run_id, args.logfile_stage2))
    print(log_file2.shape)
    begin_index = 0
    while (log_file2[begin_index][1] > 1000000): begin_index += 1
    plt.plot(log_file2[begin_index:, 0], log_file2[begin_index:, 1])
    plt.savefig(os.path.join(args.save_path, args.run_id, 'graph_stage2.jpg'), dpi = 1000)