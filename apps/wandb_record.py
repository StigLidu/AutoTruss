import matplotlib.pyplot as plt
import wandb
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
    run_dir = os.path.join(args.save_path, args.run_id)
    wandb.init(
        config = args,
        project = 'Truss_plot_fixed',
        group = args.config,
        dir = run_dir,
        job_type = 'check',
        name = args.config + args.run_id
    )
    Envs_init(args)
    log_file1 = np.loadtxt(os.path.join(args.save_path, args.run_id, args.logfile_stage1))
    print(log_file1.shape)
    log_file2 = np.loadtxt(os.path.join(args.input_path_2, args.run_id, args.logfile_stage2))
    begin_index = 0
    while (log_file1[begin_index][2] > 1000000): begin_index += 1
    for i in range(0, begin_index):
        wandb.log({'mass_stage1': log_file1[begin_index][2], 
            'mass_stage2': log_file2[i][1]}, 
        step = int(log_file2[i][0]))
    for i in range(begin_index, len(log_file1)):
        if (i < len(log_file2)):
            wandb.log({'mass_stage1': log_file1[i][2], 
                'mass_stage2': log_file2[i][1]}, step = int(log_file1[i][0]))
        else:
            wandb.log({'mass_stage1': log_file1[i][2]}, step = int(log_file1[i][0]))