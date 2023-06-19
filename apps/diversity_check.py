import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import sys, os
import time
base_path = os.getcwd()
sys.path.append(base_path)
from configs.config import get_base_config, make_config
from utils.utils import *
from truss_envs.reward import reward_fun
#change 3D作图
time_str = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime()) 

if __name__ == '__main__':
    parser = get_base_config()
    args = parser.parse_known_args(sys.argv[1:])[0]
    from truss_envs.reward import Envs_init, reward_fun
    config = make_config(args.config)
    for k, v in config.get("base", {}).items():
        if f"--{k}" not in args:
            setattr(args, k, v)
    Envs_init(args)

def check_diversity_map(truss_list):
    distinct = []
    distinct_count = []
    for truss in truss_list:
        unique = True
        for i in range(len(distinct)):
            ref_truss = distinct[i]
            if (similar_topo(truss[0], truss[1], ref_truss[0], ref_truss[1])):
                unique = False
                distinct_count[i] += 1
                break
        if (unique):
            distinct.append(truss)
            distinct_count.append(1)
    for i in range(len(distinct)):
        distinct[i] = (distinct[i], distinct_count[i])
    distinct.sort(key = lambda x1: x1[1], reverse = True)
    distinct_count.sort(reverse = True)
    return len(distinct), distinct_count, distinct

if __name__ == '__main__':
    check_path = os.path.join(args.save_path, args.run_id, 'DIVERSITY_TOPO_result')
    files = os.listdir(check_path)
    truss_list = []
    for file in files:
        if (file[-4:] == '.txt'):
            p, e = readFile(os.path.join(check_path, file))
            truss_list.append((p, e))
    print(check_diversity_map(truss_list)[0 : 2])
    check_path = os.path.join(args.input_path_2, args.run_id)
    files = os.listdir(check_path)
    truss_list = []
    for file in files:
        if (file[-4:] == '.txt'):
            p, e = readFile(os.path.join(check_path, file))
            truss_list.append((p, e))
    print(check_diversity_map(truss_list)[0 : 2])
