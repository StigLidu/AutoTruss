import numpy as np
import logging
import math, time
import os, sys, contextlib, platform

base_path = os.getcwd()
sys.path.append(base_path)

from configs.config import get_base_config, make_config
from utils.utils import readFile, readAlist, save_file_from_list, util_init
from apps.draw import *
from algo.UCTs import *
from truss_envs.reward import *

parser = get_base_config()
args = parser.parse_known_args(sys.argv[1:])[0]
config = make_config(args.config)
for k, v in config.get("base", {}).items():
    if f"--{k}" not in args:
        setattr(args, k, v)

print(config)

def main():
    p, e = readFile(args.input_path)
    if not os.path.exists('results_3d/' + args.config):
        os.mkdir('results_3d/' + args.config)
        
    # save and load path
    LOGFOLDER = args.save_path
    if not os.path.exists(LOGFOLDER): os.mkdir(LOGFOLDER)

    if (args.useAlist == True): Alist = readAlist(args.Alist_path)
    else: Alist = None

    Envs_init(args)
    UCTs_init(args, arealist__ = Alist)
    util_init(args)
    
    bestreward, pbest, ebest = UCTSearch(p, e)

    print("bestreward =",bestreward)
    print(reward_fun(pbest, ebest))

if __name__ == '__main__':
    if not os.path.exists('results_3d/'):
        os.mkdir('results_3d/')
    main()