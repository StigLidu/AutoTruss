import os, sys
base_path = os.getcwd()
sys.path.append(base_path)

from configs.config import *
parser = get_base_config()
args = parser.parse_known_args(sys.argv[1:])[0]
from collections import OrderedDict
from truss_envs.reward import *
from utils.utils import *
from Stage2.envs.dynamic2 import DynamicModel

Envs_init(args)

FILENAME = args.check_file
    
p = OrderedDict()
e = OrderedDict()
with open(FILENAME, "r") as fle:
    lines = fle.readlines()
    edge_cnt = 0
    for i in range(len(lines)):
        line = lines[i]
        vec = line.strip().split(' ')
        if (i == 0):
            vn = int(vec[0])
            en = int(vec[1])
            continue

        if (1 <= i and i <= vn):
            p[i - 1] = Point(Vector3(float(vec[0]), float(vec[1]), float(vec[2])), int(vec[3]), int(vec[4]), int(vec[5]), float(vec[6]), float(vec[7]), float(vec[8]))
        
        if (vn + 1 <= i and i <= vn + en and float(vec[2]) != -1):
            e[edge_cnt] = Bar(vec[0], vec[1], float(vec[2]), getlen2(p[int(vec[0])], p[int(vec[1])]), d = float(vec[3]), t = float(vec[4]))
            edge_cnt += 1

#print(len(p), len(e))
model = DynamicModel(dimension = 3)
print(model.run(p, e, mode = 'check'))
print(reward_fun(p, e,sanity_check = False))