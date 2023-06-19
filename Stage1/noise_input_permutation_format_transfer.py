import time
import json
import numpy as np
import math
import random
import copy

import matplotlib.pyplot as plt
import warnings
import os, sys, contextlib
import openseespy.opensees as op

import heapq

base_path = os.getcwd()
sys.path.append(base_path)
from configs.config import get_base_config, make_config

parser = get_base_config()
args = parser.parse_known_args(sys.argv[1:])[0]
config = make_config(args.config)
for k, v in config.get("base", {}).items():
    if f"--{k}" not in args:
        setattr(args, k, v)

if not os.path.exists('PostResults'):
    os.mkdir('PostResults')
if not os.path.exists('PostResults/' + args.config):
    os.mkdir('PostResults/' + args.config)

OrgF = os.path.join(args.save_path, args.run_id)
TarFolder = os.path.join(args.input_path_2, args.run_id)
if not os.path.exists(TarFolder):
    os.mkdir(TarFolder)
OrgFolder = os.path.join(OrgF, args.transfer_filefold)
print(OrgFolder)

permutation = [0, 1, 2, 3]

    #if not os.path.exists(TarFolder):
    #    os.mkdir(TarFolder)

files = os.listdir(OrgFolder)
files.sort(key = lambda x: int(x[:-4]))
selected_files = []
max_idx = len(files)
#print('max_idx', max_idx)

if (args.finetune == False):
    for _ in range(min(args.max_num_topo_truss, max_idx)):
        _idx = _
        selected_files.append(files[_idx])
else: 
    assert(max_idx > args.max_num_topo_truss)
    right = min(args.max_num_topo_truss * 2, max_idx)
    for _ in range(args.max_num_topo_truss, right):
        _idx = _
        selected_files.append(files[_idx])

print(selected_files)
for idfile, file in enumerate(selected_files):
        FILENAME = os.path.join(OrgFolder, file)
        #SAVENAME = TarFolder + file
        #SAVENAME = SAVENAME[:-4]
        #SAVENAME += folder_name
        #SAVENAME += '.txt'
        if file[-4:] != '.txt': continue
        with open(FILENAME, "r") as fle:
            lines = fle.readlines()
            for i in range(len(lines)):
                line = lines[i]
                vec = line.strip().split(' ')

                if (i == 0):
                    vn = int(vec[0])
                    en = int(vec[1])
                    Edges = [[- 1.0 for _ in range(vn)] for _ in range(vn)]
                    d = [[- 1.0 for _ in range(vn)] for _ in range(vn)]
                    t = [[- 1.0 for _ in range(vn)] for _ in range(vn)]
                    Nodes = []
                    nodes_position = []
                    continue

                if (1 <= i and i <= vn):
                    Nodes.append(line)
                    nodes_position.append([vec[0], vec[1], vec[2]])
                    continue

                if (vn + 1 <= i and i <= vn + en):
                    node1 = int(vec[0])
                    node2 = int(vec[1])
                    Edges[node1][node2] = vec[2]
                    Edges[node2][node1] = vec[2]
                    d[node1][node2] = vec[3]
                    d[node2][node1] = vec[3]
                    t[node1][node2] = vec[4]
                    t[node2][node1] = vec[4]

        mass = 0
        pho = args.pho
        for v_i in range(vn):
            for v_j in range(vn):
                if v_i < v_j:
                    i_x = float(nodes_position[v_i][0])
                    i_y = float(nodes_position[v_i][1])
                    i_z = float(nodes_position[v_i][2])
                    j_x = float(nodes_position[v_j][0])
                    j_y = float(nodes_position[v_j][1])
                    j_z = float(nodes_position[v_j][2])
                    area = float(Edges[v_i][v_j])
                    if area == -1:
                        continue
                    mass += math.sqrt((i_x - j_x) ** 2 + (i_y - j_y) ** 2 + (i_z - j_z) ** 2) * area * args.pho


        SAVENAME = os.path.join(TarFolder, str(round(mass * 1000)) + '_' + str(idfile).zfill(2) + '.txt')
        #print(mass, SAVENAME)

        PermutationEdges = [[- 1.0 for _ in range(vn)] for _ in range(vn)]
        Permutationd = [[- 1.0 for _ in range(vn)] for _ in range(vn)]
        Permutationt = [[- 1.0 for _ in range(vn)] for _ in range(vn)]
        for i in range(vn):
            for j in range(vn):
                new_i = i
                new_j = j
                if i < len(permutation):
                    new_i = permutation[i]
                if j < len(permutation):
                    new_j = permutation[j]
                PermutationEdges[i][j] = Edges[new_i][new_j]
                PermutationEdges[j][i] = Edges[new_j][new_i]
                Permutationd[i][j] = d[new_i][new_j]
                Permutationd[j][i] = d[new_j][new_i]
                Permutationt[i][j] = t[new_i][new_j]
                Permutationt[j][i] = t[new_j][new_i]

        with open(SAVENAME, "w") as f:
            print(int(vn), int(vn * (vn - 1) / 2), file=f)
            for i in range(len(Nodes)):
                new_i = i
                if i < len(permutation):
                    new_i = permutation[i]
                print(Nodes[new_i], file=f, end='')
            for j in range(vn):
                for i in range(vn):
                    if i < j:
                        print(int(i), int(j), PermutationEdges[i][j], Permutationd[i][j], Permutationt[i][j], file=f)