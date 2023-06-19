import gym
import copy
import os
import numpy as np
import random

from collections import OrderedDict
from .space import StateObservationSpace, AutoregressiveEnvObservationSpace, AutoregressiveEnvActionSpace, ActionSpace
from utils.utils import is_edge_addable, readFile, getlen2, save_file, save_trajectory
from truss_envs.dynamic import DynamicModel
from .state import State

class Truss(gym.Env):
    def __init__(self, args, num_points, initial_state_files,
                 coordinate_range, area_range, coordinate_delta_range, area_delta_range, fixed_points, variable_edges,
                 max_refine_steps,
                 min_refine_steps=10,
                 dimension=2, constraint_threshold=1e-7, best_n_results=5,
                 structure_fail_reward=-50., constraint_fail_reward=-10., reward_lambda=169000000,
                 best=10000, save_good_threshold=0, normalize_magnitude=False):
        r'''
        Create a Truss Refine environment instance.
        :param num_points: number of nodes
        :param initial_state_files: locations where initial states are stored
        :param coordinate_range: nodes' coordinate range
        :param area_range: edges' area range
        :param coordinate_delta_range: nodes' modification range
        :param area_delta_range: edges' modification range
        :param max_refine_steps: max refine steps
        :param min_refine_steps: another refine steps threshold
        :param edge_constraint: intersection of edges
        :param dis_constraint: nodes' displacement weight
        :param stress_constraint: edges' stress
        :param buckle_constraint: buckle constraint
        :param self_weight: edge's weight
        :param dimension: dimension
        :param constraint_threshold: eps to check constraint
        :param best_n_results: choose best_n_results from initial_state_files
        :param structure_fail_reward: structure fail reward
        :param constraint_fail_reward: constraint fail reward
        :param reward_lambda: reward lambda
        :param best: initial best weight
        :param save_good_threshold: threshold over best weight to best
        :param normalize_magnitude: normalize observation's magnitude
        '''
        if dimension != 2 and dimension != 3:
            raise NotImplementedError("only support 2D / 3D dimension for now")

        # Env Config
        self.num_points = num_points
        self.dimension = dimension
        self.fixed_points = fixed_points
        self.normalize_magnitude = normalize_magnitude
        self.only_position = args.only_position # if True, do not change edges' area unless greedy upd step.
        self.greedy_upd = args.greedy_upd
        self.global_env_step = 0
        if (args.useAlist): # only 3d case
            self.alist = []
            AREAFILE = args.Alist_path
            with open(AREAFILE,'r') as ar:
                section_lines = ar.readlines()
                for i in range(len(section_lines)):
                    section_line = section_lines[i]
                    section_r = section_line.strip().split(' ')
                    if (i==0):
                        section_num = int(section_r[0])
                        continue
                    if (i > 0 and i <= section_num):
                        d = float(section_r[0]) / 1000.0
                        t = float(section_r[1]) / 1000.0
                        self.alist.append((d, t))
        else: # only 2d case
            area_range = args.area_range
            delta = (area_range[1] - area_range[0]) / 50
            self.alist = [area_range[0] + delta * i for i in range(51)]

        if args.env_mode == 'Area':
            self.state_observation_space = StateObservationSpace(num_points, coordinate_range, area_range)
            self.env_observation_space = AutoregressiveEnvObservationSpace(num_points, coordinate_range, area_range)
            self.action_space = AutoregressiveEnvActionSpace(coordinate_delta_range, area_delta_range)
        
        elif args.env_mode == 'DT':
            self.state_observation_space = StateObservationSpace(num_points, coordinate_range, args.d_range, args.t_range)
            self.env_observation_space = AutoregressiveEnvObservationSpace(num_points, coordinate_range, args.d_range, args.t_range)
            self.action_space = AutoregressiveEnvActionSpace(coordinate_delta_range, args.d_delta_range, args.t_delta_range)
        
        else: raise NotImplementedError

        # Initial State
        self.initial_state_files = initial_state_files
        self.logfile = open(os.path.join(self.initial_state_files, args.logfile_stage2), 'w')
        self.best_n_results = best_n_results

        # Done
        self.refine_step = None
        self.max_refine_steps = max_refine_steps
        self.min_refine_steps = min_refine_steps

        # Dynamics
        self.args = args
        self.env_mode = args.env_mode
        self.bad_attempt_limit = args.bad_attempt_limit
        self.use_edge_constraint = args.CONSTRAINT_CROSS_EDGE
        self.use_dis_constraint = args.CONSTRAINT_DIS
        self.use_stress_constraint = args.CONSTRAINT_STRESS
        self.use_buckle_constraint = args.CONSTRAINT_BUCKLE
        self.use_slenderness_constraint = args.CONSTRAINT_SLENDERNESS
        self.use_max_length_constraint = args.CONSTRAINT_MAX_LENGTH
        self.use_min_length_constraint = args.CONSTRAINT_MIN_LENGTH
        self.use_self_weight = args.CONSTRAINT_SELF_WEIGHT
        self.constraint_threshold = constraint_threshold
        self.dynamic_model = DynamicModel(
            dimension = self.dimension, 
            use_cross_constraint = self.use_edge_constraint,
            use_self_weight = self.use_self_weight, 
            use_dis_constraint = self.use_dis_constraint, 
            use_buckle_constraint = self.use_buckle_constraint, 
            use_stress_constraint = self.use_stress_constraint,
            use_slenderness_constraint = self.use_slenderness_constraint,
            use_longer_constraint = self.use_max_length_constraint,
            use_shorter_constraint = self.use_min_length_constraint,
            E = args.E,
            pho = args.pho,
            sigma_T = args.sigma_T,
            sigma_C = args.sigma_C,
            dislimit = args.dislimit,
            slenderness_ratio_T = args.slenderness_ratio_t,
            slenderness_ratio_C = args.slenderness_ratio_c,
            max_len = args.len_range[1],
            min_len = args.len_range[0],
        )

        # State
        self.initial_state_file = None
        self.initial_state_point = None
        self.initial_state_bar = None
        self.initial_state_mass = None
        self.state = State(num_points, dimension, args.env_mode)
        self.state_dynamics = None
        self.prev_mass = None
        self.prev_reward = None
        self.action_id = None
        self.trajectory = None
        self.loads = None
        self.normalize_factor = None
        self.last_id = 0

        # Reward
        self.structure_fail_reward = structure_fail_reward
        self.constraint_fail_reward = constraint_fail_reward
        self.reward_lambda = reward_lambda

        # Result
        self.best = best
        self.save_good_threshold = save_good_threshold

    @property
    def observation_space(self):
        return self.env_observation_space

    def _reward_fn(self):
        is_struct, mass, dis_value, stress_value, buckle_value = self.state_dynamics
        extra_reward = 0
        if not is_struct:
            extra_reward += self.structure_fail_reward
        if max(dis_value, stress_value, buckle_value) > self.constraint_threshold:
            extra_reward += self.constraint_fail_reward
        reward = self.reward_lambda / ((mass * (1 + dis_value + stress_value + buckle_value)) ** 2)
        return reward, extra_reward

    def _stop_fn(self, reward_step):
        if (self.refine_step >= self.max_refine_steps): return True
        if (reward_step <= self.constraint_fail_reward):
            self.bad_attempt += 1
        return self.bad_attempt >= self.bad_attempt_limit #and self.refine_step >= self.min_refine_steps

    def valid_truss(self):
        r'''
        check whether self.state is valid
        :return: a list of four bools, a tuple of dynamics
        '''
        ret = [True for _ in range(4)]
        if (self.env_mode == 'Area'):
            if not self.state_observation_space.contains(self.state.obs(nonexistent_edge=self.state_observation_space.low[-1])):
                ret[0] = False  # Not in valid observation
        
        if (self.env_mode == 'DT'):
            if not self.state_observation_space.contains(self.state.obs(nonexistent_edge=self.state_observation_space.low[-2:])):
                ret[0] = False  # Not in valid observation

        for i in range(self.num_points):
            for j in range(i):
                if (self.state.nodes[i] == self.state.nodes[j]).all():
                    ret[1] = False  # Duplicate nodes location

        points = copy.deepcopy(self.initial_state_point)
        for i in range(self.num_points):
            points[i].vec.x = self.state.nodes[i][0]
            points[i].vec.y = self.state.nodes[i][1]
            if self.dimension == 3:
                points[i].vec.z = self.state.nodes[i][2]
        edges = copy.deepcopy(self.initial_state_bar)
        edges_list = []
        for i in range(self.num_points):
            for j in range(i):
                if (self.env_mode == 'Area'):
                    if self.state.edges[i][j] > 0:
                        edges[(j, i)]._area = self.state.edges[i][j]
                        edges[(j, i)].len = getlen2(points[j], points[i])
                        edges_list.append((j, i))
                if (self.env_mode == 'DT'):
                    if self.state.edges[i][j][0] > 0:
                        d = self.state.edges[i][j][0]
                        t = self.state.edges[i][j][1]
                        assert (d - 2 * t >= 0)
                        if (d - 2 * t == 0 or t == 0): continue
                        edges[(j, i)].d = d
                        edges[(j, i)].t = t
                        edges[(j, i)].len = getlen2(points[j], points[i])
                        edges_list.append((j, i))

        if self.use_edge_constraint and self.dimension == 2:
            for _ in edges_list:
                i, j = _
                left_edges = copy.deepcopy(edges)
                left_edges.pop((i, j))
                if not is_edge_addable(i, j, points, left_edges):
                    ret[2] = False  # Edges intersect

        is_struct, mass, dis_value, stress_value, buckle_value, slenderness_value, longer_value, shorter_value, cross_value = self.dynamic_model.run(points, edges, mode = 'train')
        ret[3] = is_struct and max(dis_value, stress_value, buckle_value) < self.constraint_threshold and max(slenderness_value, longer_value, shorter_value, cross_value) <= 0 # Dynamic constraints
        return ret, (is_struct, mass, dis_value, stress_value, buckle_value)

    def reset(self, file_name=None):
        _ = random.random()
        if _ > 0.5:
            best_n_results = self.best_n_results
        else:
            best_n_results = -1
        if file_name is None:
            _input_file_list = os.listdir(self.initial_state_files)
            input_file_list = []
            self.total_diverse_count = dict()
            for s in _input_file_list:
                if s[-3:] == 'txt' and s[0] != '_':
                    input_file_list.append(s)
                    if (self.total_diverse_count.get(int(s[-6: -4])) == None):
                        self.total_diverse_count[int(s[-6: -4])] = 0
                    self.total_diverse_count[int(s[-6: -4])] += 1
            input_file_list.sort(key = lambda x: int(x[:-7]))
            if best_n_results != -1: # maybe a bug, fixed #
                if len(input_file_list) > self.best_n_results:
                    input_file_list = input_file_list[:self.best_n_results]
            choise_id = np.random.randint(len(input_file_list))
            file_name = os.path.join(self.initial_state_files, input_file_list[choise_id])
            self.diverse_id = int(input_file_list[choise_id][-6 : -4])
        #    print(file_name, self.diverse_id)

        #print("file_name =", file_name)

        self.initial_state_file = file_name
        points, edges = readFile(file_name)
        self.initial_state_point = OrderedDict()
        self.initial_state_bar = OrderedDict()
        for i in range(self.num_points):
            self.initial_state_point[i] = points[i]
        for e in edges:
            if e.area < 0:
                continue
            u = e.u
            v = e.v
            if u > v:
                tmp = u
                u = v
                v = tmp
            self.initial_state_bar[(u, v)] = e

        self.state = State(self.num_points, self.dimension, self.env_mode)
        for i in range(self.num_points):
            self.state.nodes[i][0] = points[i].vec.x
            self.state.nodes[i][1] = points[i].vec.y
            if self.dimension == 3:
                self.state.nodes[i][2] = points[i].vec.z

        if (self.env_mode == 'Area'):
            for e in edges:
                i = e.u
                j = e.v
                self.state.edges[i][j] = e.area
                self.state.edges[j][i] = e.area
        
        if (self.env_mode == 'DT'):
            for e in edges:
                i = e.u
                j = e.v
                self.state.edges[i][j][0] = e.d
                self.state.edges[j][i][0] = e.d
                self.state.edges[i][j][1] = e.t
                self.state.edges[j][i][1] = e.t
        
        _ = self.valid_truss()
        assert _[0][0] and _[0][1] and _[0][2] and _[0][3], "Initial state {} not valid".format(file_name)
        self.state_dynamics = _[1]
        self.prev_mass = _[1][1]
        self.initial_state_mass = _[1][1]
        self.prev_reward, __ = self._reward_fn()
        self.refine_step = 0
        self.total_reward = 0
        self.bad_attempt = 0

        self.trajectory = [copy.deepcopy(self.state)]

        self.loads = []
        for i in range(self.num_points):
            self.loads.append(self.initial_state_point[i].loadY)

        self.normalize_factor = np.array([1. for _ in range(self.num_points * self.dimension)] +
                                         [100. for _ in range(self.num_points * (self.num_points - 1) // 2)] +
                                         [1. for _ in range(2)] +
                                         [1e-5 for _ in range(self.num_points)])

        return self._observation()

    def _observation(self):
        state_obs = self.state.obs()
        self.action_id, self.action_id_one_hot = self._generate_action_id()
        ret = np.concatenate((state_obs, np.array(self.loads), self.action_id))
        if self.normalize_magnitude:
            print("ret:", ret)
            print("normalize_factor:", self.normalize_factor)
            ret = ret * self.normalize_factor
            print("new_ret:", ret)
        return ret

    def _generate_action_id(self):
        point_action = np.zeros(self.num_points, dtype = np.float64)
        edge_action = np.zeros(self.num_points * (self.num_points - 1) // 2, dtype = np.float64)
        greedy_action = np.zeros(1, dtype = np.float64)
        choose = np.random.randint(0, 20)
        if (choose == 0): # now use greedy update
            greedy_action[0] = 1
            return np.array([-1, -1, 1]), np.concatenate([point_action, edge_action, greedy_action])
        if (not self.args.eval):
            id = np.random.randint(self.num_points - self.fixed_points + len(self.initial_state_bar))
        else: 
            id = (self.last_id + 1) % self.num_points - self.fixed_points + len(self.initial_state_bar)
            self.last_id += 1
        if id < self.num_points - self.fixed_points or self.only_position == True:
            i = id % (self.num_points - self.fixed_points) + self.fixed_points
            j = -1
            point_action[i] = 1
        else:
            i = -1
            u, v = list(self.initial_state_bar)[id - (self.num_points - self.fixed_points)]
            j = (u * ((self.num_points - 1) + (self.num_points - u)) // 2) + (v - u - 1)
            edge_action[j] = 1
        return np.array([i, j, -1]), np.concatenate([point_action, edge_action, greedy_action])
    
    def greedy_update(self, obs):
        n_obs = copy.deepcopy(obs)
        for i in range(len(self.initial_state_bar)):
                u, v = list(self.initial_state_bar)[i]
                j = (u * ((self.num_points - 1) + (self.num_points - u)) // 2) + (v - u - 1)
                if (self.env_mode == 'DT'):
                    pos = j * 2 + self.num_points * self.dimension
                else: pos = j + self.num_points * self.dimension
                if (self.env_mode == 'DT'): ori = n_obs[pos: pos + 2].copy()
                if (self.env_mode == 'Area'): ori = n_obs[pos: pos + 1].copy()
                minaera = 1e9
                if (self.alist != None):
                    for j in range(len(self.alist)):
                        if (self.env_mode == 'DT'): n_obs[pos: pos + 2] = self.alist[j]
                        if (self.env_mode == 'Aera'): n_obs[pos: pos + 1] = self.alist[j]
                        self.state.set(n_obs)
                        valid, temp_state_dynamics = self.valid_truss()
                        if (valid[0] and valid[1] and valid[2] and valid[3] and temp_state_dynamics[1] < minaera):
                            minaera = temp_state_dynamics[1]
                            if (self.env_mode == 'DT'): 
                                ori[0] = n_obs[pos: pos + 2][0]
                                ori[1] = n_obs[pos: pos + 2][1]
                            else: ori = n_obs[pos: pos + 1].copy()
                    if (self.env_mode == 'DT'): n_obs[pos: pos + 2] = ori.copy()
                    if (self.env_mode == 'Area'): n_obs[pos: pos + 1] = ori.copy()
                else:
                    raise NotImplementedError()
        return n_obs

    def fit_diverse(self, area):
        assert self.env_mode == 'DT'
        min_area = 1e9
        chosed_area = self.alist[-1]
        for d_area in self.alist:
            if (area[0] <= d_area[0] and area[1] <= d_area[1]):
                if (min_area > d_area[0] ** 2 - (d_area[0] - 2 * d_area[1]) ** 2):
                    chosed_area = d_area
                    min_area = d_area[0] ** 2 - (d_area[0] - 2 * d_area[1]) ** 2
        return chosed_area

    def step(self, action):
        self.global_env_step += 1
        if (self.global_env_step % 4000 == 0):
            print(self.global_env_step, self.best, file = self.logfile)
            print(self.global_env_step, self.best)
#            best_path = os.path.join(self.initial_state_files, '_best.txt')
#            save_log_path = os.path.join(self.initial_state_files, '_best_{}.txt'.format(str(self.global_env_step)))
#            os.system('cp {} {}'.format(best_path, save_log_path))
        assert self.action_space.contains(action), "actions({}) not in action space({})".format(action, self.action_space)
        obs = self.state.obs(nonexistent_edge=-1)
        n_obs = copy.deepcopy(obs)
        if (self.action_id[2] == 1): # Greedy update
            n_obs = self.greedy_update(n_obs)

        if (self.env_mode == 'Area'):
            if self.action_id[1] != -1:
                _i = int(self.action_id[1]) + self.num_points * self.dimension
                n_obs[_i] += action[-1]
                n_obs[_i] = max(min(n_obs[_i], self.state_observation_space.high[_i]), self.state_observation_space.low[_i])
            if self.action_id[0] != -1:
                n_obs[int(self.action_id[0]) * self.dimension: int(self.action_id[0] + 1) * self.dimension] += action[:-1] # act = [(a, b, c), d]
                for _i in range(int(self.action_id[0]) * self.dimension, int(self.action_id[0] + 1) * self.dimension):
                    n_obs[_i] = max(min(n_obs[_i], self.state_observation_space.high[_i]), self.state_observation_space.low[_i])

        if (self.env_mode == 'DT'):
            if self.action_id[1] != -1:
                _i = int(self.action_id[1]) * 2 + self.num_points * self.dimension
                n_obs[_i: _i + 2] += action[-2:]
                n_obs[_i] = max(min(n_obs[_i], self.state_observation_space.high[_i]), self.state_observation_space.low[_i])
                n_obs[_i + 1] = max(min(n_obs[_i + 1], self.state_observation_space.high[_i + 1]), self.state_observation_space.low[_i + 1])
                n_obs[_i: _i + 2] = self.fit_diverse(n_obs[_i: _i + 2])
                n_obs[_i + 1] = min(n_obs[_i + 1], n_obs[_i] / 2)
            if self.action_id[0] != -1:
                n_obs[int(self.action_id[0]) * self.dimension: int(self.action_id[0] + 1) * self.dimension] += action[:-2] # act = [(a, b, c), (d, e)]
                for _i in range(int(self.action_id[0]) * self.dimension, int(self.action_id[0] + 1) * self.dimension):
                    n_obs[_i] = max(min(n_obs[_i], self.state_observation_space.high[_i]), self.state_observation_space.low[_i])                

        self.state.set(n_obs)
        self.trajectory.append(copy.deepcopy(self.state))

        info = {}
        info['illegal action'] = 0
        valid, self.state_dynamics = self.valid_truss()

        reward, extra_reward = self._reward_fn()
        if not (valid[0] and valid[1] and valid[2]):
            info['illegal action'] = 1
            extra_reward += self.structure_fail_reward # check whether go out the boundary

        if (extra_reward != 0):
            reward_step = extra_reward # not satisfy the constraint
        else: 
            reward_step = reward - self.prev_reward
            self.prev_reward = reward # delta reward

        mass = self.state_dynamics[1]
        self.prev_mass = mass

        if not (valid[0] and valid[1] and valid[2] and valid[3]): mass = 10000
        done = self._stop_fn(reward_step)
        
        info['is struct'] = self.state_dynamics[0]
        info['mass'] = mass # if illegal, set to 10000
        info['displacement'] = self.state_dynamics[2]
        info['stress'] = self.state_dynamics[3]
        info['buckle'] = self.state_dynamics[4]
        info['initial_state_file'] = self.initial_state_file
        self.refine_step += 1
        self.total_reward += reward_step

        if (valid[0] and valid[1] and valid[2] and valid[3]):
            if mass < self.best + self.save_good_threshold:
                self.total_diverse_count[self.diverse_id] += 1
                if mass < self.best:
                    # if mass < self.best - 1:
                    #     save_trajectory(self.initial_state_point, self.trajectory, mass, self.initial_state_files)
                    print("best:", mass)
                    self.best = mass
                    save_file(self.initial_state_point, self.state, mass, self.initial_state_files, self.env_mode, best = True)

                max_count = self.best_n_results
                if (self.args.eval): max_count = 1
                if (self.total_diverse_count[self.diverse_id] > max_count):
                    self.total_diverse_count[self.diverse_id] -= 1
                    os.remove(self.initial_state_file)
                    save_file(self.initial_state_point, self.state, mass, self.initial_state_files, self.env_mode, diverse_id = self.diverse_id)
                    self.initial_state_file = os.path.join(self.initial_state_files, str(int(mass * 1000)) + "_" + str(self.diverse_id).zfill(2) + ".txt")
                    self.initial_state_mass = mass
                    points, edges = readFile(self.initial_state_file)
                    self.initial_state_point = OrderedDict()
                    self.initial_state_bar = OrderedDict()
                    for i in range(self.num_points): self.initial_state_point[i] = points[i]
                    for e in edges:
                        if e.area < 0: continue
                        u = e.u
                        v = e.v
                        if u > v:
                            tmp = u
                            u = v
                            v = tmp
                        self.initial_state_bar[(u, v)] = e
                else:
                    save_file(self.initial_state_point, self.state, mass, self.initial_state_files, self.env_mode, diverse_id = self.diverse_id)

            elif mass + 0.01 < self.initial_state_mass: 
                os.remove(self.initial_state_file)
                save_file(self.initial_state_point, self.state, mass, self.initial_state_files, self.env_mode, diverse_id = self.diverse_id)
                self.initial_state_file = os.path.join(self.initial_state_files, str(int(mass * 1000)) + "_" + str(self.diverse_id).zfill(2) + ".txt")
                self.initial_state_mass = mass
                points, edges = readFile(self.initial_state_file)
                self.initial_state_point = OrderedDict()
                self.initial_state_bar = OrderedDict()
                for i in range(self.num_points):
                    self.initial_state_point[i] = points[i]
                for e in edges:
                    if e.area < 0: continue
                    u = e.u
                    v = e.v
                    if u > v:
                        tmp = u
                        u = v
                        v = tmp
                    self.initial_state_bar[(u, v)] = e
        return self._observation(), reward_step, done, info
