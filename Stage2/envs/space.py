import numpy as np
import gym


class ActionSpace(gym.spaces.Box):
    def __init__(
        self,
        num_points,
        coordinate_delta_range,
        area_delta_range,
        fixed_points,
        variable_edges,
    ):
        r'''
        :param num_points: Total number of points
        :param coordinate_delta_range: Actions' coordinate delta range, a list of length D, where D is the dimension of the environment, each element is a tuple of (low, high)
        :param area_delta_range: Actions' area delta range (low, high)
        :param fixed_points: Number of points whose coordinate cannot be modified (first #fixed_points nodes)
        :param variable_edges: Number of edges whose area can be modified, -1 if all can be modified (first #variable_edges edges)
        '''
        single_low = [cr[0] for cr in coordinate_delta_range]
        single_high = [cr[1] for cr in coordinate_delta_range]
        low = []
        high = []
        edge_num = num_points * (num_points - 1) // 2
        if variable_edges != -1:
            edge_num = variable_edges
        for _ in range(num_points - fixed_points):
            low += single_low
            high += single_high
        for _ in range(edge_num):
            low.append(area_delta_range[0])
            high.append(area_delta_range[1])
        super().__init__(low=np.array(low), high=np.array(high), dtype=np.float64)


class AutoregressiveEnvActionSpace(gym.spaces.Box): #TODO: change area into d, t 
    def __init__(
        self,
        coordinate_delta_range,
        range1,
        range2 = None,
    ):
        r'''
        :param coordinate_delta_range: Actions' coordinate delta range, a list of length D, where D is the dimension of the environment, each element is a tuple of (low, high)
        :param area_delta_range: Actions' area delta range (low, high)
        '''
        low = [cr[0] for cr in coordinate_delta_range]
        high = [cr[1] for cr in coordinate_delta_range]
        low.append(range1[0])
        high.append(range1[1])
        if (range2 != None):
            low.append(range2[0])
            high.append(range2[1])
        super().__init__(low=np.array(low), high=np.array(high), dtype=np.float64)


class StateObservationSpace(gym.spaces.Box):
    def __init__(
        self,
        num_points,
        coordinate_range,
        range1,
        range2 = None,
    ):
        r'''
        :param num_points: Total number of points
        :param coordinate_range: nodes' coordinate range, a list of length D, where D is the dimension of the environment, each element is a tuple of (low, high)
        :param area_range: edges' area range (low, high)
        '''
        single_low = [cr[0] for cr in coordinate_range]
        single_high = [cr[1] for cr in coordinate_range]
        low = []
        high = []
        edge_num = num_points * (num_points - 1) // 2
        for _ in range(num_points):
            low += single_low
            high += single_high
        for _ in range(edge_num):
            low.append(range1[0])
            high.append(range1[1])
            if (range2 != None):
                low.append(range2[0])
                high.append(range2[1])
        super().__init__(low=np.array(low), high=np.array(high), dtype=np.float64)


class AutoregressiveEnvObservationSpace(gym.spaces.Box):
    def __init__(
        self,
        num_points,
        coordinate_range,
        range1,
        range2 = None,
    ):
        r'''
        :param num_points: Total number of points
        :param coordinate_range: nodes' coordinate range, a list of length D, where D is the dimension of the environment, each element is a tuple of (low, high)
        :param area_range: edges' area range (low, high)
        '''
        single_low = [cr[0] for cr in coordinate_range]
        single_high = [cr[1] for cr in coordinate_range]
        low = []
        high = []
        edge_num = num_points * (num_points - 1) // 2
        for _ in range(num_points):
            low += single_low
            high += single_high
        for _ in range(edge_num):
            low.append(range1[0])
            high.append(range1[1])
            if (range2 != None):
                low.append(range2[0])
                high.append(range2[1])
        for _ in range(num_points):
            low.append(-1000000)
            high.append(1000000)
        low.append(-1)
        high.append(num_points)
        low.append(-1)
        high.append(num_points * (num_points - 1) // 2)
        low.append(-1)
        high.append(1)
        super().__init__(low=np.array(low), high=np.array(high), dtype=np.float64)