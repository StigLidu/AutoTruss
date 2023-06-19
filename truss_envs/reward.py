import sys, os
import openseespy.opensees as op
import numpy as np
import math
from truss_envs.dynamic import DynamicModel
from utils.utils import Bar, getlen2

def Envs_init(args__):
    global args
    global truss_env
    args = args__
    '''
    global E
    global pho
    global Sigma_T
    global Sigma_C
    global slenderness_ratio_c
    global slenderness_ratio_t
    global dislimit
    global maxlen
    global minlen
    global CONSTRAINT_STRESS
    global CONSTRAINT_DIS
    global CONSTRAINT_BUCKLE
    global CONSTRAINT_SLENDERNESS
    
    #Elasticity modulus
    E = args.E #1.93*10**11
    pho = args.pho #8.0*10**3
    Sigma_T = args.sigma_T #123.0*10**6
    Sigma_C = args.sigma_C #123.0*10**6
    slenderness_ratio_c = args.slenderness_ratio_c #180.0
    slenderness_ratio_t = args.slenderness_ratio_t #220.0
    dislimit = args.dislimit #0.002
    maxlen = args.len_range[1]
    minlen = args.len_range[0]

    #constraints
    CONSTRAINT_STRESS = args.CONSTRAINT_STRESS
    CONSTRAINT_DIS = args.CONSTRAINT_DIS
    CONSTRAINT_BUCKLE = args.CONSTRAINT_BUCKLEd
    CONSTRAINT_SLENDERNESS = args.CONSTRAINT_SLENDERNESS
    '''

    truss_env = DynamicModel(
            dimension = args.env_dims,
            E = args.E,
            pho = args.pho,
            sigma_T = args.sigma_T,
            sigma_C = args.sigma_C,
            dislimit = args.dislimit,
            slenderness_ratio_C = args.slenderness_ratio_c,
            slenderness_ratio_T = args.slenderness_ratio_t,
            max_len = args.len_range[1],
            min_len = args.len_range[0],
            use_self_weight = args.CONSTRAINT_SELF_WEIGHT,
            use_dis_constraint = args.CONSTRAINT_DIS,
            use_stress_constraint = args.CONSTRAINT_STRESS,
            use_buckle_constraint = args.CONSTRAINT_BUCKLE,
            use_slenderness_constraint = args.CONSTRAINT_SLENDERNESS,
            use_longer_constraint = args.CONSTRAINT_MAX_LENGTH,
            use_shorter_constraint = args.CONSTRAINT_MIN_LENGTH,
            use_cross_constraint = args.CONSTRAINT_CROSS_EDGE,
        )

Reward_cnt = 0

def reward_fun(p, e, sanity_check = True, mode = 'train'):
    global Reward_cnt
    Reward_cnt += 1
    if (sanity_check):
        for se in e:
            new_e = Bar(se.u, se.v, se._area, leng = getlen2(p[se.u], p[se.v]), d = se.d, t = se.t)
            assert(new_e.area == se.area)
            assert(new_e.len == se.len)
            assert(new_e.v == se.v)
            assert(new_e.t == se.t)

    is_struct, mass, dis_value, stress_value, buckle_value, slenderness_value, longer_value, shorter_value, cross_value = truss_env.run(p, e, mode = mode) 

    dis_value = np.sum(dis_value)
    stress_value = np.sum(stress_value)
    buckle_value = np.sum(buckle_value)
    slenderness_value = np.sum(slenderness_value)
    longer_value = np.sum(longer_value)
    shorter_value = np.sum(shorter_value)
    cross_value = np.sum(cross_value)

    if (mode == 'check'):
        print(is_struct, mass, dis_value, stress_value, buckle_value, slenderness_value, longer_value, shorter_value, cross_value)

    #check_geometruc_stability
    if ((not is_struct) or cross_value > 0):
        return -1.0, Reward_cnt, mass, 0, 0, 0
    
    # check slenderness ratio / longer_value / shorter_value
    if (slenderness_value > 0 or longer_value > 0 or shorter_value > 0):
        return 0, Reward_cnt, mass, 0, 0, 0

    # check displacement / Stress_value / Buckle_value
    if (dis_value > 1e-7 or stress_value > 1e-7 or buckle_value > 1e-7):
        return 0, Reward_cnt, mass, dis_value, stress_value, buckle_value
    
    # pass check
    reward = mass * (1 + dis_value + stress_value + buckle_value)
    return reward, Reward_cnt, mass, dis_value, stress_value, buckle_value