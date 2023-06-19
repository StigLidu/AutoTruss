from utils.utils import Vector3, Point, Bar
from utils.utils import getrand, randpoint, Kernel1, Kernel2, getlen, getlen2, save_file_stage1, readFile, transintersect, similar_position, similar_topo, closestDistanceBetweenLines
from truss_envs.reward import *
from algo.value_network import Value_Network
import math
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
import time
import heapq

pbest = []
ebest = []
bestreward = 1e9
tmpbestreward = 1e9
FILENAME = "kr-sundial-newinput.txt"
time_str = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime()) 

def UCTs_init(args__, plist__ = [], arealist__ = []):
    global args
    global arealist
    global plist
    global minx
    global maxx
    global miny
    global maxy
    global minz
    global maxz
    global minlen
    global maxlen
    global minarea
    global maxarea
    global OUTPUT_ALL_THRESHOLD
    global MASS_ALLFOLDER
    global MASS_OUTPUT_ALL_MAX
    global OUTPUT_ALL_MAX
    global MASS_OUTPUT_ALL_THRESHOLD
    global ALLFOLDER
    global LOGFOLDER
    global logfile
    global DIVERSITY_FOLDER
    global DIVERSITY_TOPO_FOLDER
    global save_valid_count
    global save_invalid_count
    global v_network
    global global_iteration
    args = args__
    minx = args.coordinate_range[0][0]
    maxx = args.coordinate_range[0][1]
    miny = args.coordinate_range[1][0]
    maxy = args.coordinate_range[1][1]
    if (args.env_dims == 3):
        minz = args.coordinate_range[2][0]
        maxz = args.coordinate_range[2][1]
    else:
        minz = maxz = 0
    minlen = args.len_range[0]
    maxlen = args.len_range[1]
    minarea = args.area_range[0]
    maxarea = args.area_range[1]
    arealist = arealist__
    plist = plist__
    save_valid_count = 0
    save_invalid_count = 0
    global_iteration = 0
    MASS_OUTPUT_ALL_THRESHOLD = args.MASS_OUTPUT_ALL_THRESHOLD
    LOGFOLDER = os.path.join(args.save_path, args.run_id)

    if not os.path.exists(LOGFOLDER): os.mkdir(LOGFOLDER)

    ALLFOLDER = os.path.join(LOGFOLDER, 'Reward_ALL_Result/')
    print(ALLFOLDER)
    if not os.path.exists(ALLFOLDER): os.mkdir(ALLFOLDER)
    OUTPUT_ALL_MAX = 10000

    MASS_ALLFOLDER = os.path.join(LOGFOLDER, 'MASS_ALL_Result/')
    if not os.path.exists(MASS_ALLFOLDER): os.mkdir(MASS_ALLFOLDER)
    MASS_OUTPUT_ALL_MAX = 10000

    DIVERSITY_FOLDER = os.path.join(LOGFOLDER, 'DIVERSITY_result/')
    if not os.path.exists(DIVERSITY_FOLDER): os.mkdir(DIVERSITY_FOLDER)

    DIVERSITY_TOPO_FOLDER = os.path.join(LOGFOLDER, 'DIVERSITY_TOPO_result/')
    if not os.path.exists(DIVERSITY_TOPO_FOLDER): os.mkdir(DIVERSITY_TOPO_FOLDER)
    logfile = open(os.path.join(LOGFOLDER, args.logfile_stage1), 'w')

def diversity_save(reward, reward_count, Mass, Dis_value, Stress_value, Buckle_value, p, e):
    global save_valid_count
    FILES = os.listdir(ALLFOLDER)
    if (Mass > MASS_OUTPUT_ALL_THRESHOLD): return

    MASS_FILES = os.listdir(MASS_ALLFOLDER)
    if len(MASS_FILES) < MASS_OUTPUT_ALL_MAX:
        OUTFILE = MASS_ALLFOLDER + str(round(Mass)) + '.txt'
        if (not os.path.exists(OUTFILE)):
            save_file_stage1(OUTFILE, p, e)
            save_valid_count += 1

    DIVERSITY_FILES = os.listdir(DIVERSITY_FOLDER)
    d_flag = True
    for d_file in DIVERSITY_FILES:
        p_tmp, e_tmp = readFile(os.path.join(DIVERSITY_FOLDER, d_file))
        mass_tmp = int(d_file[:-4])
        if (similar_position(p, e, p_tmp, e_tmp)):
            d_flag = False
            if (mass_tmp <= Mass): continue
            os.remove(os.path.join(DIVERSITY_FOLDER, d_file))
            OUTFILE = DIVERSITY_FOLDER + str(round(Mass)) + '.txt'
            if (not os.path.exists(OUTFILE)):
                save_file_stage1(OUTFILE, p, e)

    if (d_flag):
        OUTFILE = DIVERSITY_FOLDER + str(round(Mass)) + '.txt'
        if (not os.path.exists(OUTFILE)):
            save_file_stage1(OUTFILE, p, e)

    DIVERSITY_TOPO_FILES = os.listdir(DIVERSITY_TOPO_FOLDER)
    d_flag = True
    for d_file in DIVERSITY_TOPO_FILES:
        p_tmp, e_tmp = readFile(os.path.join(DIVERSITY_TOPO_FOLDER, d_file))
        mass_tmp = int(d_file[:-4])
        if (similar_topo(p, e, p_tmp, e_tmp)):
            d_flag = False
            if (mass_tmp > Mass):
                os.remove(os.path.join(DIVERSITY_TOPO_FOLDER, d_file))
                OUTFILE = DIVERSITY_TOPO_FOLDER + str(round(Mass)) + '.txt'
                if (not os.path.exists(OUTFILE)): save_file_stage1(OUTFILE, p, e)
            break

    if (d_flag):
        OUTFILE = DIVERSITY_TOPO_FOLDER + str(round(Mass)) + '.txt'
        if (not os.path.exists(OUTFILE)):
            save_file_stage1(OUTFILE, p, e)

            
def soft_reward(env_output, p, e):
    reward, reward_cnt, Mass, Dis_value, Stress_value, Buckle_value = env_output
    global pbest
    global ebest
    global bestreward
    global tmpbestreward
    global save_invalid_count
    if (reward <= 0):
        if (save_invalid_count < save_valid_count * args.save_invalid_factor and reward == 0):
            folder = os.path.join(LOGFOLDER, "invalid")
            if (not os.path.exists(folder)): os.mkdir(folder)
            save_file_stage1(os.path.join(folder, str(save_invalid_count) + '.txt'), p, e)
            save_invalid_count += 1
        return reward
        
    if (bestreward > reward):
        bestreward = reward
        pbest = copy.deepcopy(p)
        ebest = copy.deepcopy(e)
    if (tmpbestreward > reward):
        tmpbestreward = reward
    if (args.save_diversity):
        diversity_save(reward, reward_cnt, Mass, Dis_value, Stress_value, Buckle_value, p, e)
    reward= args.reward_lambda / (reward * reward)
    return reward

def inlen(u, v):
    if (getlen2(u, v) > maxlen and args.CONSTRAINT_MAX_LENGTH):
        return False
    if (getlen2(u, v) < minlen and args.CONSTRAINT_MIN_LENGTH):
        return False
    return True

def canadd(N1, N2, p, e, area = None, d = None, t = None):
    if (not inlen(p[N1], p[N2])): return False
    if (args.CONSTRAINT_CROSS_EDGE == 1):
        if (area == None and d == None and t == None):
            if (args.env_dims == 2):
                for i in range(len(e)):
                    N3 = e[i].u
                    N4 = e[i].v
                    if (transintersect(N1, N2, N3, N4, p)): return False
            return True
        for i in range(len(e)):
            if (N1 == e[i].v or N1 == e[i].u or N2 == e[i].v or N2 == e[i].u): continue
            _, _, dis = closestDistanceBetweenLines(p[N1].Point2np(), p[N2].Point2np(), p[e[i].u].Point2np(), p[e[i].v].Point2np(), clampAll = True)
            if (d != None):
                r1, r2 = d / 2, e[i].d / 2
            else:
                if (args.env_dims == 2):
                    r1, r2 = area / 2, e[i].area / 2
                elif (args.env_dims == 3):
                    r1, r2 = np.sqrt(area / np.pi), np.sqrt(e[i].area / np.pi)
            if (dis <= r1 + r2): return False
    return True

def max_can_add_area(new_e, p, e):
    if (args.CONSTRAINT_CROSS_EDGE == 0): 
        if (args.env_mode == 'DT'):
            return None, arealist[-1][4]
        elif (args.env_mode == 'Area'):
            return maxarea, None
    if (args.env_mode == 'DT'):
        mind = arealist[-1][4]
        for i in range(len(e)):
            if (new_e.u == e[i].v or new_e.u == e[i].u or new_e.v == e[i].v or new_e.v == e[i].u): continue
            _, _, dis = closestDistanceBetweenLines(p[new_e.u].Point2np(), p[new_e.v].Point2np(), p[e[i].u].Point2np(), p[e[i].v].Point2np(), clampAll = True)
            mind = min(mind, (dis - e[i].d / 2) * 2)
        return None, mind
    elif (args.env_mode == 'Area'):
        mina = maxarea
        for i in range(len(e)):
            if (new_e.u == e[i].v or new_e.u == e[i].u or new_e.v == e[i].v or new_e.v == e[i].u): continue
            _, _, dis = closestDistanceBetweenLines(p[new_e.u].Point2np(), p[new_e.v].Point2np(), p[e[i].u].Point2np(), p[e[i].v].Point2np(), clampAll = True)
            if (args.env_dims == 2):
                mina = min(mina, (dis - e[i].area / 2) * 2)
            elif (args.env_dims == 3):
                mina = min(mina, np.pi * ((dis - np.sqrt(e[i].area / np.pi)) ** 2))
        return mina, None

class Action():
    def __init__(self, opt, u = -1, v = -1, area = -1, vec = Vector3(), eid = 0, d = None, t = None):
        self.opt = opt
        self.stateid = -1
        self.d = None
        self.t = None
        
        if (opt == 0): self.vec = vec           #add node
            
        if (opt == 1):                          #add edge
            self.opt = opt
            self.u = u
            self.v = v
            if (area != -1): self.area = area
            else: self.area = maxarea
            if (args.env_mode == 'DT'):
                self.d = arealist[-1][4]
                self.t = arealist[-1][5]

        if (opt == 2):                          #change area
            if (args.env_mode == 'DT'): assert(d > 0 and t > 0)
            self.eid = eid
            if (area != -1): self.area = area
            else: self.area = maxarea
            self.d = d
            self.t = t

    def __str__(self):
        if (self.opt == 0): return "add node at" + self.vec.__str__()
        if (self.opt == 1):
            if (self.area < 1e-8): return "do nothing"
            else: return "add edge between" + str(self.u) + "and" + str(self.v)
        if (self.opt == 2): return "modify area of " + str(self.eid) + "to " + str(self.area)

class State():
    def __init__(self, p, e, opt, fa = 0, elist = set(), eid = 0):
        self.opt = opt
        self.sons = []
        self.isEnd = False
        self.n = 0
        self.q = 0
        self.fa = fa
        self.allvisited = False
        self.w = 0.0
        self.sumq = 0.0
        self.sumw = 0.0
        self.mq = -1000
        if (opt == 0):
            for i in range(len(plist)):
                flag = False
                tmpp = plist[i]
                for j in range(len(p)):
                    if (inlen(Point(tmpp), p[j])):
                        flag = True
                        break
                if (flag == True): self.sons.append(Action(0, vec = tmpp))

            for i in range(len(self.sons), args.initson):
                flag = False
                tmpp = Vector3(getrand(minx, maxx), getrand(miny, maxy), getrand(minz, maxz))
                for j in range(len(p)):
                    if (inlen(Point(tmpp), p[j])):
                        flag = True
                        break
                if (flag == True): self.sons.append(Action(0, vec = tmpp))
                
        if (opt == 1):
            self.reward = soft_reward(reward_fun(p, e), p, e)
            if (self.reward > 1e-7): self.sons.append(Action(1)) # 当前结构稳定
            for i in elist: 
                self.sons.append(Action(1, u = i[0], v = i[1]))
            if (len(self.sons) == 0):
                self.isEnd = True
                self.reward = soft_reward(reward_fun(p, e), p, e)

        if (opt == 2):
            self.eid = eid
            if (eid >= len(e)):
                self.isEnd = True
                self.reward = soft_reward(reward_fun(p, e), p, e)
            else:
                if (args.env_mode == 'DT'):
                    for i in range(len(arealist)):
                        self.sons.append(Action(2, d = arealist[i][4], t = arealist[i][5], eid = eid))
                else:
                    for i in range(args.initson + 1):
                        self.sons.append(Action(2, area = minarea + (maxarea - minarea) / args.initson * i))

    def findunvis(self): # find unvisited node
        ret = -1
        for i in range(len(self.sons)):
            if (self.sons[i].stateid == -1):
                ret = i
                break
        if (ret == -1): self.allvisited = True
        return ret

def bestchild(now, c, alpha):
    global statelist
    ret = -1
    actid = -1
    mx = 0
    if (abs(c) < 1e-7): # final find, no explore 
        for i in range(len(statelist[now].sons)):
            v = statelist[now].sons[i].stateid
            if (statelist[now].opt == 1):
                tmp = alpha * statelist[v].q / statelist[v].n + (1 - alpha) * statelist[v].mq
                print(statelist[v].q, statelist[v].n, statelist[v].mq, statelist[v].q / statelist[v].n, 'a')
            else:
                tmp = alpha * statelist[v].sumq / statelist[v].n + (1 - alpha) * statelist[v].mq
                print(statelist[v].q, statelist[v].sumq, statelist[v].mq, statelist[v].w, statelist[v].sumq / statelist[v].n, statelist[v].n)
            if (ret == -1 or tmp > mx):
                ret = v
                mx = tmp
                actid = i
        print("**************")
        print(round(statelist[ret].n,2), round(statelist[ret].mq,2))
        print("**************")
        
        return ret, statelist[now].sons[actid]
    
    if (statelist[now].opt == 1 or statelist[now].opt == 2):
        for i in range(len(statelist[now].sons)):
            v = statelist[now].sons[i].stateid
            tmp = alpha * statelist[v].q / statelist[v].n + (1 - alpha) * statelist[v].mq + c * math.sqrt(2 * math.log(statelist[now].n) / statelist[v].n)
            if (ret == -1 or tmp > mx):
                ret = v
                mx = tmp
                actid = i
    else: # use kernel
        for i in range(len(statelist[now].sons)):
            v = statelist[now].sons[i].stateid
            if (statelist[v].w < 1e-7):
                ret = v
                actid = i
                break
            tmp = alpha * statelist[v].q / statelist[v].w + (1 - alpha) * statelist[v].mq
            tmp = tmp + c * (math.sqrt(2 * math.log(statelist[now].sumw) / statelist[v].w) * 0.8 + math.sqrt(2 * math.log(statelist[now].n) / statelist[v].n) * 0.2)
            if (ret == -1 or tmp > mx):
                ret = v
                mx = tmp
                actid = i
    return ret, statelist[now].sons[actid]

def take_action(p, e, elist, act):
    if (act.opt == 0):
        p.append(Point(act.vec))
        if (len(p) == args.maxp):
            for i in range(len(p)):
                for j in range(i + 1, len(p)):
                    if (not (p[i].isSupport and p[j].isSupport)) and inlen(p[i], p[j]):
                        elist.add((i, j))

    if (act.opt == 1):
        if (act.u != -1 and act.v != -1):
            e.append(Bar(act.u, act.v, act.area, getlen2(p[act.u], p[act.v]), d = act.d, t = act.t))
            elist.remove((act.u, act.v))
            dellist = []
            if (args.env_dims == 2 and args.CONSTRAINT_CROSS_EDGE == 1):
                for i in elist:
                    if (transintersect(act.u, act.v, i[0], i[1], p)):
                        dellist.append(i)
                for i in dellist:
                    elist.remove(i)
        else: elist.clear()

    if (act.opt == 2):
        if args.env_mode == 'DT':
            e[act.eid].d = act.d
            e[act.eid].t = act.t
        else: e[act.eid]._area = act.area

def isok(vec):
    if (minx <= vec.x and vec.x <= maxx and miny <= vec.y and vec.y <= maxy and minz <= vec.z and vec.z <= maxz):
        return True
    return False

def getnewchild0(stateid, vec):
    assert(statelist[stateid].opt == 0)
    finvec = Vector3()
    bestw = 0.0

    for iter in range(args.maxnum):
        newvec = vec + randpoint() * args.sgm2 * 0.5
        newvec.x = max(min(newvec.x, maxx), minx)
        newvec.y = max(min(newvec.y, maxy), miny)
        newvec.z = max(min(newvec.z, maxz), minz)
        w = 0.0
        for i in range(len(statelist[stateid].sons)):
            w = w + Kernel2(newvec, statelist[stateid].sons[i].vec, args.sgm2) * statelist[statelist[stateid].sons[i].stateid].n
        if (iter == 0 or w < bestw):
            bestw = w
            finvec = newvec

    statelist[stateid].sons.append(Action(0, vec = finvec))
    return len(statelist[stateid].sons) - 1
    
def getnewchild2(stateid, area):
    assert(statelist[stateid].opt == 2)
    finarea = maxarea
    bestw = 0.0

    for iter in range(args.maxnum):
        newarea = getrand(max(minarea, area - args.sgm1 * 3), min(maxarea, area + args.sgm1 * 3))
        w = 0.0
        for i in range(len(statelist[stateid].sons)):
            w = w + Kernel1(newarea, statelist[stateid].sons[i].area, args.sgm1)
        if (iter == 0 or w < bestw):
            bestw = w
            finarea = newarea

    statelist[stateid].sons.append(Action(2, area = finarea, eid = statelist[stateid].eid))
    return len(statelist[stateid].sons) - 1


def treepolicy(stateid, p_, e_, elist_):
    p = copy.deepcopy(p_)
    e = copy.deepcopy(e_)
    elist = copy.deepcopy(elist_)
    global statelist
    now = stateid
    sonid = -1
    while ((not statelist[now].isEnd) and (sonid == -1)):
        opt = statelist[now].opt
        ret = statelist[now].findunvis()
        if (ret != -1):
            sonid = ret
            break
        
        nxt, act = bestchild(now, args.c[opt], args.alpha)
        if (opt == 1):
            now = nxt
            take_action(p, e, elist, act)
        
        elif (opt == 0):
            sizeA = len(statelist[now].sons)
            if (sizeA * sizeA * 3 > statelist[now].n and (not statelist[nxt].isEnd or len(statelist[now].sons) > args.maxson)): 
                now = nxt
                take_action(p, e, elist, act)
            else: 
                sonid = getnewchild0(now, act.vec) #add new child
        
        elif (opt == 2):
            sizeA = len(statelist[now].sons)
            if (sizeA * sizeA * 3 > statelist[now].n and (not statelist[nxt].isEnd or len(statelist[now].sons) > args.maxson)) or (args.env_mode == 'DT'): 
                now = nxt
                take_action(p, e, elist, act)
            else: 
                sonid = getnewchild2(now, act.area) #add new child


    if (sonid >= 0): # add new child
        act = statelist[now].sons[sonid]
        take_action(p, e, elist, act)
        opt = statelist[now].opt
        if (opt == 0):
            if (len(p) == args.maxp): newstate = State(p, e, 1, fa = now, elist = elist)
            else: newstate = State(p, e, 0, fa = now)
            for i in range(len(statelist[now].sons)):
                if (i == sonid): continue
                if (opt == 0):
                    KAB = Kernel2(act.vec, statelist[now].sons[i].vec, args.sgm2)
                else:
                    KAB = Kernel1(act.area, statelist[now].sons[i].area, args.sgm1)
                sid = statelist[now].sons[i].stateid
                newstate.w = newstate.w + statelist[sid].n * KAB
                newstate.q = newstate.q + statelist[sid].sumq * KAB
        if (opt == 1):
            if (act.u < 0): newstate = State(p, e, 2, fa = now, eid = 0)
            else: newstate = State(p, e, 1, fa = now, elist = elist)
        if (opt == 2): 
            newstate = State(p, e, 2, fa = now, eid = act.eid + 1)
        statelist.append(newstate)
        statelist[now].sons[sonid].stateid = len(statelist) - 1
        now = len(statelist) - 1
    return now, p, e, elist

def defaultpolicy(stateid, p, e, elist):
    opt = statelist[stateid].opt
    if (opt == 0):
        while (len(p) < args.maxp):
            p.append(Point(Vector3(getrand(minx, maxx), getrand(miny, maxy), getrand(minz, maxz))))
        opt = 1

    if (opt == 1):
        for i in range(len(e)):
            area, d = max_can_add_area(e[i], p, e[0 : i])
            if (args.env_mode == 'DT'):
                can_id = 0
                while (can_id < len(arealist) - 1 and arealist[can_id + 1][4] <= d): can_id += 1
                area_random = arealist[random.randint(0, can_id)]
                e[i].d = area_random[4]
                e[i].t = area_random[5]
            else: e[i]._area = getrand(minarea, area)

        el = []
        for i in elist: el.append(i)
        if (len(el) == 0 and len(e) == 0):
            for i in range(len(p)):
                for j in range(i + 1, len(p)):
                    if (p[i].isSupport and p[j].isSupport): continue
                    if (not inlen(p[i], p[j])): continue
                    el.append((i, j))
        random.shuffle(el)
        ret = -1
        for i in range(len(el)):
            probnow = random.random()
            if (probnow > args.prob): continue
            u = el[i][0]
            v = el[i][1]
            if (args.env_mode == 'DT'): 
                area = 1.0
                area_random = arealist[random.randint(0, len(arealist) - 1)]
                d, t = area_random[4], area_random[5]
            elif (args.env_mode == 'Area'):
                area = getrand(minarea, maxarea)
                d, t = None, None
            if (canadd(u, v, p, e, area, d, t)):
                e.append(Bar(u, v, leng = getlen2(p[u], p[v]), area = area, d = d, t = t))
                ret = soft_reward(reward_fun(p, e), p, e)
                if (ret > 1e-7): return ret
        return ret

    if (opt == 2):
        for i in range(statelist[stateid].eid, len(e)):
            area, d = max_can_add_area(e[i], p, e[0 : i])
            if (args.env_mode == 'DT'):
                can_id = 0
                while (can_id < len(arealist) - 1 and arealist[can_id + 1][4] <= d): can_id += 1
                area_random = arealist[random.randint(0, can_id)]
                e[i].d = area_random[4]
                e[i].t = area_random[5]
            else: e[i]._area = getrand(minarea, area)
        ret = soft_reward(reward_fun(p, e), p, e)
        return ret

    assert(False)

def backup(now, delta, root):
    fa = statelist[now].fa
    while (True):
        statelist[now].n = statelist[now].n + 1
        statelist[now].sumq = statelist[now].sumq + delta
        if (statelist[now].mq < delta):
            statelist[now].mq = delta
        if (now == root): break
        if (statelist[fa].opt == 1 or statelist[fa].opt == 2):
            statelist[now].q = statelist[now].q + delta
        elif (statelist[fa].opt == 0):  
            sonid = -1
            for i in range(len(statelist[fa].sons)):
                if (statelist[fa].sons[i].stateid == now):
                    sonid = i
                    break
            assert(sonid != -1)
            vec0 = statelist[fa].sons[sonid].vec
            for i in range(len(statelist[fa].sons)):
                KAB = Kernel2(vec0, statelist[fa].sons[i].vec, args.sgm2)
                sid = statelist[fa].sons[i].stateid
                statelist[sid].w = statelist[sid].w + KAB
                statelist[sid].q = statelist[sid].q + KAB * delta
                statelist[fa].sumw = statelist[fa].sumw + KAB
        now = fa
        fa = statelist[now].fa

def UCTSearch(p, e):
    global statelist
    global bestreward
    global tmpbestreward
    global elist
    global global_iteration
    elist = set()
    statelist = []
    step_node = 1
    opt = 0
    eidnow = 0
    maxiter = args.UCT_maxiter
    root = 0
    while (not (opt == 2 and eidnow >= len(e))):
        statelist.clear()
        tmpbestreward = 1e9
        tmpbestreward2 = 1e9
        root = 0
        if (opt == 0):
            statelist.append(State(p, e, 0, -1))
            if (len(pbest) > len(p)):
                statelist[root].sons.append(Action(0, vec = pbest[len(p)].vec))
        if (opt == 1):
            statelist.append(State(p, e, 1, -1, elist = elist))
        if (opt == 2):
            statelist.append(State(p, e, 2, -1, eid = eidnow))
        extra_iter = 0
        if (opt == 0): extra_iter = args.UCT_extra_iter_for_point_pos
        for iter in range(maxiter + extra_iter):
            tmp, ptmp, etmp, elisttmp = treepolicy(root, p, e, elist)
            delta = defaultpolicy(tmp, ptmp, etmp, elisttmp)
            backup(tmp, delta, root)
            if (iter % 200 == 0):
                tmpbestreward2 = min(tmpbestreward2, tmpbestreward)
                print(global_iteration + iter, iter, bestreward, tmpbestreward2, tmpbestreward, len(statelist[root].sons))
                print(global_iteration + iter, iter, bestreward, tmpbestreward2, tmpbestreward, len(statelist[root].sons), file = logfile)
                tmpbestreward = 1e9
        global_iteration += maxiter + extra_iter
        
        if (opt == 0):
            act = Action(0, vec = pbest[len(p)].vec)
            # for SaveKR:
            if args.save_KR == True:
                KR_plist_x = []
                KR_plist_y = []
                KR_plist_z = []
                print(len(p)+1,"*****************************************")
#                print(len(p)+1,"*****************************************", file = LOG_result)
                for ii in range(len(statelist[root].sons)):
                    KR_plist_x.append(statelist[root].sons[ii].vec.x)
                    KR_plist_y.append(statelist[root].sons[ii].vec.y)
                    KR_plist_z.append(statelist[root].sons[ii].vec.z)
                    print(ii,statelist[root].sons[ii].vec.x, statelist[root].sons[ii].vec.y, statelist[root].sons[ii].vec.z)
#                    print(ii,statelist[root].sons[ii].vec.x, statelist[root].sons[ii].vec.y, statelist[root].sons[ii].vec.z, file = LOG_result)

                print(len(p)+1,"*****************************************")
#                print(len(p)+1,"*****************************************", file = LOG_result)
                
                X1=np.ones(len(KR_plist_x))
                Y1=np.ones(len(KR_plist_x))
                Z1=np.ones(len(KR_plist_x))
                
                fig1 = plt.figure()
                ax1 = plt.axes(projection='3d')
                for i in range(len(KR_plist_x)):
                    ax1.scatter3D([KR_plist_x[i]], [KR_plist_y[i]], [KR_plist_z[i]], color='b')
                    X1[i] = KR_plist_x[i]
                    Y1[i] = KR_plist_y[i]
                    Z1[i] = KR_plist_z[i]
                # Create cubic bounding box to simulate equal aspect ratio
                max_range = np.array([X1.max()-X1.min(), Y1.max()-Y1.min(), Z1.max()-Z1.min()]).max()
                Xb1 = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X1.max()+X1.min())
                Yb1 = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y1.max()+Y1.min())
                Zb1 = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z1.max()+Z1.min())
                
                # Comment or uncomment following both lines to test the fake bounding box:
                for xb1, yb1, zb1 in zip(Xb1, Yb1, Zb1):
                    ax1.plot([xb1], [yb1], [zb1], 'w')
                
                # plt.scatter(KR_plist_x, KR_plist_y, color='b')
                # plt.axis("equal")
                inputname = FILENAME.replace(".txt","_")
                FILENAME_add_node_jpg = "./results/" + time_str + "_" + str(args.maxp) + "p_" + inputname + "add-node-"+str(step_node) + ".jpg"
                plt.savefig(FILENAME_add_node_jpg, dpi = 1000)
                plt.close()
                step_node = step_node + 1
                print(len(KR_plist_x))

        if (opt == 1):
            if (len(e) == len(ebest)): act = Action(1)
            else: act = Action(1, u = ebest[len(e)].u, v = ebest[len(e)].v)

        if (opt == 2):
            act = Action(2, area = ebest[eidnow].area, 
                eid = eidnow,
                d = ebest[eidnow].d, 
                t = ebest[eidnow].t
            )

        take_action(p, e, elist, act)
        
        print(act)
        print(bestreward, tmpbestreward)
        
        if (opt == 0):
            if (len(p) == args.maxp): opt = 1
        elif (opt == 1):
            if (act.u == -1): opt = 2
        elif (opt == 2):
            eidnow = eidnow + 1
    return bestreward, pbest, ebest