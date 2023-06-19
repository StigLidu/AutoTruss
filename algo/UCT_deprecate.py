from glob import glob
from utils.utils import Vector3, Point, Bar
from utils.utils import getrand, randpoint, Kernel1, Kernel2, getlen, getlen2
from truss_envs.reward import *
import math
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
import time
import heapq

def intersect(N1, N2, N3, N4):
    game_stop = False
    X1=N1.x
    Y1=N1.y
    X2=N2.x
    Y2=N2.y
    X3=N3.x
    Y3=N3.y
    X4=N4.x
    Y4=N4.y

    if N1 != N3 and N1 != N4 and N2 != N3 and N2 != N4:
        SIN13_14=(X3-X1)*(Y4-Y1)-(X4-X1)*(Y3-Y1)
        SIN23_24=(X3-X2)*(Y4-Y2)-(X4-X2)*(Y3-Y2)
        SIN31_32=(X1-X3)*(Y2-Y3)-(X2-X3)*(Y1-Y3)
        SIN41_42=(X1-X4)*(Y2-Y4)-(X2-X4)*(Y1-Y4)

        if SIN13_14*SIN23_24<=0 and SIN31_32*SIN41_42<=0:
            SIN12_23=(X2-X1)*(Y3-Y2)-(X3-X2)*(Y2-Y1)
            SIN12_24=(X2-X1)*(Y4-Y2)-(X4-X2)*(Y2-Y1)
            SIN23_34=(X3-X2)*(Y4-Y3)-(X4-X3)*(Y3-Y2)
            SIN13_34=(X3-X1)*(Y4-Y3)-(X4-X3)*(Y3-Y1)

            if SIN12_23!=0 and SIN12_24!=0 and SIN23_34!=0 and SIN13_34!=0:
                game_stop=True


    SIN13_14=(X3-X1)*(Y4-Y1)-(X4-X1)*(Y3-Y1)
    SIN23_24=(X3-X2)*(Y4-Y2)-(X4-X2)*(Y3-Y2)
    if (abs(SIN13_14) < 1e-7 and abs(SIN23_24) < 1e-7):
        D13 = math.sqrt((X3 - X1) * (X3 - X1) + (Y3 - Y1) * (Y3 - Y1))
        D14 = math.sqrt((X4 - X1) * (X4 - X1) + (Y4 - Y1) * (Y4 - Y1))
        D23 = math.sqrt((X3 - X2) * (X3 - X2) + (Y3 - Y2) * (Y3 - Y2))
        D24 = math.sqrt((X4 - X2) * (X4 - X2) + (Y4 - Y2) * (Y4 - Y2))
        D1 = D13 + D24
        D2 = D23 + D14
        if (abs(D1 - D2) > 1e-7):
            game_stop = True
    return game_stop


def transintersect(u1, v1, u2, v2, p):
    if (intersect(p[u1].vec, p[v1].vec, p[u2].vec, p[v2].vec)):
        return True
    return False

class Action():
    def __init__(self, args, opt, u = -1, v = -1, area = -1, vec = Vector3(), eid = 0, d = None, t = None):
        self.opt = opt
        self.stateid = -1
        self.d = None
        self.t = None
        if (opt == 0): self.vec = vec   # add node
        if (opt == 1):                  # add edge
            self.opt = opt
            self.u = u
            self.v = v
            if (area != -1): self.area = area
            else: self.area = args.maxarea
            if (args.env_mode == 'DT'):
                self.d = arealist[-1][4]
                self.t = arealist[-1][5]
        if (opt == 2):                  # change area
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

class UCTs():
    def UCTs_init(self, args, plist__ = [], arealist__ = []):
        self.args = args
        self.pbest = []
        self.ebest = []
        self.time_str = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime())
        self.bestreward = 1e9
        self.tempbestreward = 1e9
        self.minx = args.coordinate_range[0][0]
        self.maxx = args.coordinate_range[0][1]
        self.miny = args.coordinate_range[1][0]
        self.maxy = args.coordinate_range[1][1]
        self.statelist = []
        if (args.env_dims == 3):
            self.minz = args.coordinate_range[2][0]
            self.maxz = args.coordinate_range[2][1]
        else: self.minz = self.maxz = 0
        self.minlen = args.len_range[0]
        self.maxlen = args.len_range[1]
        self.minarea = args.area_range[0]
        self.maxarea = args.area_range[1]
        self.arealist = arealist__
        self.plist = plist__
        self.save_valid_count = 0
        self.save_invalid_count = 0
        self.output_dir_init()
    
    def output_dir_init(self):
        self.OUTPUT_ALL_THRESHOLD = self.args.OUTPUT_ALL_THRESHOLD
        self.MASS_OUTPUT_ALL_THRESHOLD = self.args.MASS_OUTPUT_ALL_THRESHOLD
        self.LOGFOLDER = self.args.save_path

        if not os.path.exists(self.LOGFOLDER): os.mkdir(self.LOGFOLDER)

        self.ALLFOLDER = self.LOGFOLDER + 'Reward_ALL_Result/'
        if not os.path.exists(self.ALLFOLDER): os.mkdir(self.ALLFOLDER)
        self.OUTPUT_ALL_MAX = 10000
        print('OUTPUT_ALL_THRESHOLD:', self.OUTPUT_ALL_THRESHOLD)

        self.MASS_ALLFOLDER = self.LOGFOLDER + 'MASS_ALL_Result/'
        if not os.path.exists(self.MASS_ALLFOLDER): os.mkdir(self.MASS_ALLFOLDER)
        self.MASS_OUTPUT_ALL_MAX = 10000
        print('MASS_OUTPUT_ALL_THRESHOLD:', self.MASS_OUTPUT_ALL_THRESHOLD)

    def similar(self, tup1, tup2):
        p1 = tup1[2]
        e1 = tup1[3]
        p2 = tup2[2]
        e2 = tup2[3]
        pts1 = []
        pts2 = []
        for p in p1: pts1.append([p.vec.x, p.vec.y, p.vec.z])
        for p in p2: pts2.append([p.vec.x, p.vec.y, p.vec.z])
        es1 = []
        es2 = []
        for e in e1:
            es1.append([pts1[e.u], pts1[e.v]])
            es1.append([pts1[e.v], pts1[e.u]])
        for e in e2:
            es2.append([pts2[e.u], pts2[e.v]])
            es2.append([pts2[e.v], pts2[e.u]])
        if sorted(pts1) != sorted(pts2): return False
        if sorted(es1) != sorted(es2): return False
        return True

    def save_file(self, OUTFILE, p, e, valid = True):
        if (valid): self.save_valid_count += 1
        else: self.save_invalid_count += 1
        with open(OUTFILE, "w") as f:
            print(len(p), len(e), file=f)
            for i in range(len(p)):
                print(p[i].vec.x, p[i].vec.y, p[i].vec.z, p[i].supportX, p[i].supportY,
                    p[i].supportZ, p[i].loadX, p[i].loadY, p[i].loadZ, file=f)
            for i in range(len(e)): print(e[i].u, e[i].v, e[i].area, e[i].d, e[i].t, file = f)

    def diversity_save(self, reward, reward_count, Mass, Dis_value, Stress_value, Buckle_value, p, e):
        FILES = os.listdir(self.ALLFOLDER)
        if len(FILES) < self.OUTPUT_ALL_MAX and reward <= self.OUTPUT_ALL_THRESHOLD:
            OUTFILE = self.ALLFOLDER + str(reward_count).zfill(len(str(self.OUTPUT_ALL_MAX))) + '_' + str(
                round(Mass)) + '_' + str(round(reward)) + '.txt'
            self.save_file(OUTFILE, p, e)

        MASS_FILES = os.listdir(self.MASS_ALLFOLDER)
        if len(MASS_FILES) < self.MASS_OUTPUT_ALL_MAX and Mass <= self.MASS_OUTPUT_ALL_THRESHOLD:
            if not (Dis_value > 1e-7 or Stress_value > 1e-7 or Buckle_value > 1e-7):
                OUTFILE = self.MASS_ALLFOLDER + str(round(Mass)) + '.txt'
                self.save_file(OUTFILE, p, e)
            
    def soft_reward(self, env_output, p, e):
        reward, reward_cnt, Mass, Dis_value, Stress_value, Buckle_value = env_output
        if (reward <= 0):
            if (self.save_invalid_count * self.args.save_invalid_factor < self.save_valid_count and reward == 0):
                folder = os.path.join(self.LOGFOLDER, "invalid")
                if (not os.path.exists(folder)): os.mkdir(folder)
                self.save_file(os.path.join(folder, str(self.save_invalid_count) + '.txt'), p, e)
            return reward
        if (self.bestreward > reward):
            self.bestreward = reward
            self.pbest = copy.deepcopy(p)
            self.ebest = copy.deepcopy(e)
        if (self.tempbestreward > reward):
            self.tempbestreward = reward
        if (args.save_diversity):
            self.diversity_save(reward, reward_cnt, Mass, Dis_value, Stress_value, Buckle_value, p, e)
        reward= self.args.reward_lambda / (reward * reward)
        return reward

    def inlen(self, u, v):
        if (getlen2(u, v) > self.maxlen and self.args.CONSTRAINT_MAX_LENGTH): return False
        if (getlen2(u, v) < self.minlen and self.args.CONSTRAINT_MIN_LENGTH): return False
        return True

    def canadd(self, N1, N2, p, e):
        if (not self.inlen(p[N1], p[N2])): return False
        if (args.env_dims == 2 and args.CONSTRAINT_CROSS_EDGE == 1):
            for i in range(len(e)):
                N3 = e[i].u
                N4 = e[i].v
                if (transintersect(N1, N2, N3, N4, p)): return False
        return True

    def bestchild(self, now, c, alpha):
        ret = -1
        actid = -1
        mx = 0
        if (abs(c) < 1e-7): # final find, no explore 
            for i in range(len(self.statelist[now].sons)):
                v = self.statelist[now].sons[i].stateid
                if (self.statelist[now].opt == 1):
                    tmp = alpha * self.statelist[v].q / self.statelist[v].n + (1 - alpha) * self.statelist[v].mq
                    print(self.statelist[v].q, self.statelist[v].n, self.statelist[v].mq, self.statelist[v].q / self.statelist[v].n, 'a')
                else:
                    tmp = alpha * self.statelist[v].sumq / self.statelist[v].n + (1 - alpha) * self.statelist[v].mq
                    print(self.statelist[v].q, self.statelist[v].sumq, self.statelist[v].mq, self.statelist[v].w, self.statelist[v].sumq / statelist[v].n, statelist[v].n)
                if (ret == -1 or tmp > mx):
                    ret = v
                    mx = tmp
                    actid = i

            print("**************")
            print(round(self.statelist[ret].n,2), round(self.statelist[ret].mq,2))
            print("**************")
        
            return ret, self.statelist[now].sons[actid]
    
        if (self.statelist[now].opt == 1 or self.statelist[now].opt == 2):
            for i in range(len(self.statelist[now].sons)):
                v = self.statelist[now].sons[i].stateid
                tmp = alpha * self.statelist[v].q / self.statelist[v].n + (1 - alpha) * self.statelist[v].mq + c * math.sqrt(2 * math.log(self.statelist[now].n) / self.statelist[v].n)
                if (ret == -1 or tmp > mx):
                    ret = v
                    mx = tmp
                    actid = i
        else: # use kernel
            for i in range(len(self.statelist[now].sons)):
                v = self.statelist[now].sons[i].stateid
                if (self.statelist[v].w < 1e-7):
                    ret = v
                    actid = i
                    break
                tmp = alpha * self.statelist[v].q / self.statelist[v].w + (1 - alpha) * self.statelist[v].mq
                tmp = tmp + c * (math.sqrt(2 * math.log(self.statelist[now].sumw) / self.statelist[v].w) * 0.8 + math.sqrt(2 * math.log(self.statelist[now].n) / self.statelist[v].n) * 0.2)
                if (ret == -1 or tmp > mx):
                    ret = v
                    mx = tmp
                    actid = i
        return ret, self.statelist[now].sons[actid]

    def take_action(self, p, e, elist, act):
        if (act.opt == 0):
            p.append(Point(act.vec))
            if (len(p) == args.maxp):
                for i in range(len(p)):
                    for j in range(i + 1, len(p)):
                        if (not (p[i].isSupport and p[j].isSupport)) and self.inlen(p[i], p[j]):
                            elist.add((i, j))

        if (act.opt == 1):
            if (act.u != -1 and act.v != -1):
                e.append(Bar(act.u, act.v, act.area, getlen2(p[act.u], p[act.v]), d = act.d, t = act.t))
                elist.remove((act.u, act.v))
                if (args.env_dims == 2 and args.CONSTRAINT_CROSS_EDGE == 1):
                    dellist = []
                    for i in elist:
                        if (transintersect(act.u, act.v, i[0], i[1], p)):
                            dellist.append(i)
                    for i in dellist: elist.remove(i)
            else:
                elist.clear()

        if (act.opt == 2):
            if args.env_mode == 'DT':
                e[act.eid].d = act.d
                e[act.eid].t = act.t
            else:
                e[act.eid]._area = act.area

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
        newvec.x = max(min(newvec.y, maxy), miny)
        newvec.x = max(min(newvec.z, maxz), minz)
        w = 0.0
        for i in range(len(statelist[stateid].sons)):
            w = w + Kernel2(newvec, statelist[stateid].sons[i].vec, args.sgm2) * statelist[statelist[stateid].sons[i].stateid].n
        if (iter == 0 or w < bestw):
            bestw = w
            finvec = newvec

    statelist[stateid].sons.append(Action(0, vec = finvec))
    return len(statelist[stateid].sons) - 1
    
## gai
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
            if (sizeA * sizeA * 3 > statelist[now].n and (not statelist[nxt].isEnd or len(statelist[now].sons) > args.maxson)): 
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
            if (args.env_mode == 'DT'):
                area_random = arealist[random.randint(0, len(arealist) - 1)]
                e[i].d = area_random[4]
                e[i].t = area_random[5]
            else: e[i]._area = getrand(minarea, maxarea)
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
            if (canadd(u, v, p, e)):
                if (args.env_mode == 'DT'):
                    area_random = arealist[random.randint(0, len(arealist) - 1)]
                    e.append(Bar(u, v, leng = getlen2(p[u], p[v]), d=area_random[4], t=area_random[5]))
                else: e.append(Bar(u, v, leng = getlen2(p[u], p[v]), area = getrand(minarea, maxarea)))
                ret = soft_reward(reward_fun(p, e), p, e)
                if (ret > 1e-7): return ret
        return ret

    if (opt == 2):
        for i in range(statelist[stateid].eid, len(e)):
            if (args.env_mode == 'DT'):
                area_random = arealist[random.randint(0, len(arealist) - 1)]
                e[i].d = area_random[4]
                e[i].t = area_random[5]
            else: e[i]._area = getrand(minarea, maxarea)
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
    global bestreward
    global tmpbestreward
    global elist
    elist = set()
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
            if (iter % 100 == 0):
                tmpbestreward2 = min(tmpbestreward2, tmpbestreward)
                print(iter, bestreward, tmpbestreward2, tmpbestreward, len(statelist[root].sons))
                tmpbestreward = 1e9
    
        root2, tmpact = bestchild(root, 0.0, args.pickalpha)
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