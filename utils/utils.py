import math
import random
import os
import matplotlib.pyplot as plt
import shutil
import numpy as np
def util_init(args__):
    global args
    args = args__
    
class Vector3:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __add__(self, obj):
        return Vector3(self.x + obj.x, self.y + obj.y, self.z + obj.z)

    def __sub__(self, obj):
        return Vector3(self.x - obj.x, self.y - obj.y, self.z - obj.z)

    def __mul__(self, obj):
        if (type(obj) == Vector3):
            return Vector3(self.y * obj.z - self.z * obj.y, self.z * obj.x - self.x * obj.z,
                           self.x * obj.y - self.y * obj.x)
        if (type(obj) == float or type(obj) == int):
            return Vector3(self.x * obj, self.y * obj, self.z * obj)
        assert (False)

    def __str__(self):
        return str('(' + str(self.x) + ', ' + str(self.y) + ', ' + str(self.z) + ')')

    def length2(self):
        return float(self.x * self.x + self.y * self.y + self.z * self.z)

    def length(self):
        return (self.x * self.x + self.y * self.y + self.z * self.z) ** .5

    def norm(self):
        l = self.length()
        return Vector3(self.x / l, self.y / l, self.z / l)

    def __eq__(self, other):
        assert (type(other) == Vector3)
        if (abs(self.x - other.x) < 1e-8 and abs(self.y - other.y) < 1e-8 and abs(self.z - other.z) < 1e-8):
            return True
        else:
            return False
        

class Point:

    def __init__(self, vec=Vector3(), supportX = 0, supportY = 0, supportZ = 0, loadX = 0.0, loadY = 0.0, loadZ = 0.0):
        
        self.vec = vec
        self.supportX = supportX
        self.supportY = supportY
        self.supportZ = supportZ
        self.isSupport = False
        if (supportX == 1 and supportY == 1 and supportZ == 1):
            self.isSupport = True
        self.loadX = loadX
        self.loadY = loadY
        self.loadZ = loadZ
        self.isLoad = False
        if (abs(loadX) > 1e-7 or abs(loadY) > 1e-7 or abs(loadZ) > 1e-7):
            self.isLoad = True
    
    def Point2np(self):
        return np.array([self.vec.x, self.vec.y, self.vec.z])

class Bar:
    def __init__(self, u=-1, v=-1, area=1.0, leng=0.0, inertia=1.0, name_s = 'dt', d = None, t = None):
        self.u = int(u)
        self.v = int(v)
        self.d = d
        self.t = t
        self._area = float(area)
        self._inertia = inertia
        self.force = 0.0
        self.len = leng
        self.stress = 0.0 # calculate in dynamic
        self.name_s = name_s
            
    @property
    def area(self):
        if (self.d == None): return self._area
        else: return math.pi * self.d ** 2 / 4.0 - math.pi * (self.d - 2 * self.t) ** 2 / 4.0

    @property
    #TODO Ratio-ring
    def inertia(self):
        if (self.d == None): return self.area ** 2 * (1 + 0 ** 2) / (4 * math.pi * (1 - 0 ** 2))
        else: return math.pi * self.d ** 4 / 64.0 - math.pi * (self.d - 2 * self.t) ** 4 / 64.0

def randpoint():
    x = random.random() * 2.0 - 1.0
    y = random.random() * 2.0 - 1.0
    z = random.random() * 2.0 - 1.0
    while(x * x + y * y + z * z > 1.0):
        x = random.random() * 2.0 - 1.0
        y = random.random() * 2.0 - 1.0
        z = random.random() * 2.0 - 1.0
    return Vector3(x, y, z)

def getrand(x, y):
    if (x > y): return x
    return random.uniform(x, y)

def Kernel1(x1, x2, sgm):
    assert(type(x1) == float and type(x2) == float)
    return math.exp(-(x1 - x2) * (x1 - x2) / (2 * sgm * sgm))

def Kernel2(x1, x2, sgm):
    assert(type(x1) == Vector3 and type(x2) == Vector3)
    vec = x1 - x2
    len2 = vec.x * vec.x + vec.y * vec.y + vec.z * vec.z
    return math.exp(-(len2 / (2 * sgm * sgm)))

def getlen(vec):
    return math.sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z)

def getlen2(u, v):
    return getlen(u.vec - v.vec)

def getang(vec1, vec2):
    return (vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z) / (getlen(vec1) * getlen(vec2))

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

def getang(vec1, vec2):
    return (vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z) / (getlen(vec1) * getlen(vec2))

def transintersect(u1, v1, u2, v2, p):
    if (intersect(p[u1].vec, p[v1].vec, p[u2].vec, p[v2].vec)):
        return True
    return False

def readFile(input_file):
    r'''

    :param input_file: File name
    :return: point list, edge list
    '''

    p = []
    e = []

    with open(input_file, "r") as fle:
        lines = fle.readlines()
        for i in range(len(lines)):
            if len(lines[i]) < 2:
                continue
            line = lines[i]
            vec = line.strip().split(' ')
            if (i == 0):
                vn = int(vec[0])
                en = int(vec[1])
                continue

            if (1 <= i and i <= vn):
                p.append(Point(Vector3(float(vec[0]), float(vec[1]), float(vec[2])), int(vec[3]), int(vec[4]), int(vec[5]), float(vec[6]), float(vec[7]), float(vec[8])))
                continue

            if (vn + 1 <= i and i <= vn + en):
                if (len(vec) > 3 and vec[3] != 'None'):
                    d = float(vec[3])
                    t = float(vec[4])
                else:
                    d = None
                    t = None
                if (float(vec[2]) < 0): continue
                e.append(Bar(vec[0], vec[1], float(vec[2]), getlen2(p[int(vec[0])], p[int(vec[1])]), d = d, t = t))
    return p, e

def readFilewithload(input_file):
    r'''

    :param input_file: File name
    :return: point list, edge list
    '''

    p = []
    e = []
    load = []

    with open(input_file, "r") as fle:
        lines = fle.readlines()
        for i in range(len(lines)):
            if len(lines[i]) < 2:
                continue
            line = lines[i]
            vec = line.strip().split(' ')
            if (i == 0):
                vn = int(vec[0])
                en = int(vec[1])
                continue

            if (1 <= i and i <= vn):
                p.append(Point(Vector3(float(vec[0]), float(vec[1]), float(vec[2])), int(vec[3]), int(vec[4]), int(vec[5]), float(vec[6]), float(vec[7]), float(vec[8])))
                load.append(float(vec[7]))
                continue

            if (vn + 1 <= i and i <= vn + en):
                if (len(vec) > 3 and vec[3] != 'None'):
                    d = float(vec[3])
                    t = float(vec[4])
                else:
                    d = None
                    t = None
                if (float(vec[2]) < 0): continue
                e.append(Bar(vec[0], vec[1], float(vec[2]), getlen2(p[int(vec[0])], p[int(vec[1])]), d = d, t = t))
    return p, e, load

def readAlist(Alist_path):
    AREAFILE = Alist_path
    alist = []
    with open(AREAFILE,'r') as ar:
        section_lines = ar.readlines()
        for i in range(len(section_lines)):
            section_line = section_lines[i]
            section_r = section_line.strip().split(' ')
            if (i==0):
                section_num = int(section_r[0])
            if (i > 0 and i <= section_num):
                name_s = 'd' + str(section_r[0]) + 't' + str(int(float(section_r[1]) * 10))
                d = float(section_r[0]) / 1000.0
                t = float(section_r[1]) / 1000.0
                area_s = math.pi * d ** 2 / 4.0 - math.pi * (d - 2 * t) ** 2 / 4.0
                I_s = math.pi * d ** 4 / 64.0 - math.pi * ( d - 2 * t) ** 4 / 64.0
                i_s = math.sqrt(I_s/area_s)
                alist.append((float(area_s), float(I_s), float(i_s), str(name_s), float(d), float(t)))
    return alist

def save_file_stage1(OUTFILE, p, e):
    with open(OUTFILE, "w") as f:
        print(len(p), len(e), file=f)
        for i in range(len(p)):
            print(p[i].vec.x, p[i].vec.y, p[i].vec.z, p[i].supportX, p[i].supportY,
                p[i].supportZ, p[i].loadX, p[i].loadY, p[i].loadZ, file=f)
        for i in range(len(e)): print(e[i].u, e[i].v, e[i].area, e[i].d, e[i].t, file = f)

def save_file(initial_points, state, mass, path, mode = 'Area', best = False, diverse_id = None):
    r'''
    save state into txt
    :param initial_points: initial points, for support and load information
    :param state: truss state
    :param mass: mass of truss
    :param path: path to store
    :return: None
    '''
    if (best == False):
        if (diverse_id == None):
            fo = open(os.path.join(path, str(int(mass * 1000)) + ".txt"), "w")
        else: 
            fo = open(os.path.join(path, str(int(mass * 1000)) + "_" + str(diverse_id).zfill(2) + ".txt"), "w")
    else: fo = open(os.path.join(path, '_best.txt'), "w")
    n = state.num_points
    fo.write("{} {}\n".format(n, n * (n - 1) // 2))
    for i in range(n):
        x = state.nodes[i][0]
        y = state.nodes[i][1]
        if state.dimension == 2:
            z = 0.0
        else:
            z = state.nodes[i][2]
        fo.write("{} {} {} {} {} {} {} {} {}\n".format(x, y, z,
                                                       initial_points[i].supportX, initial_points[i].supportY, initial_points[i].supportZ,
                                                       initial_points[i].loadX, initial_points[i].loadY, initial_points[i].loadZ))
    if (mode == 'Area'):
        for i in range(n):
            for j in range(i):
                fo.write("{} {} {}\n".format(j, i, state.edges[i][j]))
    if (mode == 'DT'):
        for i in range(n):
            for j in range(i):
                if (state.edges[i][j][0] <= 0):
                    fo.write("{} {} {} {} {}\n".format(j, i, -1, -1, -1))
                else:
                    d = state.edges[i][j][0]
                    t = state.edges[i][j][1]
                    if (t == 0): 
                        fo.write("{} {} {} {} {}\n".format(j, i, -1, -1, -1))
                    else:
                        area = math.pi*state.edges[i][j][0]**2/4.0 - math.pi*(state.edges[i][j][0]-2*state.edges[i][j][1])**2/4.0
                        fo.write("{} {} {} {} {}\n".format(j, i, area, state.edges[i][j][0], state.edges[i][j][1]))
    fo.close()

def save_file_from_list(p, e, output_file):
    with open(output_file, "w") as f:
        print(len(p), len(e), file = f)
        for i in range(len(p)):
            print(p[i].vec.x, p[i].vec.y, p[i].vec.z, p[i].supportX, p[i].supportY, p[i].supportZ, p[i].loadX, p[i].loadY, p[i].loadZ, file = f)
        for i in range(len(e)):
            print(e[i].u, e[i].v, e[i].area, e[i].d, e[i].t, file = f)

def save_trajectory(initial_points, trajectory, mass, path):
    r'''
    save state into txt
    :param initial_points: initial points, for support and load information
    :param trajectory: history of truss states
    :param mass: mass of truss
    :param path: path to store
    :return: None
    '''
    current_dir = os.getcwd()
    dir = path + str(int(mass))
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.mkdir(dir)
    os.chdir(dir)
    for i in range(len(trajectory)):
        state = trajectory[i]

        def _save_file(initial_points, state, file_name):
            r'''
            save state into txt
            :param initial_points: initial points, for support and load information
            :param state: truss state
            :param mass: mass of truss
            :param path: path to store
            :return: None
            '''
            fo = open(file_name, "w")
            n = state.num_points
            fo.write("{} {}\n".format(n, n * (n - 1) // 2))
            for i in range(n):
                x = state.nodes[i][0]
                y = state.nodes[i][1]
                if state.dimension == 2:
                    z = 0.0
                else:
                    z = state.nodes[i][2]
                fo.write("{} {} {} {} {} {} {} {} {}\n".format(x, y, z,
                                                               initial_points[i].supportX, initial_points[i].supportY, initial_points[i].supportZ,
                                                               initial_points[i].loadX, initial_points[i].loadY, initial_points[i].loadZ))
            for i in range(n):
                for j in range(i):
                    fo.write("{} {} {}\n".format(j, i, state.edges[i][j]))
            fo.close()

        def _saveGraph(p, e, file):
            for i in range(len(p)):
                plt.scatter([p[i].vec.x], [p[i].vec.y], color='b')

            for i in range(len(e)):
                x0 = [p[e[i].u].vec.x, p[e[i].v].vec.x]
                y0 = [p[e[i].u].vec.y, p[e[i].v].vec.y]

                if e[i].area != -1:
                    plt.plot(x0, y0, color='b', linewidth=e[i].area / 0.01)

            plt.axis("equal")
            plt.savefig(file)
            plt.cla()

        _save_file(initial_points, state, str(i) + ".txt")

    os.chdir(current_dir)

def is_edge_addable(u, v, points, edges, enabled=False):
    r'''
    Check if adding a bar between u and v is valid, only applied to 2-d case

    :param u: index of one end of the edge
    :param v: index of the other end of the edge
    :param points: nodes
    :param edges: edges
    :param enabled: Whether use this function to check edge constraint, if False, always return True
    :return: bool
    '''

    max_length = 18
    minang = 10
    cosminang = np.cos(minang / 180.0 * np.pi)
    max_edges = 10

    #判断杆件是否交叉
    def _intersect(point_u1,point_v1,point_u2,point_v2): #四个点对象，其中u1v1为一根杆，u2v2为一根杆
        intersected = False

        u1=np.array([point_u1.vec.x,point_u1.vec.y])
        v1=np.array([point_v1.vec.x,point_v1.vec.y])
        u2=np.array([point_u2.vec.x,point_u2.vec.y])
        v2=np.array([point_v2.vec.x,point_v2.vec.y])      #取得四个点坐标向量

        u1v1=v1-u1
        u2v2=v2-u2     #杆件向量

        u1u2=u2-u1
        u1v2=v2-u1

        u2u1=u1-u2
        u2v1=v1-u2

        def compare(a,b):
            if((a[0] < b[0]) or (a[0] == b[0] and a[1] < b[1])):
                return -1
            elif(a[0] == b[0] and a[1] == b[1]):
                return 0
            else:
                return 1
        #对一条线段的两端点进行排序，横坐标大的点更大，横坐标相同，纵坐标大的点更大，升序排序
        po=[u1,v1,u2,v2]
        if compare(po[0],po[1])>0:
            temp=po[0]
            po[0]=po[1]
            po[1]=temp
        if compare(po[2],po[3])>0:
            temp=po[2]
            po[2]=po[3]
            po[3]=temp

        #考虑一般情况
        if  ((np.cross(u1v1,u1u2)*np.cross(u1v1,u1v2)<0 and np.cross(u2v2,u2u1)*np.cross(u2v2,u2v1)<0) or    #叉积均小于0，跨越交叉
            (np.cross(u1v1,u1u2)*np.cross(u1v1,u1v2)==0 and np.cross(u2v2,u2u1)*np.cross(u2v2,u2v1)<0) or    #任意一方=0， 另一方<0，为一节点位于另一杆件上
            (np.cross(u1v1,u1u2)*np.cross(u1v1,u1v2)<0 and np.cross(u2v2,u2u1)*np.cross(u2v2,u2v1)==0)):     #顺便排除了有公共点的情况，有公共点两方均为0
            intersected = True

        #考虑如果两线段共线重叠
        if np.cross(u1v1,u2v2)==0 and np.cross(u1v1,u1v2)==0: #两线段共线
            if(compare(po[0],po[2]) <= 0 and compare(po[1],po[2]) > 0):     #第一条起点小于第二条起点，第一条终点大于第二条起点
                intersected = True
            elif(compare(po[2],po[0]) <= 0 and compare(po[3],po[0]) > 0):   #第二条起点小于第一条起点，第二条终点大于第一条起点
                intersected = True

        return intersected

    def _transintersect(
        u1,v1,u2,v2,
        points,
    ): # ?
        if (
            _intersect(
                points[u1], points[v1], points[u2], points[v2]
            )
        ):
            return True

        if (u1 == u2):
            if (
                getang(
                    points[v1].vec - points[u1].vec,
                    points[v2].vec - points[u2].vec,
                ) > cosminang
            ):
                return True
        if (u1 == v2):
            if (
                getang(
                    points[v1].vec - points[u1].vec,
                    points[u2].vec - points[v2].vec,
                ) > cosminang
            ):
                return True
        if (v1 == u2):
            if (
                getang(
                    points[u1].vec - points[v1].vec,
                    points[v2].vec - points[u2].vec,
                ) > cosminang
            ):
                return True
        if (v1 == v2):
            if (
                getang(
                    points[u1].vec - points[v1].vec,
                    points[u2].vec - points[v2].vec,
                ) > cosminang
            ):
                return True

        return False

    def _is_too_long(point_u, point_v):
        return getlen2(point_u, point_v) > max_length

    # MODIFICATION: not considering EDGE_CONFIG

    if not enabled:
        return True

    if _is_too_long(points[u], points[v]):
        return False

    if points[u].isSupport and points[v].isSupport:
        return False

    for edge in edges.values():
        if (
            _transintersect(
                u, v, edge.u, edge.v, points
            )
        ):
            return False

    return True

def getuv(x):
    x += 1
    v = math.ceil(
        (math.sqrt(1 + 8 * x) - 1) / 2.0
    )
    u = x - v * (v - 1) // 2 - 1
    return u, v

def similar_position(p1, e1, p2, e2):
    pts1 = []
    pts2 = []
    for p in p1:
        pts1.append([p.vec.x, p.vec.y, p.vec.z])
    for p in p2:
        pts2.append([p.vec.x, p.vec.y, p.vec.z])
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

def similar_topo(p1, e1, p2, e2):
    pts1 = []
    pts2 = []
    for i in range(len(p1)): pts1.append(i)
    for i in range(len(p2)): pts2.append(i)
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

def closestDistanceBetweenLines(a0, a1, b0, b1, 
        clampAll = False, clampA0 = False,clampA1 = False,clampB0 = False,clampB1 = False):
        r''' 
        Given two lines defined by numpy.array pairs (a0,a1,b0,b1)
        Return the closest points on each segment and their distance
        '''
        # If clampAll=True, set all clamps to True
        if clampAll:
            clampA0=True
            clampA1=True
            clampB0=True
            clampB1=True
        # Calculate denomitator
        A = a1 - a0
        B = b1 - b0
        magA = np.linalg.norm(A)
        magB = np.linalg.norm(B)
    
        _A = A / magA
        _B = B / magB
    
        cross = np.cross(_A, _B)
        denom = np.linalg.norm(cross) ** 2
    
        # If lines are parallel (denom=0) test if lines overlap.
        # If they don't overlap then there is a closest point solution.
        # If they do overlap, there are infinite closest positions, but there is a closest distance
        if not denom:
            d0 = np.dot(_A, (b0 - a0))
        
            # Overlap only possible with clamping
            if clampA0 or clampA1 or clampB0 or clampB1:
                d1 = np.dot(_A, (b1 - a0))
            
                # Is segment B before A?
                if d0 <= 0 >= d1:
                    if clampA0 and clampB1:
                        if np.absolute(d0) < np.absolute(d1):
                            return a0,b0,np.linalg.norm(a0-b0)
                        return a0,b1,np.linalg.norm(a0-b1)
                
                # Is segment B after A?
                elif d0 >= magA <= d1:
                    if clampA1 and clampB0:
                        if np.absolute(d0) < np.absolute(d1):
                            return a1,b0,np.linalg.norm(a1-b0)
                        return a1,b1,np.linalg.norm(a1-b1)
                
            # Segments overlap, return distance between parallel segments
            return None,None,np.linalg.norm(((d0*_A)+a0)-b0)
    
        # Lines criss-cross: Calculate the projected closest points
        t = (b0 - a0)
        detA = np.linalg.det([t, _B, cross])
        detB = np.linalg.det([t, _A, cross])

        t0 = detA/denom
        t1 = detB/denom

        pA = a0 + (_A * t0) # Projected closest point on segment A
        pB = b0 + (_B * t1) # Projected closest point on segment B


        # Clamp projections
        if clampA0 or clampA1 or clampB0 or clampB1:
            if clampA0 and t0 < 0:
                pA = a0
            elif clampA1 and t0 > magA:
                pA = a1
        
            if clampB0 and t1 < 0:
                pB = b0
            elif clampB1 and t1 > magB:
                pB = b1
            
            # Clamp projection A
            if (clampA0 and t0 < 0) or (clampA1 and t0 > magA):
                dot = np.dot(_B,(pA-b0))
                if clampB0 and dot < 0:
                    dot = 0
                elif clampB1 and dot > magB:
                    dot = magB
                pB = b0 + (_B * dot)
    
            # Clamp projection B
            if (clampB0 and t1 < 0) or (clampB1 and t1 > magB):
                dot = np.dot(_A,(pB-a0))
                if clampA0 and dot < 0:
                    dot = 0
                elif clampA1 and dot > magA:
                    dot = magA
                pA = a0 + (_A * dot)

        return pA, pB, np.linalg.norm(pA - pB)