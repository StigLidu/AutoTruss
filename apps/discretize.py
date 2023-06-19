import os
import sys
import math
base_path = os.getcwd()
sys.path.append(base_path)
from utils.utils import *
from configs.config import *
parser = get_base_config()
args = parser.parse_known_args(sys.argv[1:])[0]
from truss_envs.reward import *
Envs_init(args)
if __name__ == '__main__':
    p = []
    e = []
    alist = []
    AREAFILE = args.Alist_path
    FILENAME = args.check_file
    
    with open(FILENAME, "r") as fle:
        lines = fle.readlines()
        for i in range(len(lines)):
            line = lines[i]
            vec = line.strip().split(' ')
            if (i == 0):
                vn = int(vec[0])
                en = int(vec[1])
            if (1 <= i and i <= vn):
                p.append(Point(Vector3(float(vec[0]), float(vec[1]), float(vec[2])), int(vec[3]), int(vec[4]), int(vec[5]), float(vec[6]), float(vec[7]), float(vec[8])))            
            if (vn + 1 <= i and i <= vn + en):
                if (len(vec) > 3):
                    d = float(vec[3])
                    t = float(vec[4])
                else:
                    d = None
                    t = None
                if (float(vec[2]) < 0): continue
                e.append(Bar(vec[0], vec[1], float(vec[2]), getlen2(p[int(vec[0])], p[int(vec[1])]), d = d, t = t))

    with open(AREAFILE,'r') as ar:
        section_lines = ar.readlines()
        for i in range(len(section_lines)):
            section_line = section_lines[i]
            section_r = section_line.strip().split(' ')
            if (i==0):
                section_num = int(section_r[0])
                continue
            if (i > 0 and i <= section_num):
                name_s = 'd'+str(section_r[0])+'t'+str(int(float(section_r[1])*10))
                d = float(section_r[0])/1000.0
                t = float(section_r[1])/1000.0
                area_s =  math.pi*d**2/4.0 - math.pi*(d-2*t)**2/4.0
                I_s = math.pi*d**4/64.0 - math.pi*(d-2*t)**4/64.0
                i_s = math.sqrt(I_s/area_s)
                alist.append((float(area_s), float(I_s), float(i_s), str(name_s), float(d), float(t)))
    
    print(reward_fun(p, e))

    for j in range(5):
      for i in range(len(e)):
        minaera = 1e9
        for d_edge in alist:
            e[i] = Bar(e[i].u, e[i].v, leng = getlen2(p[e[i].u], p[e[i].v]), area=d_edge[0], I_s = d_edge[1], i_s = d_edge[2], name_s = d_edge[3], d = d_edge[4], t = d_edge[5])
            if (reward_fun(p, e)[0] > 0 and d_edge[0] < minaera):
                final_edge = d_edge
                minaera = d_edge[0]
        if (minaera == 1e9):
            print('no valid truss')
        e[i] = Bar(e[i].u, e[i].v, leng = getlen2(p[e[i].u], p[e[i].v]), area=final_edge[0], I_s = final_edge[1], i_s = final_edge[2], name_s = final_edge[3], d = final_edge[4], t = final_edge[5])

    print(reward_fun(p, e))

    OUTFILE = 'D' + str(reward_fun(p, e)[0]) + '.txt'
    with open(OUTFILE, "w") as f:
        print(len(p), len(e), file=f)
        for i in range(len(p)):
            print(p[i].vec.x, p[i].vec.y, p[i].vec.z, p[i].supportX, p[i].supportY,
                p[i].supportZ, p[i].loadX, p[i].loadY, p[i].loadZ, file=f)

        for i in range(len(e)):
            print(e[i].u, e[i].v, e[i].area, e[i].d, e[i].t, file=f)
