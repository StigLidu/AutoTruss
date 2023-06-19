import numpy as np
import math
import os, sys
base_path = os.getcwd()
sys.path.append(base_path)
import openseespy.opensees as op
import matplotlib.pyplot as plt
from utils.utils import readFile

class DynamicModel:
    #构造函数
    def __init__(self,
                dimension,
                E=193*10**9,          #N/m2
                pho=8.0*10**3,        #kg/m3
                sigma_T=123*10**6,    #N/m2
                sigma_C=213*10**6,    #N/m2
                dislimit=0.002,       #m
                slenderness_ratio_T=220,
                slenderness_ratio_C=180,
                max_len=5,            #m
                min_len=0.03,         #m
                use_self_weight=True,
                use_dis_constraint=True,
                use_stress_constraint=True,
                use_buckle_constraint=True,
                use_slenderness_constraint=True,
                use_longer_constraint=True,
                use_shorter_constraint=True
                ): 
        self._dimension = dimension    #结构维度
        self._E = E                    #弹性模量
        self._pho = pho                #材料密度
        self._sigma_T = sigma_T        #容许拉应力
        self._sigma_C = sigma_C        #容许压应力
        self._limit_dis = dislimit     #容许位移
        self.slenderness_ratio_T=slenderness_ratio_T #容许受拉长细比
        self.slenderness_ratio_C=slenderness_ratio_C #容许受压长细比
        self.max_len=max_len                         #最大长度
        self.min_len=min_len                         #最小长度
        self._use_self_weight = use_self_weight              #是否计算自重
        self._use_dis_constraint = use_dis_constraint        #是否启用位移约束
        self._use_stress_constraint = use_stress_constraint         #是否启用应力约束
        self._use_buckle_constraint = use_buckle_constraint         #是否启用屈曲约束
        self._use_slenderness_constraint=use_slenderness_constraint #是否启用长细比约束
        self._use_longer_constraint=use_longer_constraint           #是否启用超长约束
        self._use_shorter_constraint=use_shorter_constraint         #是否启用过短约束

    #判定结构的几何不变性+分析计算
    def _is_struct(self, points, edges):
        ########计算自由度初判结构几何不变性##########
        total_support = 0             #保存支座约束数
        for p in points.values():
            if self._dimension == 2:  #平面桁架
                total_support += (
                    p.supportX
                    + p.supportY
                )
            else:                     #空间桁架
                total_support += (
                    p.supportX
                    + p.supportY
                    + p.supportZ
                )

        if len(points) * self._dimension - len(edges) - total_support > 0:
            return (False)   #计算自由度>0，结构不稳定直接返回False
        
        #######以下基于点和边集建立有限元模形分析########
        op.wipe()   # 清除所有已有结构
        op.model('basic', '-ndm', self._dimension, '-ndf', self._dimension)  #设置建模器

        for i, point in points.items():   #建立节点
            if self._dimension == 2:
                op.node(
                    i,
                    point.vec.x,
                    point.vec.y,
                )
            else:
                op.node(
                    i,
                    point.vec.x,
                    point.vec.y,
                    point.vec.z,
                )

        for i, point in points.items():  #施加节点支座约束
            if point.isSupport:
                if self._dimension == 2:
                    op.fix(
                        i,
                        point.supportX,
                        point.supportY,
                    )
                else:
                    op.fix(
                        i,
                        point.supportX,
                        point.supportY,
                        point.supportZ,
                    )

        op.timeSeries("Linear", 1)
        op.pattern("Plain", 1, 1)
        
        for i, point in points.items():  #添加节点荷载
            if point.isLoad:
                if self._dimension == 2:
                    op.load(
                        i,
                        point.loadX,
                        point.loadY,
                    )
                else:
                    op.load(
                        i,
                        point.loadX,
                        point.loadY,
                        point.loadZ,
                    )

        op.uniaxialMaterial("Elastic", 1, self._E)   #定义材料

        for i, edge in enumerate(edges.values()):
            op.element("Truss", i, edge.u, edge.v, edge.area, 1)  #赋予杆件截面属性
        
        
        
        if self._use_self_weight:  #如果计算自重时
            gravity = 9.8   #重力加速度
            load_gravity = [0 for _ in range(len(points))]  #初始化了一个load_gravity列表，表征杆件自重的等效结点力，len(points)个元素均为0

            for i, edge in edges.items():
                edge_mass = edge.len * edge.area * self._pho       #每根杆件质量
                load_gravity[edge.u] += edge_mass * gravity * 0.5
                load_gravity[edge.v] += edge_mass * gravity * 0.5  #每根杆件的重力向两端分一半到节点上
            
            for i in range(len(points)):        #将重力荷载等效施加于节点上
                if self._dimension == 2:        #如果是平面结构
                    op.load(i, 0.0, -1 * load_gravity[i])
                else:                           #如果是空间结构
                    op.load(i, 0.0, 0.0, -1 * load_gravity[i])	
        
        
        op.system("BandSPD")
        op.numberer("RCM")
        op.constraints("Plain")
        op.integrator("LoadControl", 1.0)
        op.algorithm("Newton")
        op.analysis("Static")
        ok = op.analyze(1)  #运行分析，ok表征是否成功运行，返回0代表成功，返回<0失败。（注:这里对结构的几何不变性进行了充分判断）
        if ok < 0:
            ok = False
        else:
            ok = True
        return ok
    
    #评估节点位移
    def _get_dis_value(self, points):
        displacement_weight = np.zeros((len(points), 1))  #初始化一个0数组，用于存放位移数据
        for i in range(len(points)):
            if self._dimension == 2:
                weight = max(   
                    abs(op.nodeDisp(i, 1)),
                    abs(op.nodeDisp(i, 2)),
                )                                  #只考虑x,y,(z)方向上的最大的一个线位移
            else:
                weight = max(
                    abs(op.nodeDisp(i, 1)),
                    abs(op.nodeDisp(i, 2)),
                    abs(op.nodeDisp(i, 3)),
                )
            print("第{:}结点位移为{:}mm".format(i,weight*10**3))
            displacement_weight[i] = max(weight / self._limit_dis - 1, 0)  #判定节点位移是否超限：超出则比例存入数组，否则记为0，最后累加数组中的数值作为位移评估参照
        return displacement_weight
    
    #评估杆件应力
    def _get_stress_value(self, edges):
        stress_weight = np.zeros((len(edges), 1))  #初始化一个0数组，用于存放应力数据

        for tag, i in enumerate(edges.keys()):

            edges[i].force = op.basicForce(tag)   #从有限元得到杆件轴力
            edges[i].stress = edges[i].force[0] / edges[i].area  #根据轴力、截面积求正应力
            print("第{:}杆件应力为{:}MPa".format(i,edges[i].stress*10**(-6)))
            if edges[i].stress < 0:                                  #压杆
                stress_weight[tag] = max(
                    abs(edges[i].stress) / self._sigma_C - 1.0,
                    0.0
                )
            else:                                                    #拉杆
                stress_weight[tag] = max(
                    abs(edges[i].stress) / self._sigma_T - 1.0,
                    0.0
                )
        return stress_weight  #判定节点应力是否超限：超出则比例存入数组，否则记为0，最后累加数组中的数值作为应力评估参照

    #评估杆件屈曲
    def _get_buckle_value(self, edges):
        buckle_weight = np.zeros((len(edges), 1))  #初始化一个0数组，用于存放屈曲数据
        miu_buckle = 1.0                           #杆件计算长度系数，桁架两端铰接取1
        
        for tag, i in enumerate(edges.keys()):
            edges[i].force = op.basicForce(tag)    #存放轴力数据
            edges[i].stress = edges[i].force[0] / edges[i].area #根据轴力、截面积求正应力
            
            if edges[i].stress < 0:    #仅压杆才考虑屈曲
                #计算欧拉临界力
                force_cr = (
                    math.pi ** 2 
                    * self._E * edges[i].inertia
                ) / (miu_buckle * edges[i].len) ** 2

                #计算欧拉临界应力
                buckle_stress_max = force_cr / edges[i].area

                buckle_weight[tag] = max(
                    abs(edges[i].stress) / abs(buckle_stress_max) - 1.0,
                    0.0
                )#判定杆件压应力是否超过屈曲临界应力：超出则比例存入数组，否则记为0
        return buckle_weight
    
    #评估杆件长细比
    def _get_slenderness_ratio(self, edges):
        lambda_weight = np.zeros((len(edges), 1))  #初始化一个0数组，用于存放长细比数据
        
        for tag, i in enumerate(edges.keys()):
            edges[i].force = op.basicForce(tag)    #存放轴力数据
            edges[i].stress = edges[i].force[0] / edges[i].area #根据轴力、截面积求正应力
            print(edges[i].len, edges[i].inertia, edges[i].area)
            lambda_weight[tag] = max(
                # self.len/(self.inertia/self.area)**0.5
                abs(edges[i].len / (edges[i].inertia / edges[i].area) ** 0.5) / abs(self.slenderness_ratio_C if edges[i].stress < 0 else self.slenderness_ratio_T) - 1.0,
                0.0
            )#判定杆件长细比是否超过限制：超出则比例存入数组，否则记为0
        return lambda_weight

    #评估杆件超长
    def _get_length_longer(self, edges):
        longer_weight = np.zeros((len(edges), 1))  #初始化一个0数组，用于存放长细比数据
        
        for tag, i in enumerate(edges.keys()):   
            longer_weight[tag] = max(
                abs(edges[i].len) / abs(self.max_len) - 1.0,
                0.0
            )#判定杆件长度是否超过限制：超出则比例存入数组，否则记为0
        return longer_weight

    #评估杆件超长
    def _get_length_shorter(self, edges):
        shorter_weight = np.zeros((len(edges), 1))  #初始化一个0数组，用于存放长细比数据
        
        for tag, i in enumerate(edges.keys()):   
            if edges[i].len<self.min_len:
                shorter_weight[tag] = 1.0-edges[i].len / self.min_len
                #判定杆件长度是否过短：超出则比例存入数组，否则记为0
        return shorter_weight

    #调用以上函数运行结构分析
    def run(self, points, edges):
        
        is_struct = self._is_struct(points, edges) #运行结构建模与分析，is_struct返回结构是否正常完成分析
        mass, dis_value, stress_value, buckle_value, slenderness_vlaue, longer_value, shorter_value= 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        if is_struct: #如果结构成功完成分析，即结构是几何不变的
            for i, edge in edges.items():
                mass += edge.len * edge.area * self._pho      #计算结构总质量
            
            if self._use_dis_constraint:
                dis_value = self._get_dis_value(points)       #若启用，获取结构位移评估结果
            if self._use_stress_constraint:
                stress_value = self._get_stress_value(edges)  #若启用，获取结构应力评估结果
            if self._use_buckle_constraint:
                buckle_value = self._get_buckle_value(edges)  #若启用，获取结构屈曲评估结果
            if self._use_slenderness_constraint:
                slenderness_vlaue = self._get_slenderness_ratio(edges)  #若启用，获取结构长细比评估结果    
            if self._use_longer_constraint:
                longer_value = self._get_length_longer(edges)  #若启用，获取结构长细比评估结果
            if self._use_shorter_constraint:
                shorter_value = self._get_length_shorter(edges)  #若启用，获取结构长细比评估结果 

        return (
            is_struct, mass, dis_value, stress_value, buckle_value, slenderness_vlaue, longer_value, shorter_value
        )

    #绘制平面桁架
    def render(self, points, edges):
        _ax = plt.axes(projection='3d')
        for point in points.values():   #绘制节点，scatter()绘制散点
            if point.isSupport:
                _ax.scatter([point.vec.x], [point.vec.y],[point.vec.z], color='g') #支座点为绿色
            elif point.isLoad:
                _ax.scatter([point.vec.x], [point.vec.y],[point.vec.z], color='r') #荷载作用的节点为红色
            else:
                _ax.scatter([point.vec.x], [point.vec.y],[point.vec.z], color='b') #其余节点蓝色

        for edge in edges.values():    #绘制杆件
            x0 = [points[edge.u].vec.x, points[edge.v].vec.x]   #杆件起点
            y0 = [points[edge.u].vec.y, points[edge.v].vec.y]   #杆件终点
            z0 = [points[edge.u].vec.z, points[edge.v].vec.z]   #杆件起点
            
            if edge.stress < -1e-7:
                _ax.plot(x0, y0, z0, color='g', linewidth=(edge.area / math.pi)**0.5*500)    #压杆绿色
            elif edge.stress > 1e-7:
                _ax.plot(x0, y0, z0, color='r', linewidth=(edge.area / math.pi)**0.5*500)    #拉杆红色
            else:
                _ax.plot(x0, y0, z0, color='k', linewidth=(edge.area / math.pi)**0.5*500)    #零杆黑色
        plt.show() #显示图像


if __name__=='__main__':
    truss=DynamicModel(3)    #创建结构对象
    point_list, edge_list = readFile("input_file_3d.txt") #读取数据输入文件中的预设点和边

    #将point_list, edge_list转换成truss.run的数据结构
    points = {}
    edges = {}
    for i, point in enumerate(point_list):  #将预设点对象加入点集
        points[i] = point

    for i, edge in enumerate(edge_list):    #将预设边对象加入边集
        if edge.u > edge.v:                 #边端点重新编号
            tmp = edge.u
            edge.u = edge.v
            edge.v = tmp
        edges[(edge.u, edge.v)] = edge        

    #运行模型分析                                 
    is_struct, mass, dis_value, stress_value, buckle_value, slenderness_vlaue, longer_value, shorter_value  = truss.run(points, edges) 
    
    #后处理，输出结构设计结果的提示信息
    print(np.sum(dis_value), np.sum(stress_value), np.sum(buckle_value), np.sum(slenderness_vlaue), np.sum(longer_value), np.sum(shorter_value))
    if not is_struct:
        print("结构几何不稳定")
    elif np.sum(dis_value) > 0.0 or np.sum(stress_value) > 0.0 or np.sum(buckle_value) > 0.0 or np.sum(slenderness_vlaue) or np.sum(longer_value) or np.sum(shorter_value):
        for i in range(len(dis_value)):
            if dis_value[i]>0.0:
                print("第{:}结点位移超出限值{:}%".format(i,dis_value[i]*100))

        for i in range(len(stress_value)):
            if stress_value[i]>0.0:
                print("第{:}杆件应力超出限值{:}%".format(i,stress_value[i]*100))

        for i in range(len(buckle_value)):
            if buckle_value[i]>0.0:
                print("第{:}杆件屈曲应力超出限值{:}%".format(i,buckle_value[i]*100))

        for i in range(len(slenderness_vlaue)):
            if slenderness_vlaue[i][0]>0.0:
                print("第{:}杆件长细比超出限值{:}%".format(i,slenderness_vlaue[i]*100))

        for i in range(len(longer_value)):
            if longer_value[i]>0.0:
                print("第{:}杆件长度超出限值{:}%".format(i,longer_value[i]*100))

        for i in range(len(shorter_value)):
            if shorter_value[i]>0.0:
                print("第{:}杆件长度短过限值{:}%".format(i,shorter_value[i]*100))                 

    else:
        print("结构几何稳定，且所有约束满足。当前结构总质量为：{:.3f}kg".format(mass))

    truss.render(points, edges) #显示桁架图像


   
