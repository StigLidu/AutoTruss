import math

#定义空间向量类及其运算
class Vector3:
	#构造函数
	def __init__(self, x = 0.0, y = 0.0, z = 0.0):
		self.x = float(x)
		self.y = float(y)
		self.z = float(z)   #空间坐标

	#向量加运算
	def __add__(self, obj):
		return Vector3(self.x + obj.x, self.y + obj.y, self.z + obj.z)

	#向量减运算
	def __sub__(self, obj):
		return Vector3(self.x - obj.x, self.y - obj.y, self.z - obj.z)

	#向量数乘及叉乘
	def __mul__(self, obj):
		if (type(obj) == Vector3):
			return Vector3(self.y*obj.z-self.z*obj.y, self.z*obj.x-self.x*obj.z, self.x*obj.y-self.y*obj.x)#向量叉乘
		if (type(obj) == float or type(obj) == int):
			return Vector3(self.x * obj, self.y * obj, self.z * obj)#数乘向量
		assert(False)
	
	#向量文本化表示(x,y,z)
	def __str__(self):
		return str('(' + str(self.x) + ', ' + str(self.y) + ', ' + str(self.z) + ')')

	#向量模长的平方
	def length2(self):
		return float(self.x * self.x + self.y * self.y + self.z * self.z)

	#向量模长
	def length(self):
		return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

	#向量单位化
	def norm(self):
		l = self.length()
		return Vector3(self.x / l, self.y / l, self.z / l)

	#判断向量是否相等
	def __eq__(self, other):
		assert(type(other) == Vector3)
		if (abs(self.x - other.x) < 1e-8 and abs(self.y - other.y) < 1e-8 and abs(self.z - other.z) < 1e-8):
			return True
		else:
			return False

#节点类
class Point:
	#构造函数
	def __init__(self, vec = Vector3(), supportX = 0, supportY = 0, supportZ = 0, loadX = 0.0, loadY = 0.0, loadZ = 0.0):
		self.vec = vec  #用空间向量类创建位置
		
		self.supportX = supportX       #表征节点某方向支座约束，1是0否
		self.supportY = supportY
		self.supportZ = supportZ
		self.isSupport = False
		if (supportX == 1 or supportY == 1 or supportZ == 1):
			self.isSupport = True
		
		self.loadX = loadX            #表征节点某方向的点荷载的大小
		self.loadY = loadY
		self.loadZ = loadZ
		self.isLoad = False
		if (abs(loadX) > 1e-7 or abs(loadY) > 1e-7 or abs(loadZ) > 1e-7):
			self.isLoad = True

#杆件类
class Bar:
	#构造函数
	def __init__(self, u = -1, v = -1, area = 1.0, leng = 0.0, inertia=1.0):

		self.u = int(u)
		self.v = int(v)             #杆件两端的节点编号
		self.area = float(area)     #杆件截面积
		self.force = float(0.0)     #杆件轴力
		self.len = leng             #杆件长度
		self.stress = 0.0           #杆件应力
		self.inertia=float(inertia) #杆件惯性矩
		self.slenderness_ratio=self.len/(self.inertia/self.area)**0.5
		
#求空间向量的模长
def getlen(vec):
	return math.sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z)

#求两点间杆件长度
def getlen2(u, v):
	return getlen(u.vec - v.vec)

#两向量夹角的余弦
def getang(vec1, vec2):
	return (vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z) / (getlen(vec1) * getlen(vec2))

#从文件读取设计数据
def readFile(input_file):

	p = []  #点集
	e = []  #边集

	with open(input_file, "r") as fle:
		lines = fle.readlines()        #读取文件全部行，返回一个字符串列表，每个元素为文件的一行内容
		for i in range(len(lines)):
			line = lines[i]            #第i行内容
			vec = line.strip().split(' ')  #strip()用于移除字符串头尾指定的字符（默认为空格）或字符序列，split(' ')通过指定分隔符对字符串进行切片，sep默认为所有的空字符，包括空格、换行(\n)、制表符(\t)
			
			if (i == 0):               #第一行
				vn = int(vec[0])       #预设节点数量
				en = int(vec[1])       #预设边数量
				continue

			if (1 <= i and i <= vn):   #预设节点信息
				p.append(Point(Vector3(float(vec[0]), float(vec[1]), float(vec[2])), int(vec[3]), int(vec[4]), int(vec[5]), float(vec[6]), float(vec[7]), float(vec[8])))
				continue               #点集p里每个元素为Point类对象：{点坐标3+支座约束+荷载}

			if (vn + 1 <= i and i <= vn + en):
				e.append(Bar(vec[0], vec[1], float(vec[2]), getlen2(p[int(vec[0])], p[int(vec[1])]),math.pi*(float(vec[3])**4-(float(vec[3])-2*float(vec[4]))**4)/64))
				continue               #边集里每个元素为bar类对象：{两端节点编号2+截面积+长度+(惯性矩缺省)}
	return p, e
	