import matplotlib.pyplot as plt
import math, os


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
		return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

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

	def __init__(self, vec=Vector3(), supportX=0, supportY=0, supportZ=1, loadX=0.0, loadY=0.0, loadZ=0.0):
		self.vec = vec

		self.supportX = supportX
		self.supportY = supportY
		self.supportZ = supportZ
		self.isSupport = False
		# 2D
		if (supportX == 1 or supportY == 1):
			self.isSupport = True
		# #3D
		# if (supportX == 1 or supportY == 1 or supportZ == 1):
		# 	self.isSupport = True
		self.loadX = loadX
		self.loadY = loadY
		self.loadZ = loadZ
		self.isLoad = False
		if (abs(loadX) > 1e-7 or abs(loadY) > 1e-7 or abs(loadZ) > 1e-7):
			self.isLoad = True


class Bar:

	def __init__(self, u=-1, v=-1, area=1.0, leng=0.0, inertia=1.0):
		self.u = int(u)
		self.v = int(v)
		self.area = float(area)
		self.force = float(0.0)
		self.len = leng
		self.stress = 0.0
		self.inertia = float(inertia)


class Load:

	def __init__(self, u=-1, fx=0.0, fy=0.0, fz=0.0):
		self.u = int(u)
		self.fx = float(fx)
		self.fy = float(fy)
		self.fz = float(fz)


def getlen2(u, v):
	return math.sqrt((u.vec.x-v.vec.x)**2+(u.vec.y-v.vec.y)**2+(u.vec.z-v.vec.z)**2)


def readFile(FILENAME):
	p = []
	e = []
	pload = []

	with open(FILENAME, "r") as fle:
		lines = fle.readlines()
		for i in range(len(lines)):
			line = lines[i]
			vec = line.strip().split(' ')
			if (i == 0):
				vn = int(vec[0])
				en = int(vec[1])
				continue

			if (1 <= i and i <= vn):
				p.append(Point(Vector3(float(vec[0]), float(vec[1]), float(vec[2])), int(vec[3]), int(vec[4]), int(vec[5]), float(vec[6]), float(vec[7]), float(vec[8])))
				pload.append(Load(i-1, float(vec[6]), float(vec[7]), float(vec[8])))
				continue

			if (vn + 1 <= i and i <= vn + en):
				e.append(Bar(vec[0], vec[1], float(vec[2]), getlen2(p[int(vec[0])], p[int(vec[1])])))
				continue

	return p, e, pload


def saveGraph(p, e):
	for i in range(len(p)):
		plt.scatter([p[i].vec.x], [p[i].vec.y], color='b')

	for i in range(len(e)):
		x0 = [p[e[i].u].vec.x, p[e[i].v].vec.x]
		y0 = [p[e[i].u].vec.y, p[e[i].v].vec.y]

		# plt.scatter(x0, y0, color='b')
		# plt.text((x0[0] + x0[1]) / 2, (y0[0] + y0[1]) / 2, '%.3f'%e[i]['len'], ha = 'center',va = 'bottom',fontsize=7)
		# plt.plot(x0, y0, color='g')
		#if (e[i].stress < 0):
		#	plt.plot(x0, y0, color='g', linewidth=e[i].area / 0.01)
		#else:
		#	plt.plot(x0, y0, color='r', linewidth=e[i].area / 0.01)
		if e[i].area != -1:
			plt.plot(x0, y0, color='b', linewidth=e[i].area / 0.01)

	#plt.figure()
	plt.axis("equal")
	# FILENAME = ".\\" + str(len(p)) + "p_case1_" + str(reward) + ".jpg"
	inputname = FILENAME.replace(".txt", "")
	FILENAME_jpg = inputname + ".jpg"
	# FILENAME = ".\\" + str(len(p)) + "p_case1_" + str(round(reward, 2)) + ".jpg"

	plt.savefig(FILENAME_jpg, dpi=1000)
	plt.clf()


if __name__ == '__main__':
	#FILENAME = '../results/20211107/generate-tg-reward_v2-keep_policy-brute_force-nn-s1ts100-s2ts100-area0.0001x0.02-cdl37x17-1107_mp_7_cdl_37_17/stage_1_best_2822_obs.txt'
	#FILENAME = '../thu_wsy/truss_refine-master_v2/best_results/2139/2139.txt'
	#FILENAME = './MasterTransformerEmbedding_v1/ExperiementResult/Max9p_2/2523997.txt'
	#FILENAME = './AllExperiment/AlphaTrussStage1/6p_2305.txt'
	#FILENAME = './Stage1/PostResults/Eval_noise/9p_1/'

	#p, e, pload = readFile(FILENAME)
	#saveGraph(p, e)


	FILEfolder = './AllExperiment/Noise/'
	filelist = os.listdir(FILEfolder)
	for file in filelist:
		if file[-4:] == '.txt':
			print(file)
			FILENAME = FILEfolder + file
			p, e, pload = readFile(FILENAME)
			saveGraph(p, e)
