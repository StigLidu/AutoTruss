import gym
import time
import json
import numpy as np
import math
import random
import copy

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import warnings
import os, sys, contextlib, platform

sysstr = platform.system()
if sysstr == "Windows" or sysstr == "Linux":
    isMac = False
else:
    isMac = True

if isMac:
    import openseespymac.opensees as op
else:
    import openseespy.opensees as op

from utils.utils import readFile, Point, Bar, getlen2, Vector3, getang


def blockPrint():
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

class DynamicModel:

    def __init__(self,
                 dimension,
                 E=6.895 * 10 ** 10,
                 pho=2.76799 * 10 ** 3,
                 sigma_T=172.3 * 10 ** 6,
                 sigma_C=172.3 * 10 ** 6,
                 dislimit=0.0508,
                 ratio_ring=0.0,
                 use_self_weight=True,
                 use_dis_constraint=True,
                 use_stress_constraint=True,
                 use_buckle_constraint=True,
                 ):
        self._dimension = dimension
        self._E = E
        self._pho = pho
        self._sigma_T = sigma_T
        self._sigma_C = sigma_C
        self._limit_dis = dislimit
        self._ratio_ring = ratio_ring
        self._use_self_weight = use_self_weight
        self._use_dis_constraint = use_dis_constraint
        self._use_stress_constraint = use_stress_constraint
        self._use_buckle_constraint = use_buckle_constraint

    def _is_struct(self, points, edges):
        total_support = 0
        for p in points.values():
            if self._dimension == 2:
                total_support += (
                        p.supportX
                        + p.supportY
                )
            else:
                total_support += (
                        p.supportX
                        + p.supportY
                        + p.supportZ
                )

        if len(points) * self._dimension - len(edges) - total_support > 0:
            return (False)  # 计算自由度>0，结构不稳定直接返回False
        blockPrint()
        op.wipe()
        op.model('basic', '-ndm', self._dimension, '-ndf', self._dimension)

        for i, point in points.items():
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
        for i, point in points.items():
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
        op.uniaxialMaterial("Elastic", 1, self._E)
        for i, edge in enumerate(edges.values()):
            op.element("Truss", i, edge.u, edge.v, edge.area, 1)
        op.timeSeries("Linear", 1)
        op.pattern("Plain", 1, 1)
        for i, point in points.items():
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
        if self._use_self_weight:
            gravity = 9.8
            load_gravity = [0 for _ in range(len(points))]

            for i, edge in edges.items():
                edge_mass = edge.len * edge.area * self._pho
                load_gravity[edge.u] += edge_mass * gravity * 0.5
                load_gravity[edge.v] += edge_mass * gravity * 0.5

            for i in range(len(points)):
                if self._dimension == 2:  # 如果是平面结构
                    op.load(i, 0.0, -1 * load_gravity[i])
                else:  # 如果是空间结构
                    op.load(i, 0.0, 0.0, -1 * load_gravity[i])

        op.system("BandSPD")
        op.numberer("RCM")
        op.constraints("Plain")
        op.integrator("LoadControl", 1.0)
        op.algorithm("Newton")
        op.analysis("Static")
        ok = op.analyze(1)
        enablePrint()
        if ok < 0:
            ok = False
        else:
            ok = True
        return ok

    def _get_dis_value(self, points):
        displacement_weight = np.zeros((len(points), 1))
        for i in range(len(points)):
            if self._dimension == 2:
                weight = max(
                    abs(op.nodeDisp(i, 1)),
                    abs(op.nodeDisp(i, 2)),
                )
            else:
                weight = max(
                    abs(op.nodeDisp(i, 1)),
                    abs(op.nodeDisp(i, 2)),
                    abs(op.nodeDisp(i, 3)),
                )

            displacement_weight[i] = max(weight / self._limit_dis - 1.0, 0)
        return np.sum(displacement_weight)

    def _get_stress_value(self, edges):
        stress_weight = np.zeros((len(edges), 1))
        for tag, i in enumerate(edges.keys()):
            edges[i].force = op.basicForce(tag)
            if isMac:
                edges[i].stress = edges[i].force / edges[i].area
            else:
                edges[i].stress = edges[i].force[0] / edges[i].area

            if edges[i].stress < 0:
                stress_weight[tag] = max(
                    abs(edges[i].stress) / self._sigma_C - 1.0,
                    0.0
                )
            else:
                stress_weight[tag] = max(
                    abs(edges[i].stress) / self._sigma_T - 1.0,
                    0.0
                )
        return np.sum(stress_weight)

    def _get_buckle_value(self, edges):
        buckle_weight = np.zeros((len(edges), 1))
        miu_buckle = 1.0
        for tag, i in enumerate(edges.keys()):
            edges[i].force = op.basicForce(tag)
            if isMac:
                edges[i].stress = edges[i].force / edges[i].area
            else:
                edges[i].stress = edges[i].force[0] / edges[i].area

            if edges[i].stress < 0:
                edges[i].inertia = (
                        edges[i].area ** 2
                        * (1 + self._ratio_ring ** 2)
                        / (
                                4 * math.pi
                                * (1 - self._ratio_ring ** 2)
                        )
                )
                force_cr = (
                                   math.pi ** 2
                                   * self._E * edges[i].inertia
                           ) / (miu_buckle * edges[i].len) ** 2
                buckle_stress_max = force_cr / edges[i].area
                buckle_weight[tag] = max(
                    abs(edges[i].stress) / abs(buckle_stress_max) - 1.0,
                    0.0
                )
        return np.sum(buckle_weight)

    def run(self, points, edges):
        if 'IS_RUNNING_DYNAMIC' not in os.environ:
            os.environ['IS_RUNNING_DYNAMIC'] = 'no'
        while os.environ['IS_RUNNING_DYNAMIC'] == 'yes':
            print('waiting for dynamics to be enabled')
            time.sleep(0.1)
        os.environ['IS_RUNNING_DYNAMIC'] = 'yes'
        is_struct = self._is_struct(points, edges)
        mass, dis_value, stress_value, buckle_value = 0.0, 0.0, 0.0, 0.0
        if is_struct:
            for i, edge in edges.items():
                mass += edge.len * edge.area * self._pho
            if self._use_dis_constraint:
                dis_value = self._get_dis_value(points)
            if self._use_stress_constraint:
                stress_value = self._get_stress_value(edges)
            if self._use_buckle_constraint:
                buckle_value = self._get_buckle_value(edges)
        os.environ['IS_RUNNING_DYNAMIC'] = 'no'
        return (
            is_struct,
            mass, dis_value, stress_value, buckle_value
        )
