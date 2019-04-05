import numpy as np
import matplotlib.pyplot as plt
import random


def fit_fun(pos, vel, data_X, data_Y):  # 适应函数
    # print(np.array(data_X).shape)
    # print(np.array(data_Y).shape)
    # print(np.array(pos).shape)
    # print(np.array(vel).shape)

    # print(sum(np.array(data_Y) - (np.array(data_X) * np.array(list(pos)) + np.array(vel))))
    return -(sum((np.array(data_Y) - (np.array(data_X) * np.array(list(pos)) + np.array(vel)) ** 2)))


class Particle:
    # 初始化
    def __init__(self, position, speed, data_X, data_Y, x_max, max_vel, dim):
        self.__pos = position  # 粒子的位置
        self.__vel = speed  # 粒子的速度
        self.__data_X = data_X
        self.__data_Y = data_Y
        self.__bestPos = [0.0 for i in range(dim)]  # 粒子最好的位置
        self.__fitnessValue = fit_fun(self.__pos, self.__vel, self.__data_X, self.__data_Y)  # 适应度函数值

    def set_pos(self, i, value):
        self.__pos[i] = value

    def get_pos(self):
        return self.__pos

    def set_best_pos(self, i, value):
        self.__bestPos[i] = value

    def get_best_pos(self):
        return self.__bestPos

    def get_data_X(self):
        return self.__data_X

    def get_data_Y(self):
        return self.__data_Y

    def set_vel(self, i, value):
        self.__vel[i] = value

    def get_vel(self):
        return self.__vel

    def set_fitness_value(self, value):
        self.__fitnessValue = value

    def get_fitness_value(self):
        return self.__fitnessValue


class PSO:
    def __init__(self, data_w, data_b, data_X, data_Y, dim, size, iter_num, x_max, max_vel,
                 best_fitness_value=float('Inf'), C1=2, C2=2, W=1):
        self.C1 = C1
        self.C2 = C2
        self.W = W
        self.dim = dim  # 粒子的维度
        self.size = size  # 粒子个数
        self.iter_num = iter_num  # 迭代次数
        self.x_max = x_max
        self.max_vel = max_vel  # 粒子最大速度
        self.best_fitness_value = best_fitness_value
        self.best_position = [0.0 for i in range(dim)]  # 种群最优位置
        self.fitness_val_list = []  # 每次迭代最优适应值
        self.best_b = []

        # 对种群进行初始化
        self.Particle_list = [
            Particle([d[i] for d in data_w], list(data_b[i]) * dim, data_X[i], data_Y[i], self.x_max, self.max_vel,
                     self.dim) for i in range(size)]

    def set_bestFitnessValue(self, value):
        self.best_fitness_value = value

    def get_bestFitnessValue(self):
        return self.best_fitness_value

    def set_bestPosition(self, i, value):
        self.best_position[i] = value

    def get_bestPosition(self):
        return self.best_position

    # 更新位置
    def update_vel(self, part):
        for i in range(self.dim):
            vel_value = self.W * part.get_vel()[i] + self.C1 * random.random() * (
                        part.get_best_pos()[i] - part.get_pos()[i]) \
                        + self.C2 * random.random() * (self.get_bestPosition()[i] - part.get_pos()[i])
            if vel_value > self.max_vel:
                vel_value = self.max_vel
            elif vel_value < -self.max_vel:
                vel_value = -self.max_vel
            part.set_vel(i, vel_value)

    # 更新位置
    def update_pos(self, part):
        for i in range(self.dim):
            pos_value = part.get_pos()[i] + part.get_vel()[i]
            part.set_pos(i, pos_value)
        value = fit_fun(part.get_pos(), part.get_vel(), part.get_data_X(), part.get_data_Y())
        if value < part.get_fitness_value():
            part.set_fitness_value(value)
            for i in range(self.dim):
                part.set_best_pos(i, part.get_pos()[i])
        if value < self.get_bestFitnessValue():
            self.set_bestFitnessValue(value)
            self.best_b = []
            for p in self.Particle_list:
                self.best_b.append(p.get_vel()[0])
            for i in range(self.dim):
                self.set_bestPosition(i, part.get_pos()[i])

    def update(self):
        for i in range(self.iter_num):
            for part in self.Particle_list:
                self.update_vel(part)  # 更新速度
                self.update_pos(part)  # 更新位置
            self.fitness_val_list.append(self.get_bestFitnessValue())  # 每次迭代完把当前的最优适应度存到列表
        da_w = []
        for p in self.Particle_list:
            da_w.append(p.get_best_pos())
        # print(np.array(da_w).transpose().shape)
        # print(np.array(self.best_b).shape)
        return self.fitness_val_list, self.get_bestPosition(), np.array(da_w).transpose(), self.best_b
