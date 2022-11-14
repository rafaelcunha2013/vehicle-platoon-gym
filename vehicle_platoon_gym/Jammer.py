import numpy as np
import scipy.stats as stats
import matplotlib
import  matplotlib.pyplot as plt
import pickle
import os
from pathlib import Path


class MyJammer:
    def __init__(self, **kwargs):
        myclip_a, myclip_b = -2, 2
        my_mean, my_std = 0, 1.0
        a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
        self.noise = stats.truncnorm(a, b, loc=my_mean, scale=my_std)

        self.wvelo_const = kwargs['wvelo_const'] if 'wvelo_const' in kwargs else 80/3.6
        self.wvelo_aggre_max = kwargs['wvelo_aggre_max'] if 'wvelo_aggre_max' in kwargs else 100/3.6
        self.wvelo_aggre_min = kwargs['wvelo_aggre_min'] if 'wvelo_aggre_min' in kwargs else 55/3.6
        self.wvelo_max = kwargs['wvelo_max'] if 'wvelo_max' in kwargs else 126/3.6

        self.th_interval = kwargs['th_int'] if 'th_int' in kwargs else 0.1
        self.wvelo = self.wvelo_const
        self.wacc = 0
        self.Ts = 0.1
        self.myjammer = []
        self.step_count = 0
        self.inclination = -1

    def mode_constant_without_noise(self):
        self.wacc = 0
        if self.wvelo < self.wvelo_const - self.th_interval:
            self.wacc = 0.4854369
        if self.wvelo > self.wvelo_const + self.th_interval:
            self.wacc = -0.4854369
        self.wvelo = max(0.0, min(self.wvelo_max, self.wvelo + self.wacc * self.Ts))

    def mode_constant(self):
        if self.step_count % 50 == 0:
            self.wacc = 0.01 * self.noise.rvs()
            # self.wacc = 0.01 * np.random.uniform(-2, 2)
        if self.wvelo < 75/3.6:
            self.wacc = 0.4854369
        self.wvelo = max(0.0, min(self.wvelo_max, self.wvelo + self.wacc * self.Ts))

    def mode_crazy(self):
        g = 9.8
        if self.wvelo < self.wvelo_aggre_min:
            self.inclination = 1
        if self.wvelo > self.wvelo_aggre_max:
            self.inclination = -1
        if self.inclination == -1:
            wacc = -0.3*g
        else:
            wacc = 0.4854369
        self.wvelo = max(0.0, min(self.wvelo_max, self.wvelo + wacc * self.Ts))

    def mode_random(self):
        if self.step_count % 100 == 0:
            self.wacc = 0.1*self.noise.rvs()
        self.wvelo = max(30/3.6, min(150/3.6, self.wvelo + self.wacc * self.Ts))

    def step(self, mode):
        if mode == 0:
            # self.mode_constant()
            self.mode_constant_without_noise()
        if mode == 1:
            self.mode_crazy()
        if mode == 2:
            self.mode_random()
        self.myjammer.append(self.wvelo)
        self.step_count += 1

    def reset(self):
        self.wvelo = self.wvelo_const
        self.myjammer = []
        self.wacc = 0
        self.step_count = 0

    def generate(self, length):
        for i in range(int(length)):
            self.step(0)
        for i in range(int(length)):
            self.step(1)
        for i in range(int(length)):
            self.step(2)
        return self.myjammer

    def generate_markov(self, length, exp_dist):
        mode = 0
        while True:
            tm = np.random.exponential(exp_dist[mode])
            for i in range(int(tm)):
                self.step(mode)
                if self.step_count == length:
                    break
            mode = (mode + 1) % 2
            if self.step_count == length:
                break
        return self.myjammer

    def jammer_episodes(self, episodes, length, type="Markov", exp_dist=[1, 0]):
        self.many_jammers = []
        self.exp_dist = exp_dist
        for i in range(episodes):
            if type == "Markov":
                self.generate_markov(length, exp_dist)
            if type == "Mixed":
                self.generate(length)
            self.many_jammers.append(self.myjammer)
            self.reset()
        return self.many_jammers

    def plot(self):
        plt.plot(self.myjammer)
        plt.show()

    def plot_many(self, n=3, m=5, max_steps=300):
        # Plot many jammer profiles
        cont = 0
        time_vec = np.linspace(0, max_steps / 10, max_steps*100)
        # plt.figure(2,figsize=(15, 5))
        for i in range(n):
            for j in range(m):
                plt.subplot2grid((n, m), (i, j))
                plt.plot(time_vec, self.many_jammers[cont][0:max_steps*100])
                plt.xlabel('Time (s)')
                plt.ylabel('Velocity (m/s)')
                cont += 1
        # plt.show()

    @staticmethod
    def path(system):
        if system == 'colab':
            path = 'drive/MyDrive/Colab Notebooks/Simulations/Jammer/'
        if system == 'mac':
            path = '../Jammer/'
        if system == 'peregrine':
            path = '/data/p285087/Jammer/'
        if system == 'windows':
            path = '..\\Jammer\\'
        Path(path).mkdir(parents=True, exist_ok=True)
        # if not os.path.isdir(path):
        #    os.mkdir(path)
        return path

    def save(self, my_jammers, name, system):
        path = self.path(system)
        pickle.dump(my_jammers, open(path + name + '.pkl', "wb"))

    def load(self, name, system):
        path = self.path(system)
        jammer = pickle.load(open(path + name + '.pkl', "rb"))
        return jammer

    def name(self):
        name = 'mk_len_' + str(np.shape(self.many_jammers)[1]) + '_ep_' + \
               str(np.shape(self.many_jammers)[0]) + '_dist_' + str(self.exp_dist[0]) + '_' + str(self.exp_dist[1])
        return name


if __name__ == "__main__":
    system = 'windows'

    w_const = 80/3.6
    w_agg_max = 80/3.6
    w_agg_min = 30/3.6
    w_max = 83/3.6
    length = 10000
    exp_dist = [40*200, 20*200]
    episodes = 2
    for i in range(3):
        # jammer = MyJammer()
        jammer = MyJammer(wvelo_const=w_const, wvelo_aggre_max=w_agg_max, wvelo_aggre_min=w_agg_min, wvelo_max=w_max)
    # constant_jammer = jammer.generate(length)
    # jammer.plot()
    # jammer.reset()

    # exp_dist = [0, 1]
    # markov_jammer = jammer.generate_markov(length, exp_dist)
    # jammer.plot()
    # jammer.reset()

        my_jammers = jammer.jammer_episodes(episodes, length, exp_dist=exp_dist)
    # jammer.plot_many()
    # markov_length_episodes_exp_dist_01
        name = jammer.name()
        if i != 0:
            name = name + '_' + '{0:02}'.format(i)
        jammer.save(jammer, name, system)
    # jammer05 = jammer.load(name, 'windows')
    # jammer05.plot_many()

