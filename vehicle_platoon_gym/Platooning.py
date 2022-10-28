import numpy as np
import gym
from gym import spaces
from scipy.interpolate import interp1d
import os.path
import copy

from ACC_Model import model
from Jammer import MyJammer


class Platooning(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}
    N_DISCRETE_ACTIONS = 2

    def __init__(self,
                 myjammer,
                 system='None',
                 my_states='None',
                 num_states=7,
                 model_type='ACC_CACC',
                 stop_type='time_limit',
                 fixed_time=200,
                 reward_type='total_distance_per_fuel',
                 smooth_change=True,
                 **kwargs):
        super(Platooning, self).__init__()
        if my_states == 'None':
            if num_states == 6:
                self.my_state_type = 'fl1_fl2'
            if num_states == 7:
                self.my_state_type = 'fl1_fl2_fuel'
            if num_states == 8:
                self.my_state_type = 'fl1_fl2_leader'
        if my_states == 'fl1_fl2_a0':
            self.my_state_type = my_states
            num_states = 7

        # reward parameters
        self.inst_delta = kwargs['inst_delta'] if 'inst_delta' in kwargs else 0
        self.rwd_const = kwargs['rwd_const'] if 'rwd_const' in kwargs else 0
        self.rwd_mult = kwargs['rwd_mult'] if 'rwd_mult' in kwargs else 1
        self.rwd_hc_weight = kwargs['rwd_hc_weight'] if 'rwd_hc_weight' in kwargs else 140

        self.obs_size = 1e30 * np.ones((num_states,), dtype=np.float32)
        self.action_space = spaces.Discrete(Platooning.N_DISCRETE_ACTIONS)
        self.observation_space = spaces.Box(-self.obs_size, self.obs_size, dtype=np.float32)
        if model_type == 'ACC':
            if 'h' in kwargs:
                h = kwargs['h']
            else:
                h = ['None'] * 2
                h[0] = [3, 3, 3]
                h[1] = [3, 1, 1]
            A = np.zeros((2, 9, 9), dtype=float)
            B = np.zeros((2, 9, 2), dtype=float)
            for i in [0, 1]:
                A[i], B[i] = model('ACC', Ts=0.1, lamb=0.5, tal=0.2, h=h[i])
            self.A = A
            self.B = B
        else:
            self.A, self.B = model(model_type)
        self.Ts = 0.1
        self.myjammer = myjammer.many_jammers
        self.jammer_name = myjammer.name()
        self.jammer_exp_dist = myjammer.exp_dist
        self.system = system
        self.my_full_jammer = myjammer

        self.model_type = model_type
        self.stop_type = stop_type
        self.reward_type = reward_type

        # Initialization of the states
        ddes = 0
        self.N = 3
        # self.pj = kwargs['pj'] if 'pj' in kwargs else 100

        self.pj = kwargs['position'][0] if 'position' in kwargs else 100
        p0 = kwargs['position'][1] if 'position' in kwargs else 70
        p1 = kwargs['position'][2] if 'position' in kwargs else 40
        p2 = kwargs['position'][3] if 'position' in kwargs else 10
        # self.pj, p0, p1, p2 = kwargs['position'][0] if 'position' in kwargs else 100, 70, 40, 10
        # self.pj, p0, p1, p2 = kwargs['position'][0] if 'position' in kwargs else 100, 70, 40, 10
        # p0, p1, p2 = 150, 80, 10  # 70, 40, 10
        self.x_original = np.zeros((self.A[1][1].size, 1))
        self.dim = int(self.A[1][1].size / self.N)
        self.x_original[0] = p0
        self.x_original[self.dim] = p1 - p0 + ddes
        self.x_original[2*self.dim] = p2 - p1 + ddes

        self.x_original[1] = 80 / 3.6
        self.x = copy.deepcopy(self.x_original)
        self.w = np.zeros((2, 1))
        self.w[0] = self.pj

        # self.Ddes_ACC, self.Ddes_CACC = kwargs['Ddes'] if 'Ddes' in kwargs else 7, 7
        self.Ddes_ACC = kwargs['Ddes'][0] if 'Ddes' in kwargs else 7
        self.Ddes_CACC = kwargs['Ddes'][1] if 'Ddes' in kwargs else 7
        self.Ddes_vec = np.array([self.Ddes_ACC, self.Ddes_CACC])
        self.gas = 7.5 * 10 ** 6

        # Initialization of loop variables
        self.episode = 0
        self.mode = 0
        self.fixed_action_time = fixed_time
        self.time_lapse = 0
        self.jammer_count = 0

        self.myalpha = 1
        self.myalpha_step = kwargs['alpha_step'] if 'alpha_step' in kwargs else self.fixed_action_time

        # Fuel computation parameters
        self.Cd = kwargs['cd'] if 'cd' in kwargs else 1
        self.ro = kwargs['ro'] if 'ro' in kwargs else 1.225
        self.m = kwargs['m'] if 'm' in kwargs else 1000
        self.Area_front = kwargs['area'] if 'area' in kwargs else 2.1
        self.Cr = 0.008
        self.vref = 10
        self.g = 9.8
        self.Fair = 0.5 * self.Cd * self.Area_front * self.ro
        self.Froll = self.Cr * self.m * self.g

        self.total_fuel = np.zeros((self.N,), dtype=float)
        self.total_co2 = np.zeros((self.N,), dtype=float)
        self.car_drag = self.generate_car_drag()
        self.hc_table = self.generate_hc_table()

        # Initialization of 'memory' variables
        self.h_len = kwargs['h_len'] if 'h_len' in kwargs else 1
        self.platoonfuel = []
        self.platoonco2 = []
        self.co2_emission = []
        self.co2_emission.append(np.zeros(self.N))
        self.mystates_history = [[]]
        self.mycontrol_history = [[]]
        self.myalpha_history = [[]]
        self.myfuel_history = [[]]
        self.myenergy_air_history = [[]]
        self.myenergy_acc_history = [[]]
        self.myhc_history = [[]]

        # Flag conditions
        self.collision = False
        self.smooth_change = smooth_change

        # End conditions
        self.finalStep = np.shape(myjammer.many_jammers)[1]
        self.done = False

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def generate_name(self):
        my_name = 'Ag_' + self.jammer_name + '_' + self.my_state_type + '_' + self.model_type + '_' + \
                self.stop_type + '_' + str(self.fixed_action_time) + '_' + self.reward_type
                # '_hd_' + str(hidden_dim) + '_batch_' + str(batch_size) + '_buffer_' + str(buffer_size) + '_targNN_' + str(steps2update_target_NN)
        return my_name

    @staticmethod
    def generate_car_drag():
        dist_x0 = [0, 5, 10, 15, 20, 30, 40, 50, 60, 120, 130]
        points_y0 = [0.15, 0.06, 0.04, 0, 0, 0, 0, 0, 0, 0, 0]

        dist_x1 = [0, 2.5, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80]
        points_y1 = [0.65, 0.45, 0.41, 0.40, 0.35, 0.31, 0.28, 0.24, 0.20, 0.16, 0.12, 0]

        dist_x2 = [0, 2.5, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80]
        points_y2 = [0.72, 0.55, 0.49, 0.46, 0.44, 0.41, 0.40, 0.36, 0.32, 0.28, 0.24, 0]

        car_drag = np.zeros([3, 2000])
        for i in range(150):
            car_drag[0][i] = 1 - interp1d(dist_x0, points_y0)(i / 10)

        for i in range(800):
            car_drag[1][i] = 1 - interp1d(dist_x1, points_y1)(i / 10)
            car_drag[2][i] = 1 - interp1d(dist_x2, points_y2)(i / 10)

        return car_drag

    def speed_acceleration_control(self):
        self.x[1] = min(self.x[1], 30)
        # if self.x[1] == 30:
        #     self.x[2] = 0
        self.x[4] = min(self.x[4], 30-self.x[1])
        self.x[7] = min(self.x[7], 30-self.x[1]-self.x[4])
        for i in range(self.N):
            self.x[self.dim * i + 2] = max(-3, min(self.x[self.dim * i + 2], 2))

    def compute_total_fuel(self):
        fuel_vehicles = np.zeros((self.N,), dtype=float)
        v = [[]] * self.N
        self.energy_air = [[]] * self.N
        self.energy_acc = [[]] * self.N
        position = int(10 * round(float(abs(self.x[self.dim])) + float(self.Ddes_vec[self.mode]), 1))
        v[0] = self.x[1]
        for j in range(self.N):
            if j > 0:
                position = int(10 * round(float(abs(self.x[self.dim * j])) + float(self.Ddes_vec[self.mode]), 1))
                v[j] = v[j - 1] + self.x[1 + self.dim * j]
            cd_aux = self.car_drag[j][position]
            fu = self.m * self.x[2 + self.dim * j] + self.Froll + (cd_aux * self.Fair * v[j] ** 2)  # Think about the aceleration
            self.energy_air[j] = cd_aux * self.Fair * v[j] ** 2
            self.energy_acc[j] = self.m * self.x[2 + self.dim * j]

            if fu > 0:
                fuel_vehicles[j] += fu * v[j] * self.Ts
        self.total_fuel += fuel_vehicles
        return fuel_vehicles

    def compute_co2_emission(self):
        b0 = 7.613534994965560
        b1 = -0.138565467462594
        b2 = 0.003915102063854
        b3 = -0.000049451361017
        b4 = 0.000000238630156
        ms_to_mph = 2.23694
        v_leader = self.x[1]
        v_follower1 = self.x[1] + self.x[self.dim + 1]
        v_follower2 = self.x[1] + self.x[self.dim + 1] + self.x[2*self.dim + 1]
        v = ms_to_mph * np.array([v_leader, v_follower1, v_follower2])
        aux = b0 + b1 * v + b2 * v ** 2 + b3 * v ** 3 + b4 * v ** 4
        co2_vehicles = np.exp(aux)
        self.total_co2 += co2_vehicles.squeeze(1)
        return co2_vehicles

    def compute_total_hc1(self):
        v_leader = self.x[1]
        v_follower1 = self.x[1] + self.x[self.dim + 1]
        v_follower2 = self.x[1] + self.x[self.dim + 1] + self.x[2*self.dim + 1]
        v_cars = np.array([v_leader, v_follower1, v_follower2])
        a = np.array([self.x[2], self.x[5], self.x[8]])

        L = [[-0.87605, 0.03627, -0.00045, 2.55E-06],
             [0.081221, 0.009246, -0.00046, 4.00E-06],
             [0.037039, -0.00618, 2.96E-04, -1.86E-06],
             [-0.00255, 0.000468, -1.79E-05, 3.86E-08]]
        L_array = np.array(L)
        y = 0
        for idi, my_l in np.ndenumerate(L_array):
            y += my_l * v_cars**idi[0] * a**idi[1]
        hc = np.exp(y).squeeze(1)
        for i in range(3):
            if self.time_lapse > 3:
                error = hc[i] - self.myhc_history[self.episode % self.h_len][self.time_lapse-2][i]
                if error < 0:
                    hc[i] = self.myhc_history[self.episode % self.h_len][self.time_lapse-2][i]
        return hc

    @staticmethod
    def generate_hc_table1():
        l = [[-0.87605, 0.03627, -0.00045, 2.55E-06],
             [0.081221, 0.009246, -0.00046, 4.00E-06],
             [0.037039, -0.00618, 2.96E-04, -1.86E-06],
             [-0.00255, 0.000468, -1.79E-05, 3.86E-08]]

        m = [[-0.75584, 0.021283, -0.00013, 7.39e-07],
             [-0.00921, 0.011364, -0.0002, 8.45e-07],
             [0.036223, 0.000226, 4.03e-08, -3.5e-08],
             [0.003968, -9e-05, 2.4e-06, -1.6e-08]]
        l_array = np.array(l).T
        m_array = np.array(m).T
        # a = np.round(3.6 * np.linspace(-2, 2, 41), 1)
        a = np.round([3.6 * 1.8, 3.6*0.9], 1)
        v = np.round(3.6 * np.linspace(0, 30, 301), 1)
        hc = np.zeros((len(v), len(a)), dtype=float)
        # hc = dict()
        for i_a, value_a in enumerate(a):
            y_old = 0
            for i_v, value_v in enumerate(v):
                y = 0
                if value_a >= 0:
                    for idi, value_l in np.ndenumerate(l_array):
                        y += value_l * value_v*idi[0] * value_a**idi[1]
                else:
                    for idi, value_m in np.ndenumerate(m_array):
                        y += value_m * value_v*idi[0] * value_a**idi[1]
                y = np.exp(y)
                if y > y_old:
                    hc[i_v][i_a] = y
                    # hc[(value_a, value_v)] = y
                    y_old = y
                else:
                    hc[i_v][i_a] = y_old
                    # hc[(value_a, value_v)] = y_old
        return hc

    @staticmethod
    def generate_hc_table():
        l = [[-0.87605, 0.03627, -0.00045, 2.55E-06],
             [0.081221, 0.009246, -0.00046, 4.00E-06],
             [0.037039, -0.00618, 2.96E-04, -1.86E-06],
             [-0.00255, 0.000468, -1.79E-05, 3.86E-08]]

        m = [[-0.75584, 0.021283, -0.00013, 7.39e-07],
             [-0.00921, 0.011364, -0.0002, 8.45e-07],
             [0.036223, 0.000226, 4.03e-08, -3.5e-08],
             [0.003968, -9e-05, 2.4e-06, -1.6e-08]]
        l_array = np.array(l)
        m_array = np.array(m)
        a = 3.6 * np.linspace(-2, 2, 41)
        # a = [3.6 * 1.8, 3.6*0.9]
        v = 3.6 * np.linspace(0, 34, 341)
        hc = np.zeros((len(v), len(a)), dtype=float)
        y = np.zeros((len(v), len(a)), dtype=float)
        y_old = np.zeros(len(a), dtype=float)
        # hc = dict()
        for idk, a_value in np.ndenumerate(a):
            # y_old = 0
            for idz, v_value in np.ndenumerate(v):
                if a_value >= 0:
                    for idi, i in np.ndenumerate(l_array):
                        y[idz[0], idk[0]] = y[idz[0], idk[0]] + l_array[idi[1], idi[0]] * v[idz[0]]**(idi[0]) * a[idk[0]]**(idi[1])
                    if y[idz[0], idk[0]] > y_old[idk[0]]:
                        hc[idz[0], idk[0]] = np.exp(y[idz[0], idk[0]])
                        y_old[idk[0]] = y[idz[0], idk[0]]
                    else:
                        hc[idz[0], idk[0]] = np.exp(y_old[idk[0]])

                if a_value < 0:
                    for idi, i in np.ndenumerate(m_array):
                        y[idz[0], idk[0]] = y[idz[0], idk[0]] + m_array[idi[1], idi[0]] * v[idz[0]]**(idi[0]) * a[idk[0]]**(idi[1])
                    hc[idz[0], idk[0]] = np.exp(y[idz[0], idk[0]])

        return hc

    def compute_total_hc(self):
        v_leader = self.x[1]
        v_follower1 = self.x[1] + self.x[self.dim + 1]
        v_follower2 = self.x[1] + self.x[self.dim + 1] + self.x[2*self.dim + 1]
        v = np.round(np.array([v_leader, v_follower1, v_follower2]), 1)
        a = np.round(np.array([self.x[2], self.x[5], self.x[8]]), 1)
        hc = np.zeros((self.N,), dtype=float)
        for i in range(3):
            a[i] = max(-2, min(a[i], 2))
            v[i] = max(0, min(v[i], 30))
            hc[i] = self.hc_table[int(10 * v[i][0])][int((2 + a[i][0]) * 10)]
        return hc

    def collision_control(self):
        collision = False
        for j in range(1, self.N):
            condition = -(self.x[j * self.dim] - float(self.Ddes_vec[self.mode]))
            if condition < 1 or condition > 150:
                collision = True
                self.collision = True
        return collision

    def save_variables(self):
        self.platoonfuel.append(self.total_fuel[2] + self.total_fuel[1] + self.total_fuel[0])
        # self.co2_emission.append(self.compute_co2_emission())
        self.mystates_history[self.episode % self.h_len].append(self.x)
        self.mycontrol_history[self.episode % self.h_len].append(self.mode)
        self.myalpha_history[self.episode % self.h_len].append(self.myalpha)
        self.myfuel_history[self.episode % self.h_len].append(self.compute_total_fuel())
        self.myenergy_air_history[self.episode % self.h_len].append(self.energy_air)
        self.myenergy_acc_history[self.episode % self.h_len].append(self.energy_acc)
        self.myhc_history[self.episode % self.h_len].append(self.compute_total_hc())

    def append_variables(self):
        self.mystates_history.append([])
        self.mycontrol_history.append([])
        self.myalpha_history.append([])
        self.myfuel_history.append([])
        self.myenergy_air_history.append([])
        self.myenergy_acc_history.append([])
        self.myhc_history.append([])

    def organize_state(self):
        if self.model_type == 'ACC':
            if self.my_state_type == 'fl1_fl2_leader':
                state = copy.deepcopy(self.x[1:].squeeze(1))
            if self.my_state_type == 'fl1_fl2_a0':
                state = copy.deepcopy(self.x[2:].squeeze(1))
            if self.my_state_type == 'fl1_fl2':
                state = copy.deepcopy(self.x[3:].squeeze(1))
            if self.my_state_type == 'fl1_fl2_fuel':
                state = copy.deepcopy(self.x[3:].squeeze(1))
                state = np.concatenate((state, np.array([self.total_fuel[2] + self.total_fuel[1] + self.total_fuel[0]])))
                state[6] = 10 * state[6] / self.gas
        else:
            if self.my_state_type == 'fl1_fl2_a0':
                state = self.x[[2, 5, 6, 7, 10, 11, 12]].squeeze(1)
            if self.my_state_type == 'fl1_fl2':
                state = self.x[[5, 6, 7, 10, 11, 12]].squeeze(1)
            if self.my_state_type == 'fl1_fl2_fuel':
                state = self.x[[5, 6, 7, 10, 11, 12]].squeeze(1)
                state = np.concatenate((state, np.array([self.total_fuel[2] + self.total_fuel[1]])))
                state[6] = 10 * state[6] / self.gas
        if self.my_state_type == 'fl1_fl2_leader':
            state[0] = state[0] / 10
            state[2] = state[2] / 10
            state[5] = state[5] / 10
        else:
            state[0] = state[0] / 10
            state[3] = state[3] / 10
        return state

    def delta_load_and_position(self, delta, myload_history):
        delta_load = np.sum(myload_history[self.episode % self.h_len][-delta:], axis=0)
        position_history = np.moveaxis(self.mystates_history[self.episode % self.h_len], 0, 1)[0].squeeze(1)
        delta_position = position_history[-1:] - position_history[-min(len(position_history), delta)]
        return (delta_load[1] + delta_load[2])/2, delta_position

    def done_check(self):
        if self.collision_control():
            self.done = True
        if self.stop_type == 'time_limit':
            if self.time_lapse >= self.finalStep or len(self.myjammer[self.episode - len(self.myjammer) * self.jammer_count]) == self.time_lapse:
                self.done = True
        if self.stop_type == 'fuel_limit':
            if self.total_fuel[2] + self.total_fuel[1] >= 2 * self.gas:
                self.done = True
        return self.done

    def dynamics(self, action):
        self.w[0] += self.Ts * self.myjammer[self.episode - len(self.myjammer) * self.jammer_count][self.time_lapse - 1]
        self.x = self.myalpha * (np.matmul(self.A[action], self.x) + np.matmul(self.B[action], self.w)) \
                 + (1 - self.myalpha) * (np.matmul(self.A[1 - action], self.x) + np.matmul(self.B[1 - action], self.w))
        next_state = self.organize_state()
        return next_state

    def reward_efficiency(self, k, collision):
        if self.done:
            if collision:
                reward = -1
            else:
                reward = k / self.fixed_action_time
        else:
            reward = 1
        return reward

    def reward_distance_per_load(self, k, collision, load, myload_history):
        delta_load, delta_position = load(k, myload_history)
        if self.done:
            if collision:
                reward = [-50.0]
            else:
                if delta_load < 1:
                    reward = [0]
                else:
                    reward = 1000 * delta_position / delta_load
        else:
            if delta_load < 1:
                reward = [0]
            else:
                reward = 1000 * delta_position / delta_load

        return reward[0]

    def generate_reward(self, type, k, collision):
        if type == 'efficiency':
            reward = self.reward_efficiency(k, collision)
        if type == 'inst_delta_distance_per_fuel' or 'total_distance_per_fuel':
            hist = self.myfuel_history
            load = self.delta_load_and_position
            reward = self.reward_distance_per_load(self.inst_delta, collision, load, hist) * self.rwd_mult - self.rwd_const
        if type == 'inst_delta_distance_per_hc':
            hist = self.myhc_history
            load = self.delta_load_and_position
            reward = self.reward_distance_per_load(self.inst_delta, collision, load, hist) * self.rwd_mult - self.rwd_const
        if type == 'inst_delta_distance_per_cost':
            hist_fuel = self.myfuel_history
            hist_hc = self.myhc_history
            load = self.delta_load_and_position
            reward_fuel = self.reward_distance_per_load(self.inst_delta, collision, load, hist_fuel)
            reward_hc = self.reward_distance_per_load(self.inst_delta, collision, load, hist_hc)
            reward = (reward_fuel + self.rwd_hc_weight * reward_hc) * self.rwd_mult - self.rwd_const
        return reward

    def alpha_dyn(self):
        if len(self.mycontrol_history[self.episode % self.h_len]) > 1:
            if self.mycontrol_history[self.episode % self.h_len][-1] != self.mycontrol_history[self.episode % self.h_len][-2]:
                if self.myalpha != 1:
                    self.myalpha = 1 - self.myalpha
                else:
                    self.myalpha = 0
        self.myalpha = min(1, self.myalpha + (1/self.myalpha_step))

    def load_jammer(self):
        self.jammer_count += 1
        jammer = MyJammer(wvelo_const=self.my_full_jammer.wvelo_const,
                          wvelo_aggre_max=self.my_full_jammer.wvelo_aggre_max,
                          wvelo_aggre_min=self.my_full_jammer.wvelo_aggre_min,
                          wvelo_max=self.my_full_jammer.wvelo_max,
                          th_int=self.my_full_jammer.th_interval)
        name = self.jammer_name + '_' + '{0:02}'.format(self.jammer_count)
        if os.path.isfile(jammer.path(self.system) + name + '.pkl'):
            jammer = jammer.load(name, self.system)
        else:
            episodes, length = np.shape(self.myjammer)
            _ = jammer.jammer_episodes(episodes, length, exp_dist=self.jammer_exp_dist)
            jammer.save(jammer, name, self.system)
        return jammer.many_jammers

    def step(self, action):
        self.mode = action
        for k in range(1 * self.fixed_action_time):
            self.time_lapse += 1
            self.save_variables()
            self.alpha_dyn()
            next_state = self.dynamics(action)
            if self.done_check():
                break

        reward = self.generate_reward(self.reward_type, k, self.collision_control())
        if self.done:
            self.episode += 1
            if self.episode % len(self.myjammer) == 0:
                self.myjammer = self.load_jammer()
            self.append_variables()

        return next_state, reward, self.done, {}

    def reset(self):
        self.done = False
        self.x = self.x_original
        self.w[0] = self.pj
        self.total_fuel = np.zeros((self.N,), dtype=float)
        self.total_co2 = np.zeros((self.N,), dtype=float)
        self.time_lapse = 0
        self.platoonfuel = []
        self.platoonco2 = []
        self.co2_emission = []
        self.myalpha = 1
        if self.episode % self.h_len == 0 and self.episode > 0:
            self.mystates_history = [[]]
            self.mycontrol_history = [[]]
            self.myalpha_history = [[]]
            self.myfuel_history = [[]]
            self.myenergy_air_history = [[]]
            self.myenergy_acc_history = [[]]
            self.myhc_history = [[]]

        state = self.organize_state()
        self.mode = 0
        self.collision = False
        return state


if __name__ == '__main__':
    # from stable_baselines3.common.env_checker import check_env
    #system = platform.system()
    system = 'windows'
    w_const = 80/3.6
    w_agg_max = 100/3.6
    w_aggre_min = 55/3.6
    w_max = 126/3.6
    length = 10000
    episodes = 10
    th_int = 0.1
    jammer = MyJammer(wvelo_const=w_const,
                      wvelo_aggre_max=w_agg_max,
                      wvelo_aggre_min=w_aggre_min,
                      wvelo_max=w_max,
                      th_int=th_int)
    jammer.jammer_episodes(episodes, length, exp_dist=[20*200, 10*200])
    # name = 'mk_len_10000_ep_30_dist_4000_600'
    # name = 'mk_len_10000_ep_1_dist_1_0'
    # name = 'markov_length_10000_episodes_15_dist_4000_600'
    # name = 'mk_len_90000_ep_500_dist_4000_2000'
    # jammer = jammer.load(name, system)

    delta = 303
    h = [[3, 3, 3], [3, 0.4, 0.4]]
    Ddes = [1, 1]
    position = [129, 20, 10, 0]
    cd = 0.1
    m = 1000
    alpha_step = 400
    h_len = 3

    env = Platooning(jammer,
                     system=system,
                     num_states=8,
                     model_type='ACC',
                     stop_type='time_limit',
                     fixed_time=200,
                     reward_type='inst_delta_distance_per_hc',
                     inst_delta=delta,
                     h=h,
                     Ddes=Ddes,
                     position=position,
                     cd=cd,
                     m=m,
                     alpha_step=alpha_step,
                     h_len=h_len)
    # env = Platooning(jammer.many_jammers, num_states=7)
    # env = Platooning_6_states(jammer.many_jammers)
    # check_env(env)
    num_episodes = 3
    for episode in range(num_episodes):
        reward_history = []
        state = env.reset()
        done = False
        i = 0
        while not done:
            i += 1
            if i < 10:
                action = 1
            elif i < 20:
                action = 0
            else:
                action = 1

            # action = 1 # np.random.randint(2)
            next_state, reward, done, info = env.step(action)
            reward_history.append(reward)
            # state = next_state
            # plt.plot(reward_history)
            # plt.show()
        print(episode)
