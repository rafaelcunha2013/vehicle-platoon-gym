from Jammer import MyJammer
from Platooning import Platooning
import platform
import matplotlib.pyplot as plt


system = platform.system()
w_const = 80 / 3.6
w_agg_max = 80 / 3.6
w_aggre_min = 30 / 3.6
w_max = 83 / 3.6
length = 30000
th_int = 0.1
exp_dist = [40 * 200, 20 * 200]
episodes = 10
jammer = MyJammer(wvelo_const=w_const,
                  wvelo_aggre_max=w_agg_max,
                  wvelo_aggre_min=w_aggre_min,
                  wvelo_max=w_max,
                  th_int=th_int)
jammer.jammer_episodes(episodes, length, exp_dist=exp_dist)


delta = 5000
h = [[3, 3, 3], [3, 0.4, 0.4]]
Ddes = [1, 1]
position = [129, 20, 10, 0]
cd = 0.6
m = 20 * 1000
alpha_step = 1000
h_len = 1
area = 10.26
fixed_time = 100
rwd_const = 1.5
rwd_mult = 5
rwd_hc_weight = 1 / 5000
env = Platooning(jammer,
                 system=system,
                 num_states=6,
                 model_type='ACC',
                 stop_type='time_limit',
                 fixed_time=fixed_time,
                 reward_type='inst_delta_distance_per_fuel',
                 inst_delta=delta,
                 h=h,
                 Ddes=Ddes,
                 position=position,
                 cd=cd,
                 m=m,
                 alpha_step=alpha_step,
                 h_len=h_len,
                 area=area,
                 rwd_const=rwd_const,
                 rwd_mult=rwd_mult,
                 rwd_hc_weight=rwd_hc_weight)

num_episodes = 4
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

        next_state, reward, done, info = env.step(action)
        reward_history.append(reward)
    plt.plot(reward_history)
    plt.title("Reward (Normalized dist/fuel)")
    plt.show()

