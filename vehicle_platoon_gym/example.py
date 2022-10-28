from Jammer import MyJammer
from Platooning import Platooning


w_const = 80 / 3.6
w_agg_max = 100 / 3.6
w_aggre_min = 55 / 3.6
w_max = 126 / 3.6
length = 10000
episodes = 10
th_int = 0.1
jammer = MyJammer(wvelo_const=w_const,
                  wvelo_aggre_max=w_agg_max,
                  wvelo_aggre_min=w_aggre_min,
                  wvelo_max=w_max,
                  th_int=th_int)
jammer.jammer_episodes(episodes, length, exp_dist=[20 * 200, 10 * 200])


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
