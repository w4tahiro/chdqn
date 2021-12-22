import gym
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import matplotlib.pyplot as plt


class QFunction(chainer.Chain):
    def __init__(self, obs_size, hidden, n_actions):
        super().__init__()
        with self.init_scope():
            self.l0 = L.Linear(obs_size, hidden)                                   # 入力数,中間層のノード
            self.l1 = L.Linear(hidden, hidden)                                     # 中間層,中間層のノード        
            self.l2 = L.Linear(hidden, n_actions)                                  # 中間層,出力層のノード
    def __call__(self, x, test=False):
        h0 = F.tanh(self.l0(x))                                                    # 中間層の活性化関数
        h1 = F.tanh(self.l1(h0))                                                   # 中間層の活性化関数
        h2 = self.l2(h1)                                                           # 出力層の活性化関数(恒等関数)
        return chainerrl.action_value.DiscreteActionValue(h2)                      # 深層強化学習関数

env        = gym.make('CartPole-v0')                                               # 倒立振子
steps      = 200                                                                   # 1試行のstep数
n_episodes = 5                                                                   # 総試行回数def400
gamma      = 0.99
loop = 10
history = [[[0 for j in range(1)]for i in range(loop)]for i in range(6)]
ave = [[0 for i in range(1)]for j in range(6)]
episodehistory = [[[0 for j in range(1)]for i in range(loop)]for i in range(6)]
episodeave = [[0 for i in range(1)]for j in range(6)]
unit = [8,16,32,64,128,256]
color = ['red','blue','yellow','green','black','brown']
count = 0

for u in unit:
    for i in range(loop):
        # DQN設定
        print('unit:',u, 'loop:',i+1)
        q_func     = QFunction(env.observation_space.shape[0], u, env.action_space.n)     # 入力数4, 中間層def50, 出力層2
        print('初期化')
        opt        = chainer.optimizers.Adam(eps=1e-2)                                     # 最適化関数
        opt.setup(q_func)
        explorer   = chainerrl.explorers.LinearDecayEpsilonGreedy \
                    (start_epsilon=1, end_epsilon=0.1, decay_steps=n_episodes, random_action_func=env.action_space.sample)                # ε-greedy法
        ex_rep     = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)                # 経験再生(experience replay)
        phi        = lambda x: x.astype(np.float32, copy=False)
        agent      = chainerrl.agents.DQN \
                    (q_func, opt, ex_rep, gamma, explorer, replay_start_size=500, update_interval=1, target_update_interval=100, phi=phi) # 深層強化学習
        sumsum = 0
        total_step = 0
        print(ex_rep)
        for episode in range(n_episodes):                                                  # エピソード数のループ
            observ      = env.reset()
            done        = False
            reward      = 0
            reward_sum  = 0
            for t in range(steps):                                                         # 1試行のループ
                action                   = agent.act_and_train(observ, reward)             # アクション決定
                observ, reward, done, _  = env.step(action)                                # アクション後の観測値
                reward_sum              += reward
                total_step += 1                                          # 報酬追加
                if done: break
            
            sumsum = sumsum + reward_sum
            agent.stop_episode_and_train(observ, reward, done)                             # DQNの重み更新
            print('episode:', episode, 'reward_sum:', reward_sum)
            history[count][i].append(sumsum)
            episodehistory[count][i].append(reward_sum)
        del q_func
        del opt
        del explorer
        del ex_rep
        del phi
        del agent
    count += 1

total = 0
episode_total = 0
count = 0
for count in range(6):
    for i in range(n_episodes):
        for j in range(loop):
            total += history[count][j][i]
            episode_total += episodehistory[count][j][i]
        total = total / loop
        episode_total = episode_total / loop
        ave[count].append(total)
        episodeave[count].append(episode_total)

Figure = plt.figure() #全体のグラフを作成
ax1 = Figure.add_subplot(2,1,1) #1つ目のAxを作成
ax2 = Figure.add_subplot(2,1,2) #2つ目のAxを作成

count = 0
for c in color:
    for i in range(loop):
        ax1.plot(history[count][i],marker='.',linestyle='None',color=c,alpha=0.4)
        ax2.plot(episodehistory[count][i],marker='.',linestyle='None',color=c,alpha=0.4)
        
    count += 1

"""ax1.plot(ave[0],label='8',color='red',linewidth=5.0)
ax1.plot(ave[1],label='16',color='blue',linewidth=5.0)
ax1.plot(ave[2],label='32',color='yellow',linewidth=5.0)
ax1.plot(ave[3],label='64',color='green',linewidth=5.0)
ax1.plot(ave[4],label='128',color='black',linewidth=5.0)
ax1.plot(ave[5],label='256',color='brown',linewidth=5.0)
ax2.plot(episodeave[0],label='8',color='red',linewidth=5.0)
ax2.plot(episodeave[1],label='16',color='blue',linewidth=5.0)
ax2.plot(episodeave[2],label='32',color='yellow',linewidth=5.0)
ax2.plot(episodeave[3],label='64',color='green',linewidth=5.0)
ax2.plot(episodeave[4],label='128',color='black',linewidth=5.0)
ax2.plot(episodeave[5],label='256',color='brown',linewidth=5.0)"""
"""ax1.ylabel('total_step')
ax1.xlabel('episode')
ax1.title('DeepQNetwork', loc='center')"""
ax1.legend(loc=0)
"""ax2.ylabel('step')
ax2.xlabel('episode')
ax2.title('DeepQNetwork', loc='center')"""
ax2.legend(loc=0)
plt.show()

