import matplotlib.pyplot as plt
import numpy as np
np.random.seed(0)
num = np.random.rand()
print(num)
episode10 = []
episode5 = []
epiql = []
ore = []
#episode.append(0)
"""for i in range(100):
    episode10.append(1-(i * 0.01))
"""
"""
for i in range(1,100):
    episode5.append(1/i)
for i in range(100):
    a = max(1-(i*0.05),0.01)
    ore.append(a)
"""
for i in range(100):
    epiql.append(0.5 * (1 / (i + 1)))

#print(epiql)
#plt.plot(episode10,label='x',marker="x",linestyle='None')
#plt.plot(episode5,label='y',marker='.',linestyle='None')
plt.plot(epiql,label='z')
#plt.plot(ore,label='o')
plt.ylabel('epsilon')
plt.xlabel('episode')
plt.ylim([0,1])
plt.title('Epsilon', loc='center')
plt.show()