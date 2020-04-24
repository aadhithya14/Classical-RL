
# coding: utf-8

# In[60]:


import numpy as np
import matplotlib.pyplot as plt
import random
import scipy


# In[61]:


class K_armed_bandit():
    def __init__(self,mean,variance):
        self.mean=mean
        self.variance=variance
    
    def sample(self):
        return random.gauss(self.mean,self.variance)
    


# In[76]:


def K_arms(k):
    Q_stars=np.zeros(k)
    k_arms=[]
    actions=[]
    for i in range(k):
        k_arms.append(K_armed_bandit(random.uniform(-5,5),random.uniform(-5,5)))
        Q_stars[i]=k_arms[i].mean
        actions.append(i)
    return Q_stars,k_arms,actions

k=8
(Q_stars,k_arms,actions) = K_arms(k)
        


# In[77]:


def epsilon_greedy(Q_value,epsilon,k):
    r=random.random()
    if r>epsilon:
        action=random.randint(0,k-1)
    else:
        action=np.argmax(Q_value)
    return int(action)
        


# In[78]:



def policy(n):
    identifier_func=np.zeros(k,np.uint8)
    Q=np.zeros(k)
    Q_max=[]
    epsilon=1
    for i in range(1,n+1):
        action=epsilon_greedy(Q,epsilon,k)
        reward=k_arms[action].sample()
        Q[action]=(Q[action]*identifier_func[action]+reward)/(identifier_func[action]+1)
        identifier_func[action]+=1
        if i>n/10:
            epsilon=1/i
        Q_max.append(np.max(Q))
    return Q,Q_max

        
        
        


# In[79]:


n=1000
(Q,Q_max)=policy(n)
def accuracy(episodes):
    count=0
    for i in range(episodes):
        (Q,Q_max)=policy(n)
        if np.argmax(Q)==np.argmax(Q_stars):
            count=count+1
    return count/episodes


# In[83]:


def plot(Q_max,time,policy,QstarMax):
    n=np.arange(time)
    plt.plot(n,Q_max,label = "expected payoff of current estimated best arm")
    plt.plot([1,time], [QstarMax,QstarMax], "--y", label = "Exected payoff of best arm")
    plt.ylabel('maximum expected reward')
    plt.xlabel('time steps')
    plt.title(policy)
    plt.legend()
    plt.show()


# In[81]:


num_iters=10000
accuracy=accuracy(num_iters)
print("the value function of best arm calculated is ",Q)
print("the arm  chosen is ",np.argmax(Q))
print("the true value function of the best arm is",Q_stars)
print("the best arm is",np.argmax(Q_stars))


# In[82]:


print(accuracy)


# In[87]:


plot(Q_max,n, "epsilon_greedy policy", np.max(Q_stars))

