
# coding: utf-8

# In[49]:



import numpy as np
import random
import scipy
import matplotlib.pyplot as plt


# In[50]:


class K_armed_bandit():
    def __init__(self,mean,variance):
        self.mean=mean
        self.variance=variance
    def sample(self):
        return random.gauss(self.mean,self.variance)
    


# In[66]:


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
        


# In[60]:


def softmax_select_action(Q_value,beta=1000):
    prob=np.copy(np.exp(Q_value/beta)/np.sum(np.exp(Q_value/beta)))
    action =np.random.choice(actions, p = prob)
    return int(action)

def softmax(iters,beta=1000):
    identifier_func=np.zeros(k)
    Q=np.zeros(k)
    Q_max=[]
    for i in range(iters):
        action=softmax_select_action(Q)
        reward=k_arms[action].sample()
        identifier_func[action]+=1
        Q[action]=(Q[action]*identifier_func[action]+reward)/(identifier_func[action]+1)
        Q_max.append(np.max(Q))
    return Q,Q_max
        
        
        
        


# In[61]:


reward=0
iters=1000
(Q,Q_max)=softmax(iters)

def accuracy(episodes):
    count=0
    for i in range(episodes):
        (Q,Q_max)=softmax(iters)
        if np.argmax(Q)==np.argmax(Q_stars):
            count=count+1
    return count/episodes

            
            
        
    
    
    


# In[62]:


def plot(Q_max,time,policy,QstarMax):
    t=np.arange(time)
    plt.plot(t,Q_max,label = "expected payoff of current estimated best arm")
    plt.plot([1,time], [QstarMax,QstarMax], "--y", label = "Exected payoff of best arm")
    plt.ylabel('maximum expected reward')
    plt.xlabel('time steps')
    plt.title(policy)
    plt.legend()
    plt.show()
    
    
    


# In[63]:




num_iters=10000
accuracy=accuracy(num_iters)
print("the value function of best arm calculated is ",Q)
print("the arm  chosen is ",np.argmax(Q))
print("the true value function of the best arm is",Q_stars)
print("the best arm is",np.argmax(Q_stars))
print("accuracy")


# In[64]:


print(accuracy)


# In[65]:


plot(Q_max,iters, "softmax policy", np.max(Q_stars))

