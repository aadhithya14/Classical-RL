
# coding: utf-8

# In[86]:


import numpy as np
import matplotlib.pyplot as plt
import random


# In[87]:


def value_iteration(p_h,gamma=1,theta=0.0001):
    V=np.zeros(101)
    reward=np.zeros(101)
    reward[100]=1
    while True:
        delta=0
        for s in range(1,100):
            actions=range(1,min(s,100-s)+1)
            A=np.zeros(101)
            for a in (actions):
                A[a]=p_h*(reward[s+a]+gamma*(V[s+a]))+(1-p_h)*(reward[s-a]+gamma*V[s-a])
            best_action=np.max(A)
            delta=max( delta,np.abs(best_action-V[s]) )
            V[s]=best_action
        if delta<theta:
            break
    pi=np.zeros(100)
    for s in range(1,100):
        actions=range(1,min(s,100-s)+1)
        B=np.zeros(101)
        for a in (actions):
            B[a]=p_h*(reward[s+a]+gamma*(V[s+a]))+(1-p_h)*(reward[s-a]+gamma*V[s-a])
        pi[s]=np.argmax(B)
    return pi,V

# In[88]:


(pi,V)=value_iteration(0.25)
#(pi,V)=value_iteration(0.65) uncomment it when the probability is 0.65 


# In[89]:


print(pi)


# In[96]:


def plot_value():
    x=range(100)
    y=V[:100]
    plt.plot(x,y)
    plt.title('Value estimates vs Capital')
    plt.xlabel('Capital')
    plt.ylabel('value estimates')
    plt.show()


# In[97]:


plot_value()


# In[106]:


def plot_policy():
    x=range(100)
    y=pi
    plt.bar(x,y,align='center',alpha=0.5)
    plt.title('Final Policy')
    plt.xlabel('Capital')
    plt.ylabel('Policy')
    plt.show()


# In[107]:


plot_policy()

