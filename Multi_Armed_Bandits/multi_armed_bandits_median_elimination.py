
# coding: utf-8

# In[2]:


import numpy as np
import random
import matplotlib.pyplot as plt


# In[3]:


class Bandit():
    def __init__(self,mean,variance,k):
        self.Q_stars=np.array(mean)
        self.variance=variance
        self.num_arms=k
        self.best_value=np.max(self.Q_stars)
        self.best_arm=np.argmax(self.Q_stars)
        self.regret=[]
        self.arm_count=np.zeros(k)
    
    def pull(self,arm):
        self.arm_count[arm]=self.arm_count[arm]+1
        self.regret.append(self.best_value-self.Q_stars[arm])
        return np.random.choice([1,0], p = [self.Q_stars[arm], 1-self.Q_stars[arm]])
    
    def get_regret(self):
        return self.regret
    
    def get_best_arm(self):
        return self.best_arm
    
    def get_best_value(self):
        return self.best_value
    
    def get_armcount(self):
        return self.arm_count
    
    def get_Q_stars(self):
        return self.Q_stars
    
    def reset(self):
        self.arm_count = np.zeros(self.num_arms)
        self.regret = []


# In[26]:


class MedianElimination():
    def __init__(self,epsilon,delta,bandit):
        self.epsilon=epsilon
        self.delta=delta
        self.bandit=bandit
        self.Q=np.zeros(self.bandit.num_arms)
        self.S=np.arange(self.bandit.num_arms)
        self.Q_max=[]
    
    def play(self):
        self.epsilon=self.epsilon/4
        self.delta=self.delta/2
        self.l=1
        while len(self.S)!=1:
            self.lvalue=int((2/np.square(self.epsilon)*np.log(3/self.delta)))
            for k in range(self.bandit.num_arms):
                for i in range(self.lvalue):
                    arm_count=self.bandit.get_armcount()
                    reward=self.bandit.pull(k)
                    self.Q[k]=(self.Q[k]*arm_count[k]+reward)/(arm_count[k]+1)
                    self.Q_max.append(np.max(self.Q))
            med=np.median(self.Q[self.S])
            self.S=np.delete(self.S,np.where(self.Q[self.S]<med))
            self.epsilon=(3/4)*self.epsilon
            self.delta=self.delta/2
            self.l=self.l+1
        regret=self.bandit.get_regret()
        return self.Q,self.Q_max,self.bandit.get_best_arm(),self.bandit.get_best_value(),regret
    
                    
            
            
            
                    
                    
            
        
        


# In[27]:


k=5
mean=np.array([1,3,6,2,9])*0.1
variance=[1,1,1,1,1]
bandit=Bandit(mean,variance,k)
memodel=MedianElimination(0.1,0.1,bandit)
(Q,Q_max,best_arm,best_value,regret)=memodel.play()
Q_stars=bandit.get_Q_stars()


# In[28]:


def accuracy(episodes):
    count=0
    for i in range(episodes):
        (Q,Q_max,best_arm,best_value,regret)=memodel.play()
        if np.argmax(Q)==np.argmax(Q_stars):
            count+=1
    return count/episodes

        


# In[29]:


num_iters=10000
accuracy=accuracy(num_iters)
print("the value function of best arm calculated is ",Q)
print("the arm  chosen is ",np.argmax(Q))
print("the true value function of the best arm is",Q_stars)
print("the best arm is",best_arm)
print("accuracy",accuracy)


# In[32]:


def plot_regret(data, bandit, algorithm, k):
    t = np.arange(478820)
    plt.plot(t, regret, color='green', label=algorithm)
    plt.xlabel("Time steps")
    plt.ylabel("Regret")
    plt.legend()
    plt.show()


# In[34]:


plot_regret(regret,"bernoulli_bandit","MedianElimination",k)

