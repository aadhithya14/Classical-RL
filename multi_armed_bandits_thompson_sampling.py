
# coding: utf-8

# In[186]:


import random
import numpy as np
import matplotlib.pyplot as plt


# In[187]:


class Bandit():
    def __init__(self,mean,variance,k):
        self.Q_stars=np.array(mean)
        self.variance=variance
        self.num_arms=k
        self.best_arm=np.argmax(self.Q_stars)
        self.best_value=np.max(self.Q_stars)
        self.arm_count=np.zeros(k)
        self.regret=[]
        
        
    def pull(self,arm):
        self.arm_count[arm]+=1
        self.regret.append(self.best_value-self.Q_stars[arm])
        return np.random.choice([1,0], p = [self.Q_stars[arm], 1-self.Q_stars[arm]])
    
    
    def get_regret(self):
        return self.regret
    
    def get_best_arm(self):
        return self.best_arm
    
    def get_Q_stars(self):
        return self.Q_stars
    
    def get_best_value(self):
        return self.best_value
    
    def get_arm_count(self):
        return self.arm_count
    
    
        


# In[188]:


class thompson():
    def __init__(self,num_iters,bandit):
        self.bandit=bandit
        self.num_iters=num_iters
        self.S=np.zeros(self.bandit.num_arms)
        self.Q_max=[]
        self.Q_star_sample=np.zeros(self.bandit.num_arms)
        self.F=np.zeros(self.bandit.num_arms)
        self.theta=np.zeros(self.bandit.num_arms)
    """def play(self):
        for i in range(self.num_iters):
            Q_stars=self.bandit.get_Q_stars()
            for i in range(self.bandit.num_arms):
                self.Q_star_sample[i]=random.choice(Q_stars[i])
            arm=np.argmax(self.Q_star_sample)
            reward=self.bandit.pull(arm)
            arm_count=self.bandit.get_arm_count()
            self.Q[arm]=(self.Q[arm]*arm_count[arm]+reward)/(arm_count[arm]+1)
            Q_stars[arm]=self.Q[arm]
        regret=self.bandit.get_regret()
        return Q_stars,self.Q,
    
"""
    def select_arm(self):
        arm=np.argmax(self.S)
        return arm
    
    def bernoulli(self):
        for i in range(self.bandit.num_arms):
            self.S[i]=0
            self.F[i]=0
        for i in range(self.bandit.num_arms):
            self.S[i]=self.bandit.pull(i)
            self.F[i]=1-self.bandit.pull(i)
        for t in range(num_iters):
            arm=self.select_arm()
            reward=self.bandit.pull(arm)
            arm_count=self.bandit.get_arm_count()
            self.S[arm]=(self.S[arm]*arm_count[arm]+reward)/(arm_count[arm]+1)
            self.F[arm]=(self.F[arm]*arm_count[arm]+1-reward)/(arm_count[arm]+1)
        regret=self.bandit.get_regret()
        return self.S,self.F,regret
    
    def Thompson_sampling(self,S,F):
        for i in range(self.bandit.num_arms):
            self.theta[i]=np.random.beta(S[i]+1,F[i]+1)
        return np.argmax(self.theta)
    
            
            
        


# In[189]:


k=5
mean=np.array([1,3,6,2,9])*0.1
variance=[1,1,1,1,1]
num_iters=4995
bandit=Bandit(mean,variance,k)
thompson=thompson(num_iters,bandit)
(S,F,regret)=thompson.bernoulli()
arm=thompson.Thompson_sampling(S,F)
Q_stars=bandit.get_Q_stars()


# In[190]:


def accuracy(episodes):
    count=0
    for i in range(episodes):
        arm=thompson.Thompson_sampling(S,F)
        if arm==np.argmax(Q_stars):
            count+=1
    return count/episodes


# In[191]:


num_iters=10000
accuracy=accuracy(num_iters)
print("the value function of best arm calculated is ",S)
print("the arm  chosen is ",np.argmax(S))
print("the true value function of the best arm is",Q_stars)
print("the best arm is",np.argmax(Q_stars))


# In[192]:


print(accuracy)

