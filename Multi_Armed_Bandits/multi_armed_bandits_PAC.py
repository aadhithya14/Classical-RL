
# coding: utf-8

# In[3]:


import numpy as np
import random
import matplotlib.pyplot as plt


# In[4]:


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
    
    def get_armhistory(self):
        return self.arm_count
    
    def get_Q_stars(self):
        return self.Q_stars
    
    def reset(self):
        self.arm_history = np.zeros(self.num_arms)
        self.regret = []


# In[5]:


class PAC():
    def __init__(self,epsilon,delta,bandit):
        self.epsilon=epsilon
        self.delta=delta
        self.bandit=bandit
        self.Q=np.zeros(self.bandit.num_arms)
        self.Q_max=[]
        
    def pac(self):
        self.lvalue=int((2/np.square(self.epsilon))*np.log(2*self.bandit.num_arms/self.delta))
        for k in range(self.bandit.num_arms):
            for i in range(self.lvalue):
                arm_count=self.bandit.get_armhistory()
                reward = self.bandit.pull(k)
                self.Q[k]=(self.Q[k]*arm_count[k]+reward)/(arm_count[k]+1)
                self.Q_max.append(np.max(self.Q))
        regret=self.bandit.get_regret()
        return self.Q,self.Q_max,self.bandit.get_best_arm(),self.bandit.get_best_value(),regret
    

   
    
                
        
        
        


# In[6]:


k=5
mean=np.array([1,3,6,2,9])*0.1
variance=[1,1,1,1,1]

bandit=Bandit(mean,variance,k)
Pac=PAC(0.1,0.1,bandit)
(Q,Q_max,best_arm,best_value,regret)=Pac.pac()
Q_stars=bandit.get_Q_stars()




# In[7]:


def accuracy(episodes):
    count=0
    for i in range(episodes):
        (Q,Q_max,best_arm,best_value,regret)=Pac.pac()
        if np.argmax(Q)==np.argmax(Q_stars):
            count+=1
    return count/episodes


# In[8]:


num_iters=10000
accuracy=accuracy(num_iters)
print("the value function of best arm calculated is ",Q)
print("the arm  chosen is ",np.argmax(Q))
print("the true value function of the best arm is",Q_stars)
print("the best arm is",best_arm)
print("accuracy")


# In[9]:


def plot_regret(data, bandit, algo, k):
    #Plots regret of any one bandit at a time
    t = np.arange(5000)
    plt.plot(t, regret, color='blue', label=algo)
    plt.xlabel("Time steps")
    plt.ylabel("Regret")
    plt.legend()
    plt.show()


# In[10]:


plot_regret(regret,"bernoulli_bandit","PAC",k)


# In[47]:


print(accuracy)

