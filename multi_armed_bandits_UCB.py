
# coding: utf-8

# In[1]:


import numpy as np
import random
import matplotlib.pyplot as plt


# In[5]:


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

    
    


# In[6]:


class UCB():
    def __init__(self,num_iters,bandit):
        self.num_iters=num_iters
        self.time=0
        self.bandit=bandit
        self.Q=np.zeros(self.bandit.num_arms)
        self.Q_max=[]
        self.confidence=np.zeros(self.bandit.num_arms)
      
        
    def select_arm(self):
        arm=np.argmax(np.add(self.Q,self.confidence))
        return arm
    
    def ucb(self):
        self.bandit.reset()
        for i in range(self.bandit.num_arms):
            self.Q[i]=self.bandit.pull(i) 
            self.time+=1
        for n in range(self.num_iters):
            arm_count=self.bandit.get_armhistory()
            arm=self.select_arm()
            self.Q[arm]=(self.Q[arm]*arm_count[arm]+self.bandit.pull(arm))/(arm_count[arm]+1)
            self.confidence=np.sqrt(2*np.log(self.time)/(arm_count[arm]))
            self.Q_max.append(np.max(self.Q))
        regret=self.bandit.get_regret()
        return self.Q,self.Q_max,self.bandit.get_best_arm(),self.bandit.get_best_value(),regret
    


# In[7]:


"""def K_arms(k):
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

        """
k=5
mean=np.array([1,3,6,2,9])*0.1
variance=[1,1,1,1,1]
num_iters=4995
bandit=Bandit(mean,variance,k)
ucb_model=UCB(num_iters,bandit)
(Q,Q_max,best_arm,best_value,regret)=ucb_model.ucb()
Q_stars=bandit.get_Q_stars()




# In[8]:


"""num_iters=1000
def ucb_select_action(Q_value,k):
        arm=np.argmax(np.add(Q_value,confidence))
        return arm
def ucb_value(num_iters):
    Q=np.zeros(k)
    Q_max=[]
    identifier_func=np.zeros(k)
    for arm in range(k):
        reward=k_arms[arm].pull()
        arm_count[arm]+=1
        Q[arm]=(Q[arm]*identifier_func[arm]+reward)/(identifier_func[arm]+1)
    arm=np.argmax(np.max(Q))
    for i in range(1,num_iters+1):
        reward=k_arms[arm].sample()
        identifier_func[arm]+=1
        Q[arm]=(Q[arm]*identifier_func[arm]+reward)/(identifier_func[arm]+1) 
        confidence=np.sqrt(2*np.log())
        Q_max.append(np.max(Q))
        
    return Q,Q_max
"""
        
    
        
        


# In[ ]:





# In[9]:


def accuracy(episodes):
    count=0
    for i in range(episodes):
        (Q,Q_max,best_arm,best_value,regret)=ucb_model.ucb()
        if np.argmax(Q)==np.argmax(Q_stars):
            count+=1
    return count/episodes

        


# In[12]:


num_iters=10000
accuracy=accuracy(num_iters)
print("the value function of best arm calculated is ",Q)
print("the arm  chosen is ",np.argmax(Q))
print("the true value function of the best arm is",Q_stars)
print("the best arm is",best_arm)
print("accuracy",accuracy)


# In[13]:


print(accuracy)


# In[14]:


def plot_regret(data, iters, bandit, player, k):
    #Plots regret of any one bandit at a time
    t = np.arange(5000)
    plt.plot(t, regret, color='green', label=player)
    plt.xlabel("Time steps")
    plt.ylabel("Regret")
    plt.legend()
    plt.show()


# In[15]:


plot_regret(regret,5000,"bernoulli_bandit","UCB",k)

