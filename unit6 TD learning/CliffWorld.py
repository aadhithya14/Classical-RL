
# coding: utf-8

# In[1]:


import random
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


rows=4
columns=12


# In[3]:


def createQtable(rows,columns):
    #This function creates the action value table with rows and columns as input
    qtable=np.zeros((4,rows*columns))
    action_dict={"UP":qtable[0,:],"Left":qtable[1,:],"Right":qtable[2,:],"Down":qtable[3,:]}
    
    return qtable
    
    


# In[4]:


def agent_pos_after_action(agent,action):
    #gives the position of the agent after taking a certain action in a state
    (posX,posY)=agent
    if action==0 and posX>0:
        posX=posX-1
    if action==1 and posY>0:
        posY=posY-1
    if action==2 and posY<11:
        posY=posY+1
    if action==3 and posX<3:
        posX=posX+1
    agent=(posX,posY)
    return agent


# In[5]:


def epsilon_greedy(state,qtable,epsilon=0.1):
    #selecting an action based on epsilon greedy approach
    sample=np.random.random()
    probs=np.zeros(4)
    if sample<epsilon:
        action=np.random.choice(4)
        probs[action]=epsilon#calculate the probabilty of each action
    else:
        action=np.argmax(qtable[:,state])
        probs[action]=1-epsilon
    return action,probs


# In[6]:


def get_state(agent,qtable):
    #calculate the state  of an agent in the qtable formed  
    (posX,posY)=agent
    state=12*posX+posY
    state_value=qtable[:,state]
    #calculate the max Qvalue over all actions for a particular state
    state_value_of_max_action=state_value.max()
    
    return state,state_value_of_max_action

    


# In[7]:


def get_reward(state):
    if state==47:
        reward=10
        done=True
    elif state>=37 and state<=46:
        reward=-100
        done=True
    else:
        reward=-1
        done=False
     
    return reward,done
        
        


# In[ ]:



    
    
    
    


# In[9]:


def Qlearning(num_episodes=500,gamma=1.0,alpha=0.5):
    #function for qlearning update
    qtable=createQtable(rows,columns)
    agent=(3,0)
    reward_list=[]
    step_list=[]
    for i in range(num_episodes):
        env=np.zeros((4,12))
        env=visited_env(agent,env)
        done=False
        reward_cum=0
        step_cum=0
        while not done:
            (state,_)=get_state(agent,qtable)
            action,probs=epsilon_greedy(state,qtable)
            agent=agent_pos_after_action(agent,action)
            step_cum+=1
            env=visited_env(agent,env)
            next_state, max_next_state_value = get_state(agent, qtable)
            reward,done=get_reward(next_state)
            reward_cum+=reward
            td_error=reward+gamma*max_next_state_value-qtable[action,state]
            qtable[action,state]+=alpha*td_error
            state=next_state
        reward_list.append(reward_cum)
        step_list.append(step_cum)
        
    return qtable,reward_list,step_list
            
            
            
            
            
            
    
    


# In[10]:


def visited_env(agent, env):
    """
        Visualize the path agent takes
        
    """
    (posY, posX) = agent
    env[posY][posX] = 1
    return env


# In[11]:


def sarsa(num_episodes=500,alpha=0.5,gamma=1.0):
    #function for sarsa update
    reward_list=[]
    step_list=[]
    qtable=createQtable(rows,columns)
    agent=(3,0)
    for i in range(num_episodes):
        env=np.zeros((4,12))
        env=visited_env(agent,env)
        cum_reward=0
        cum_steps=0
        done=False
        (state,_)=get_state(agent,qtable)
        action,probs=epsilon_greedy(state,qtable)
        while not done:
            agent=agent_pos_after_action(agent,action)
            cum_steps+=1
            (nextstate,state_value_of_best_action)=get_state(agent,qtable)
            reward,done=get_reward(nextstate)
            nextaction,probs=epsilon_greedy(nextstate,qtable)
            qtable[action,state]=qtable[action,state]+alpha*(reward+gamma*qtable[nextaction,nextstate]-qtable[action,state])
            state=nextstate
            action=nextaction
            cum_reward+=reward
        reward_list.append(cum_reward)
        step_list.append(cum_steps)

    return qtable,reward_list,step_list
            
            
            
            
            
            
            
    


# In[12]:


def Expected_Sarsa(num_episodes=500,alpha=0.5,gamma=1.0):
    #function for expected sarsa update
    qtable=createQtable(rows,columns)
    agent=(3,0)
    reward_list=[]
    step_list=[]
    for i in range(num_episodes):
        env=np.zeros((4,12))
        env=visited_env(agent,env)
        done=False
        reward_cum=0
        step_cum=0
        while not done:
            (state,_)=get_state(agent,qtable)
            action,probs=epsilon_greedy(state,qtable)
            agent=agent_pos_after_action(agent,action)
            step_cum+=1
            env=visited_env(agent,env)
            nextstate, max_next_state_value = get_state(agent, qtable)
            reward,done=get_reward(nextstate)
            reward_cum+=reward
            for a in range(4):
                qtable[action,state]=qtable[action,state]+alpha*(reward+probs[a]*(qtable[a,nextstate])-qtable[action,state])
            state=nextstate
        reward_list.append(reward_cum)
        step_list.append(step_cum)
    return qtable,reward_list,step_list
            
    


# In[17]:


def normalised_reward_distribution(reward_list_q,reward_list_sarsa,reward_list_expected_sarsa):
    #plots the cummulative normalised rewards across episodes
    normalised_list_q=[]
    reward_std=np.array(reward_list_q).std()
    reward_mean=np.array(reward_list_q).mean()
    count=0
    cum_reward=0
    for reward in reward_list_q:
        count=count+1
        cum_reward+=reward
        if count==10:
            normalised_reward=(cum_reward-reward_mean)/reward_std
            count=0
            cum_reward=0
            normalised_list_q.append(normalised_reward)
            
    normalised_list_sarsa=[]
    reward_std=np.array(reward_list_sarsa).std()
    reward_mean=np.array(reward_list_sarsa).mean()
    count=0
    cum_reward=0
    for reward in reward_list_sarsa:
        count=count+1
        cum_reward+=reward
        if count==10:
            normalised_reward=(cum_reward-reward_mean)/reward_std
            count=0
            cum_reward=0
            normalised_list_sarsa.append(normalised_reward)
    
    normalised_list_Expected_sarsa=[]
    reward_std=np.array(reward_list_expected_sarsa).std()
    reward_mean=np.array(reward_list_expected_sarsa).mean()
    count=0
    cum_reward=0
    for reward in reward_list_expected_sarsa:
        count=count+1
        cum_reward+=reward
        if count==10:
            normalised_reward=(cum_reward-reward_mean)/reward_std
            count=0
            cum_reward=0
            normalised_list_Expected_sarsa.append(normalised_reward)
    
    plt.subplot(131)
    plt.plot(normalised_list_q,label='qlearning')
    plt.ylabel('Cumulative Rewards of batch of episodes')
    plt.xlabel('Batch of episodes batch_size=10')
    plt.title('Qlearning')
    plt.subplot(132)
    plt.plot(normalised_list_sarsa,label='sarsa')
    plt.title('sarsa')
    plt.subplot(133)
    plt.plot(normalised_list_Expected_sarsa,label='Expectedsarsa')
    plt.title('Expected sarsa')
    plt.show()
    
        


# In[18]:


"""def plot_cum_reward(reward_list_q,reward_list_sarsa):
    plt.plot(reward_list_q,label='qlearning')
    plt.plot(reward_list_sarsa,label='sarsa')
    plt.ylabel('sum of rewards during epsiode')
    plt.xlabel('episodes')
    plt.title('Cummulative rewards')
    plt.legend()
    plt.show()"""


# In[19]:


def main():
    qtable_SARSA, reward_cache_SARSA, step_cache_SARSA = sarsa()
    qtable_qlearning, reward_cache_qlearning, step_cache_qlearning = Qlearning()
    qtable_expected_sarsa, reward_cache_expected_sarsa, step_cache_expected_sarsa = Expected_Sarsa()
    normalised_reward_distribution(reward_cache_qlearning,reward_cache_SARSA,reward_cache_expected_sarsa)
    #plot_cum_reward(reward_cache_qlearning,reward_cache_SARSA)
    


# In[20]:


if __name__=="__main__":
    main()

