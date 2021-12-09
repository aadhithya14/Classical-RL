

# TD LEARNING

It is a sample based algorithm which uses features of both Monte-Carlo and Dynamic Programming methods. Like Monte Carlo methods, TD methods can learn directly from raw experience without a model of the environment’s dynamics. Like DP, TD methods update estimates based in part on other learned estimates, without waiting for a final outcome (they bootstrap).

Sources of topics are:-

[Balaraman Ravindran's lectures](https://nptel.ac.in/courses/106106143/) 

[Sutton and Barto chapter 6](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)

[Google deepmind lectures by David Silver.](https://www.youtube.com/watch?v=0g4j2k_Ggc4&list=PLqYmG7hTraZBiG_XpjnPrSNw-1XQaM_gB&index=5)

## CLIFFWORLD 
This is a standard un-discounted, episodic task, with start and goal states, and the usual actions causing movement up, down, right, and left. Reward is -1 on all transitions except those into the region marked Cliff. Stepping into this region incurs a reward of optimal path -100 and sends the agent instantly back to the start.The environment looks like below

![alt text](https://github.com/aadhithya14/RLResearch/blob/master/TDLearning/Results/Cliff.png)


I have implemented the following algorithms

- [x] SARSA in CliffWorld
- [x] Qlearning in CliffWorld
- [x] Expected SARSA in CliffWorld


### RESULTS:

![alt text](https://github.com/aadhithya14/RLResearch/blob/master/TDLearning/Results/Results.png)

[Check out my notes on this unit](https://hackmd.io/CWNwEj-IR7eq5Nh6vUSefQ?both)

