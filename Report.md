


#Reacher 

In Reacher task, the robotic arm needs to learn how to control and move a ball around. The longer time it sticks to the ball and controls it, the more rewards it accumulates. The observation state of environment consists of 33 variables, and all are in continuous space.


The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector is a number between -1 and 1.

Respecting the robotic arm, there are two scenarios, one being single robotic agent, and the other being multiple robotic agents, which are 20 agents in total, each with their own copy of environment. As we now have both single and multiple agents scenarios, we can then explore and compare the learning efficiency between the two.
Each agent has 4 action variables, all in continuous space within -1 and 1.

In these models, the practice of sampling on replay memory breaks up the correlation between transitions.

DDPG
D4PG


The Reacher environment used contains 20 identical agents, each with its own copy of the environment. In order to be considered solved, the agents must get an average score of +30 (over 100 consecutive episodes, and over all 20 agents). In particular, after each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 20 (potentially different) scores. We then take the average of these 20 scores. This yields an average score for each episode (where the average is over all 20 agents). The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30.

In multiple agents case, [Synchronous Advantage Actor Critic (A2C)] is used. A2C is a synchronous version of [A3C]. A2C is easier to be implemented and in some studies, its performance is even better than A3C. In the case of training on multiple agents, for each own has its unique transition experience, the collective transition experiences will resolve the problem of sequential correlation by nature. Furthermore, we can even enjoy the training stability brought about by on-policy learning algorithms.

A3C
A2C

So, what the target of this task? In the task, if the agent’s arm is in the goal location, then a reward of +0.1 is given. The longer time the agent is able to maintain its arm in the goal location, the more rewards it accumulates. The agent needs to be able to earns rewards above 30 in one episode averagely in order to solve the task.


From the results shown later on, we can clearly tell that A2C significantly outperforms than the rest two algorithms in terms of training speed, which is not surprising, nonetheless the improvement is really impressive. In this case specifically, the A2C successfully trains the agents to accumulate rewards above 30 (the target rewards) in less than 500 training episodes. While D4PG requires roughly 5000 episodes and DDPG even fails to solve the task at all no matter how much time it trains the agent.





2. DDPG paper (https://arxiv.org/pdf/1509.02971.pdf)



It would be interesting to solve the environment with PPO

The Ornstein-Uhlenbeck noise is added in action values for action exploration.





We know that when training based on single agent, the sequence of trasition experiences will be correlated, so that off-policy such as DDPG/D4PG will be more suitable in this case, the practice of sampling on replay memory breaking up the correlation between transitions.

 in the case of training on multiple agents, for each own unique transition experience, the collective transitions will resolve the problem of correlation. Furthermore, we can leverage the training stability brough on by the on-policy learning algorithm.

DDPG


Hyperparameters
Learning Rate (Actor/Critic): 1e-4
Weight Decay: 1e-2
Batch Size: 64
Buffer Size: 100000
Gamma: 0.99
Tau: 1e-3
Repeated Learning per time: 10
Learning Happened per timestep: 20
Max Gradient Clipped for Critic: 1
Hidden Layer 1 Size: 128
Hidden Layer 2 Size: 64


In this experiment, I set up the training happens at every 20 timesteps. Weight-update will iterate for 10 times in each training. Finally, the critic gradient is clipped with maximum value as 1 to enhance the training stability. As you will see below, the episodic reward lingers around 0.04 to 0.05 in the first 1000 episodes, which pretty means the agent not able to learn from these experiences at all.

To encourage early exploration an Ornstein-Uhlenbeck (OU) process is used to add noise to the actions, which is then clipped to be in the (-1,1) range. An OU process is a type of auto-correlated noise process, where noise from past observations is compounded with new random noise, with a mean-reverting trajectory.


Ornstein-Uhlenbeck

As with other techniques such as epsilon-greedy exploration in DQN, it can be useful to anneal the noise over the course of training. This encourages random exploration of the state space early on and better convergence later on. In this implementation, the std. dev. of the noise is decayed after each episode, and this was found to be essential to solving the environment.


D4PG

This work adopts the very successful distributional perspective on reinforcement learning and adapts it to the continuous control setting. We combine this within a distributed framework for off-policy learning in order to develop what we call the Distributed Distributional Deep Deterministic Policy Gradient algorithm, D4PG. We also combine this technique with a number of additional, simple improvements such as the use of N-step returns and prioritized experience replay. Experimentally we examine the contribution of each of these individual components, and show how they interact, as well as their combined contributions. Our results show that across a wide variety of simple control tasks, difficult manipulation tasks, and a set of hard obstacle-based locomotion tasks the D4PG algorithm achieves state of the art performance.


Hyperparameters
Learning Rate (Actor/Critic): 1e-4
Batch Size: 64
Buffer Size: 100000
Gamma: 0.99
Tau: 1e-3
Repeated Learning per time: 10
Learning Happened per timestep: 150
Max Gradient Clipped for Critic: 1
N-step: 1
N-Atoms: 51
Vmax: 10
Vmin: -10
Hidden Layer 1 Size: 128
Hidden Layer 2 Size: 64

 the D4PG can train on multiple transition trajectory(N-Steps), but I choose to train on one-step for its simplicity. However, according to other reviews, one-step training is the most unstable one and not recoomended, but I still go for it anyway. Two hidden layers are with size 128, 64 each. Buffer memory is with size 100000. Weights are soft-updated. One minor difference from DDPG is the action exploration. In D4PG it uses simple random noise from normal distribution instead of OU noise.


 A2C

 
Hyper-parameters
Number of learning episode: 1000
Number of N-Step: 10
Learning rate: 0.00015
GAMMA: 0.99