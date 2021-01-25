
<br />
  <h3 align="center">Project 2 Continuous Control</h3>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#one-agent-ddpg">One Agent DDPG</a></li>
    <li><a href="#parameter-exploration">Parameter exploration in DDPG</a></li>
    <li><a href="#d4pg">D4PG</a></li>
    <li><a href="#a2c-multi-agent">A2C Multi agent</a></li>
    <li><a href="#conclusion-and-future-improvements">Conclusion and future improvements</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#references">References</a></li>
  </ol>
</details>


## About The Project

 In this project an agent is trained to move an arm robot in a continuous environment. In the Reacher task, there is a ball which the robot arm must follow and touch, in other words, it needs to learn how to control and move a ball around. The longer the arm sticks to the ball, the more rewards it accumulates. The observation state of environment consists of 33 variables, and all are in continuous space. 
 
The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector is a number between -1 and 1. Here is the first difficulty: we are dealing with a continuous space in which small decimals in one of the variables can change the result drastically (from not touching to touching the ball). To see more details about the states, actions and rewards, have a look at the README. 

There are two scenarios, one being single robotic agent, and the other being multiple agents, 20 arms in total, each with their own copy of the environment. In order to be considered solved, the agents must get an average score of +30 (over 100 consecutive episodes, and over all 20 agents). In particular, after each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 20 (potentially different) scores which we average. This yields an average score for each episode (where the average is over all 20 agents). The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30.

## DDPG

In DDPG(1), by sampling on replay memory we are breaking the correlation between transitions. In the implementation provided by Udacity, Ornstein-Uhlenbeck (OU) noise (random walking noise, which grows correlated with the mean moving in a close random direction, instead of being purely gaussian) is added in action values for action exploration. The action is then clipped to be in between (-1,1).

It was not possible to make it work simply by adjusting parameters from other environments like the Bipedal. In this case we need to recopilate a big amount of cases before the network starts to learn. The learning happens at every 40 timesteps, and then there is 10 times a weight-update, each time with a batch of 128 transitions picked randomly.



*Hyperparameters
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

Variability due to random seeds, neural network weight initialization and different machines can make a huge difference.

One of the critical aspects of DDPG is the balance between explotaiton and exploration, which is indirectly affected by one of the critical parameters:the OU noise. Playing with it, specially with the variance, affected a lot the stability of the algorithm. Since the noise is used to explore, an improvement could be to anneal it over the training, to force exploration at the beginning and convergence towards the end. 

 In this implementation, the std. dev. of the noise is decayed after each episode, and this was found to be essential to solving the environment.




<figure>
<img src="images/DQN.png" alt="drawing" style="width:400px;" caption="f"/>
<figcaption><i>Figure 1. Evolution of rewards (score) for the first 500 episodes. The red line is the moving average over the last 100 episodes</i></figcaption>
 </figure>
number of time steps per episode





## D4PG


This work adopts the very successful distributional perspective on reinforcement learning and adapts it to the continuous control setting. We combine this within a distributed framework for off-policy learning in order to develop what we call the Distributed Distributional Deep Deterministic Policy Gradient algorithm, D4PG. We also combine this technique with a number of additional, simple improvements such as the use of N-step returns and prioritized experience replay. Experimentally we examine the contribution of each of these individual components, and show how they interact, as well as their combined contributions. Our results show that across a wide variety of simple control tasks, difficult manipulation tasks, and a set of hard obstacle-based locomotion tasks the D4PG algorithm achieves state of the art performance.



 the D4PG can train on multiple transition trajectory(N-Steps), but I choose to train on one-step for its simplicity. However, according to other reviews, one-step training is the most unstable one and not recoomended, but I still go for it anyway. Two hidden layers are with size 128, 64 each. Buffer memory is with size 100000. Weights are soft-updated. One minor difference from DDPG is the action exploration. In D4PG it uses simple random noise from normal distribution instead of OU noise.


Parameters
* Learning Rate (Actor/Critic): 1e-4
* Batch Size: 64
* Buffer Size: 100 000
* Gamma: 0.99
* Tau: 1e-3
* Repeated Learning per time: 10
* Learning Happened per timestep: 150
* Max Gradient Clipped for Critic: 1
* N-step: 1
* N-Atoms: 51
* Vmax: 10
* Vmin: -10
* Hidden Layer 1 Size: 128
* Hidden Layer 2 Size: 64



## A2C

A typical algorithm for multiple agents case is [Synchronous Advantage Actor Critic (A2C)] is used. Here, each agent has its set of unique transition experiences, it is the collective transition experiences update what solves the problem of sequential correlation. Since it is an on-policy learning algorithms, it is likely that is more stable than off-policy ones (where the sampling of a second policy can slow or bias the training).


Parameters:
* Episodes: 1000
* N-Step: 10
* Learning rate: 0.00015
* GAMMA: 0.99


## Conclusion and future improvements
From the results shown later on, we can clearly tell that A2C significantly outperforms than the rest two algorithms in terms of training speed, which is not surprising, nonetheless the improvement is really impressive. In this case specifically, the A2C successfully trains the agents to accumulate rewards above 30 (the target rewards) in less than 500 training episodes. While D4PG requires roughly 5000 episodes and DDPG even fails to solve the task at all no matter how much time it trains the agent.

For one agent, we used DDPG and D4PG, but it would be interesting to use [PPO](https://openai.com/blog/openai-baselines-ppo/). 


In general, this project provided a very valuable lesson: RL has a long way to go yet. During the study of this part about continuous control, in about 20 or so papers with different continuous RL and its variations, it always seems by the abstract that each algorithm the most robust and the one that solves more efficiently. DDPG proved to be a really hard conditioned problem, very sensitive to random seeds, learning rates (to be specific, the right relationship between actor and critic) and noise injected for exploration. It took me one week to make it work, even with the advice of many people in forums, simply because their recommended parameters did not work for me. At the end it was a combination of all of that what made it run, but the low reproducibility in science is always a concern. 

Thus, even if I consider algorithms like DDPG interesting to learn, I would never use them in a serious project (like an industrial robot), due to its mathematical unstability. DQN with rainbow proved to be quite robust, hence its success and reproduction in hundreds of repositories and blogs. However, in the branch of continuous control I still see a lot of divergence with a huge variety of algorithms aiming towards stability and reproducibility without success. And the excuse that RL is still an art, is not enough for me. I am though, hopeful that we will find better solutions in the future!

## Acknowlegments


## Contact

Jose Miguel Serra Lleti - serrajosemi@gmail.com

Project Link: [https://github.com/josemiserra/navigation_drlnd](https://github.com/josemiserra/navigation_drlnd)


## References
* (1) [DDPG](https://arxiv.org/pdf/1509.02971.pdf)
* ()  [Noisy networks](https://arxiv.org/pdf/1706.10295.pdf)


