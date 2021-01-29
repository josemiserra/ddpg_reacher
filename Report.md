
<br />
  <h3 align="center">Project 2 Continuous Control</h3>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#one-agent-ddpg">One Agent DDPG</a></li>
    <li><a href="#parameter-exploration">Parameter exploration in DDPG</a></li>
    <li><a href="#d4pg-multi-agent">D4PG Multi agent</a></li>
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

The agent is the same that in ddpg_multi_agent.py, except the number of times the agent learns after each update must be bigger (20 times and update every 40). Even of that, it was not possible, as shown in the plot, to make it stable over 30. Variability due to random seeds, neural network weight initialization and different machines can make a huge difference. There was one training that reached over 30, but when I tried to reproduce it, it was not possible to replicate the same growth.

One of the critical aspects of DDPG is the balance between explotaiton and exploration, which is indirectly affected by the OU noise. Playing with it, specially with the variance, affected a lot the stability of the algorithm. Since the noise is used to explore, an improvement could be to anneal it over the training, to force exploration at the beginning and convergence towards the end. 


* Learning rate - Actor: 1e-3
* Learning rate - Critic: 1e-4
* Batch Size: 128
* Buffer Size: 1e6
* Gamma: 0.99
* Tau: 1e-3
* Iterations of learning per update step: 20
* Update gradient networks step: 40
* Architecture Actor :  
* Architecture Critic :
* N-episodes : 



<figure>
<img src="images/DQN.png" alt="drawing" style="width:400px;" caption="f"/>
<figcaption><i>Figure 1. Evolution of rewards (score) for the first 500 episodes. The red line is the moving average over the last 100 episodes</i></figcaption>
 </figure>
number of time steps per episode

Since the environment was not solved but the agent seemed stable enough, I decide to try with a multi-agent environment.
## DDPG Multi-agent
With the same parameters as before

## D4PG

These extensions, which we will detail in this section,
- include a distributional critic update,
- the use of distributed parallel actors, 
- N-step returns 
- and prioritization of the experience replay

The core idea is to replace a single Q-value from the critic with a probability distribution.


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


