[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"


# Project 2: Continuous Control

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#introduction">Usage</a></li>
    <li><a href="#gettingstarted">Getting Started</a></li>
    <li><a href="#Instructions">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>


### Introduction

This project is provided by Udacity and comes from the Unity [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

The environment presents a free empty space where an arm robot lays. The arm is  double-jointed arm able to move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. The location is indicated by a green balloon that moves around the arm. The goal is to maintain the position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1 (continuous space).

For this project, there are two separate versions of the Unity environment:
- The first version contains a single agent.
- The second version contains 20 identical agents, each with its own copy of the environment.  

#### One Agent

The task is episodic, and in order to solve the environment,  the agent must get an average score of +30 over 100 consecutive episodes. It is located in the folder _one_agent_. The environment was close to be solved but not completely (either it was unstable or did not pass the 30 value). However, it can be used to create one stable agent.


#### Multiagent

It is the default folder. Here there is a total of 20 agents, who must get an average score of +30 (over 100 consecutive episodes, and over all agents).  Specifically, 
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent.  This yields 20 (potentially different) scores.  We then take the average of these 20 scores. 
- This yields an **average score** for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. 


### Getting Started

### Getting Started

1. Clone the repo
   ```sh
   git clone https://github.com/josemiserra/ddpg_reacher
   ```
2. If you don't have Anaconda or Miniconda installed, go to [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and install Miniconda in your computer (miniconda is a lightweight version of the Anaconda python environment). 

3. It is recommended that you install your own environment with Conda. Follow the instructions here: [Conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). After that, open an anaconda command prompt or a prompt, and activate your environment.
  ```sh
  activate your-environment
  ```
4. Install the packages present in requirements.txt
   ```sh
   pip install requirements.txt
   pip install mlagents
   ```
5. If you want to use pytorch with CUDA, it is recommmended to go to https://pytorch.org/get-started/locally/ and install pytorch following the instructions there, according to your CUDA installation.

6. Move into the folder of the project, and run jupyter notebook.
   ```sh
    cd jupyter notebook
   ```
   Alternatively you can execute from the python console using the execute_train.py for training, execute_test.py for testing the result of the network.
   ```sh
    python execute_train.py
    python execute_test.py
   ```

If you have a different OS than Windows 64, you can download the environment from one of the links below (provided by Udacity):

- **_Version 1: One (1) Agent_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

- **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
(_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.


### Instructions

There are two environments, independent:
    - In the main folder is the Multi-Agent. After running jupyter notebook in the folder, run the `Continuous_Control - Multiagent.ipynb`. If you want to run it by console, use the `execute_train.py` file.
    - In the folder `one_agent`, you will find the same environment but with one agent. After running the jupyter notebook, follow the instructions in `Continuous_Control.ipynb` to get started with training and testing. You can also train using the `one_agent\execute_train.py`
    
You will not be able to run both environments simultaneously. Just run one or the other.
    
For more info about the algorithms and tests done, read the file Report.md.

## License

Distributed under the MIT License from Udacity Nanodegree. See `LICENSE` for more information.


## Contact

Jose Miguel Serra Lleti - serrajosemi@gmail.com

Project Link: https://github.com/josemiserra/ddpg_reacher

