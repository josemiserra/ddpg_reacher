from unityagents import UnityEnvironment
import numpy as np
import torch
from dqn_agent import Agent


def testFunction():
    agent = Agent(state_size=37, action_size=4, seed=0)
    agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    state = env_info.vector_observations[0]  # get the current state
    score = 0  # initialize the score
    time_steps = 100000
    for t in range(time_steps):
        action = agent.act(state)  # select an action
        env_info = env.step(action.astype(np.int32))[brain_name]  # send the action to the environment
        next_state = env_info.vector_observations[0]  # get the next state
        reward = env_info.rewards[0]  # get the reward
        done = env_info.local_done[0]  # see if episode has finished
        score += reward  # update the score
        state = next_state  # roll over the state to next time step
        if done:  # exit loop if episode finished
            break

    print("Score: {}".format(score))

if __name__ == "__main__":
    env = UnityEnvironment(file_name="Banana.exe")

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    # examine the state space
    state = env_info.vector_observations[0]
    print('States look like:', state)
    state_size = len(state)

    print('States have length:', state_size)

    testFunction()



