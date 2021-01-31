from unityagents import UnityEnvironment
import numpy as np
from collections import deque
from ddpg_multi_agent import Agent


def trainFunction(n_episodes=2000, num_agents = 20):
    agent = Agent(state_size=33, action_size=4, seed=37)
    avg_scores = []
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = 1.0
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        states = env_info.vector_observations
        scores = np.zeros(num_agents)
        while True:
            actions =[]
            for an_agent in range(num_agents):
                agent.reset()
                actions.append(agent.act(states[an_agent],eps))
            eps *=0.99
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations  # get the next state
            rewards = env_info.rewards  # get the reward
            dones = env_info.local_done  # see if episode has finished
            for an_agent in range(num_agents):
                agent.step(states[an_agent], actions[an_agent], rewards[an_agent], next_states[an_agent], dones[an_agent])
            states = next_states
            scores += rewards
            if np.any(dones):
                break
        scores_window.append(scores.mean())  # save most recent score
        avg_scores.append(scores.mean())  # save most recent score

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        # if np.mean(scores_window)>=13.0:

    print('\nEnvironment finished in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

    return scores

if __name__ == "__main__":
    env = UnityEnvironment(file_name="Reacher.exe")

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

    n_episodes = 150
    scores = trainFunction(n_episodes)
    print(scores)


