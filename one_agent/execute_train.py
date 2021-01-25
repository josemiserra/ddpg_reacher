from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import torch
from ddpg_agent import Agent


def trainFunction(n_episodes=2000, max_t=1000):
    agent = Agent(state_size=33, action_size=4, seed=0)
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state, i_episode)
            env_info = env.step(action.astype(np.float32))[brain_name]
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished
            if t == max_t-1:
                done = True
            agent.step(state, action, reward, next_state, done, t)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        if i_episode > 50 :
            agent.scheduler_actor.step(score)
            agent.scheduler_critic.step(score)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        # if np.mean(scores_window)>=13.0:

    print('\nEnvironment finished in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
    # torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
    return scores

if __name__ == "__main__":
    env = UnityEnvironment(file_name="Reacher_one.exe")

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

    n_episodes = 500
    max_t = 1000
    scores= trainFunction(n_episodes, max_t)
    print(scores)


