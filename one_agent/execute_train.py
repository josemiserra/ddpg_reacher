from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import torch
from ddpg_agent import Agent

I_PATIENCE = 5

def trainFunction(n_episodes=2000):
    agent = Agent(state_size=33, action_size=4, seed=7)
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    old_v = 0
    update_noise = False
    patience = 0
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        agent.reset()
        state = env_info.vector_observations[0]
        score = 0
        while True:
            action = agent.act(state, i_episode, update_noise)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished
            agent.step(state, action, reward, next_state, done, i_episode)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        if i_episode > 100 and i_episode < 500:
            agent.scheduler_actor.step(score)
            agent.scheduler_critic.step(score)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        new_v = float('{0:.2f}'.format(np.mean(scores_window)))
        if old_v > new_v:
            patience+=1
            if patience > I_PATIENCE:
                update_noise = True
                patience = 0
        else:
            update_noise = False
            patience = patience-1 if patience > 0 else 0
        old_v = new_v


    print('\nEnvironment finished in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
    torch.save(agent.local_actor.state_dict(), 'checkpoint_local_actor.pth')           # save local actor
    torch.save(agent.target_actor.state_dict(), 'checkpoint_target_actor.pth')         # save target actor
    torch.save(agent.local_critic.state_dict(), 'checkpoint_local_critic.pth')         # save local critic
    torch.save(agent.target_critic.state_dict(), 'checkpoint_target_critic.pth')       # target critic
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

    n_episodes = 3000
    scores= trainFunction(n_episodes)
    print(scores)


