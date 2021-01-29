import numpy as np
import random
import copy

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

from replay_buffer import ReplayBuffer
from collections import namedtuple, deque

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128       # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3       # learning rate of the actor
LR_CRITIC = 1e-3         # learning rate of the critic
UPDATE_EVERY = 20
LEARN_N = 10
N_STEP = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
eps = np.finfo(np.float32).eps

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
        self.scheduler_actor = optim.lr_scheduler.ReduceLROnPlateau(self.actor_optimizer, 'max',
                                             patience=20,
                                             factor=0.95, verbose=True)

        self.complete_update(self.actor_local, self.actor_target)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, seed).to(device)
        self.critic_target = Critic(state_size, action_size, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)
        self.scheduler_critic = optim.lr_scheduler.ReduceLROnPlateau(self.critic_optimizer, 'max',
                                             patience=10,
                                             factor=0.95, verbose=True)
        self.complete_update(self.critic_local, self.critic_target)
        # Noise process
        self.noise = OUNoise(action_size, seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.frame_step = 0
        self.rewards_queue=deque(maxlen=N_STEP)
        self.states_queue=deque(maxlen=N_STEP)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done, episode):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        # self.memory.add(state, action, reward, next_state, done)

        self.states_queue.appendleft([state, action])
        self.rewards_queue.appendleft(reward * GAMMA ** (N_STEP))
        for i in range(len(self.rewards_queue)):
            self.rewards_queue[i] = self.rewards_queue[i] / GAMMA

        if len(self.rewards_queue) >= N_STEP:
            last = self.states_queue.pop()
            self.memory.add(last[0], last[1], sum(self.rewards_queue), next_state, done)
            self.rewards_queue.pop()

        if done:
            self.states_queue.clear()
            self.rewards_queue.clear()


        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # Learn, if enough samples are available in memory
            if len(self.memory) > BATCH_SIZE:
                for _ in range(LEARN_N):
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA, episode)

    def act(self, state, episode, update_noise = False,
            add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state.unsqueeze(0)).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample(episode, update_noise)
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma, episode):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """

        states, actions, rewards, next_states, dones = experiences
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)


        # Minimize the loss
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1.0)
        self.critic_optimizer.step()
        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        if episode % LEARN_N == 0:
            self.soft_update(self.critic_local, self.critic_target, 2e-1)
            self.soft_update(self.actor_local, self.actor_target, 2e-1)
        else:
            self.soft_update(self.critic_local, self.critic_target, TAU)
            self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def complete_update(self,local_model, target_model):
        """Copy source network weights to target"""
        for target_param, local_param in zip(target_model.parameters(),local_model.parameters()):
            target_param.data.copy_(local_param.data)


class OUNoise:
    """Ornstein-Uhlenbeck process."""


    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.05): # default was 0.2
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()
        self.episodes_to_end = 200
        self.dt = 1.0
        self.var_up = 0.001

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self, current_episode, update_noise = False):
        """Update internal state and return it as a noise sample."""
        x = self.state
        if update_noise:
            dtt = max(0.001, 1.-(current_episode/ self.episodes_to_end))
            if dtt == 0.001:
                self.episodes_to_end = (current_episode + self.episodes_to_end) + self.episodes_to_end * np.random.random()
                self.dt = max(0.001, 1. - (current_episode / self.episodes_to_end))
            self.dt = dtt
            # print("Annealing noise :"+str(self.dt))
        dx = self.theta * (self.mu - x)+ self.sigma*np.array([random.random() for _ in range(len(x))])
        dx = dx*self.dt
        self.state = x + dx
        return self.state


