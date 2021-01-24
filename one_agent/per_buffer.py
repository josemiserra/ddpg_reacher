import numpy as np
import random
from collections import namedtuple, deque
import torch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)



DEFAULT_PRIORITY = 1e-10 # how much is the minimum priority given to each experience

class PriorityReplayBuffer(ReplayBuffer):
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, alpha = 0.6):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["priority","tie_breaker","state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.time_point = 0
        self.alpha = alpha
        self.default_priority = DEFAULT_PRIORITY**self.alpha
        self.max_priority = self.default_priority

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(self.max_priority, self.time_point, state, action, reward, next_state, done)
        self.time_point += 1
        self.memory.append(e)

    def sample(self, beta = 0.4):
        """Randomly sample a batch of experiences from memory."""
        
        priorities = [e.priority for e in self.memory]
        total = len(priorities)
        sampling_probs = np.array(priorities)
        sampling_probs /= sampling_probs.sum()
        weights = (total * sampling_probs) ** (-beta)
        weights /= weights.max()

        indices = np.random.choice(range(len(self.memory)), size=self.batch_size, replace=False, p=sampling_probs)
        experiences = [self.memory[ind] for ind in indices]

        weights = torch.from_numpy(np.array(weights[indices], dtype=np.float32)).float().to(device)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states, actions, rewards, next_states, dones, weights, indices)

    def update_priorities(self, experience_indices, errors):
        # update of the priorities
        for exp, error in zip(experience_indices, errors.cpu().data.numpy()):
            self.memory[exp] = self.memory[exp]._replace(priority=np.abs(error.item())**self.alpha)
            self.max_priority = max(self.memory[exp].priority, self.max_priority)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)