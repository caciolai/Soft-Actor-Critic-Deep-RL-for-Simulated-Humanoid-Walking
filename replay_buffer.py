import random
import numpy as np

class ReplayBuffer:
    """
    A class used to represent the replay buffer
    It holds all the transitions (s_t, a_t, r_t, s_{t+1}, d_t) observed at timestep t

    Attributes
    ----------
    capacity : int
        the capacity of the buffer
    buffer : list
        the actual buffer, implemented as a list
    position : int
        the position of the first free slot in the buffer
        (returns to 0 once past capacity, and slots start to get overwritten)

    Methods
    -------
    append(s, a, r, s', d)
        Appends a transition to the buffer

    sample(batch_size) : batch
        Returns a randomly drawn sample of size batch_size from the buffer
    """
    def __init__(self, capacity):
        """
        Constructor
        :param capacity: capacity
        :type capacity: int
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def append(self, state, action, reward, next_state, done):
        """
        Appends a transition to the buffer
        :param state: state
        :type state: numpy array
        :param action: action taken in state
        :type action: numpy array
        :param reward: reward observed from the environment after action
        :type reward: numpy array (singleton)
        :param next_state: state observed from the environment after action
        :type next_state: numpy array
        :param done: done signal
        :type done: float (0. or 1.)
        :return: None
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Returns a randomly drawn sample of size batch_size from the buffer
        :param batch_size: size of the batch to be sampled
        :type batch_size: int
        :return: a numpy array in which each row is a transition
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

