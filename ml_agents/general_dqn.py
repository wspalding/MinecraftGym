import numpy as np
import random

from ml_models.cartpol_model import create_CartPol_model_from_config
from collections import deque


class GeneralDQNAgent():
    def __init__(self, model_func, env, **kwargs):
        # get env info
        self.environment = env
        env_shape = env.observation_space.shape
        action_shape = env.action_space.n
            # environment input and output shapes

        # load models
        self.load_model(model_func, env_shape, action_shape, file=kwargs.get('model_file'))
        self.target_update_count = 0
            # load models:
            # self.model is our main model used to make predictions
            # input => state of environment
            #   ex. [x, y, velocity, ect]
            # output => an array with reward predictions for each 
            #   ex. [reward if action 0 taken, reward if action 1 taken, ...]

            # target model is used  
        
        # init replay memory
        self.min_experiences = kwargs.get('min_experiences', 0)
        self.max_experiences = kwargs.get('max_experiences', 2000)
        self.replay_memory = deque(maxlen=self.max_experiences)
            # used to store (state, action, reward, next_state, done) tuples
            # that are used for training

        # set exploration rate
        self.exploration_rate = kwargs.get('exploration_rate', 1.0)
        self.exploration_rate_min = kwargs.get('explration_rate_min', 0.01)
        self.exploration_rate_decay = kwargs.get('explration_rate_decay', 0.995)
            # probablility that the agent will take a random action 
            # rather than one given by the model
            # exploration rate starts high, when the model has not been 
            # trained much, and decays as the model is trained, till it hits
            # a minimum value

        # set gamma
        self.gamma = kwargs.get('gamma', 0.95)
            # used to calculate uture discounted reward
            # best_target_reward = argmax(target_model.predict(next_state))
                # the best reward our target model thinks we can get in the next state
            # best_model_reward = argmax(model.predict(state)) 
                # The reward our model predited for the action it took
            # loss = (reward + gamma * best_target_reward - best_model_reward)^2
            # 
            # when we want to fit our model we call
            # target = reward + gamma * best_target_reward
            # model.fit(state, target)


        self.callbacks = kwargs.get('callbacks', [])

    def load_model(self, model_func, env_shape, action_shape, file=None):
        if file:
            self.model = load_model(file)
        else:
            self.model = model_func(env_shape, action_shape)

    def get_action(self, state, explore=False):
        if explore and np.random.rand() <= self.exploration_rate:
            # take random action
            return random.randrange(self.environment.action_space.n)
        return np.argmax(self.model.predict(state))

    def remember(self, memory_item):
        self.replay_memory.append(memory_item)

    def memory_replay(self, batch_size, **kwargs):
        if len(self.replay_memory) < batch_size or len(self.replay_memory) < self.min_experiences:
            # if we dont have enough memories to train from, dont do anything
            return

        training_epochs = kwargs.get('epochs', 1)

        sample_batch = random.sample(self.replay_memory, batch_size)

        states, targets = self.calculate_targets(sample_batch, cur_epoch=kwargs.get('ce'))

        self.model.fit(np.squeeze(states), targets, 
                            epochs=training_epochs, 
                            verbose=0,
                            callbacks=self.callbacks)

        self.exploration_rate = max(self.exploration_rate_min, 
                                    self.exploration_rate * self.exploration_rate_decay)

    def calculate_targets(self, sample_batch, cur_epoch=0):
        states = np.array([mi[0] for mi in sample_batch])
        actions = np.array([mi[1] for mi in sample_batch])
        rewards = np.array([mi[2] for mi in sample_batch])
        next_states = np.array([mi[3] for mi in sample_batch])
        dones = np.array([mi[4] for mi in sample_batch]) 

        # my way
        targets = []
        predictions = self.model.predict(np.squeeze(states))
        weighted_predictions = self.gamma * np.amax(self.model.predict(next_states), axis=1)
        for i in range(len(sample_batch)):
            t = predictions[i]
            if not dones[i]:
                t[actions[i]] = rewards[i] + weighted_predictions[i]
            else:
                t[actions[i]] = rewards[i]
            targets.append(t)
        targets = np.array(targets)

        return states, targets
            # rewards.shape = (batch_size,)
            # gamma.shape = scalar
            # self.target_model.predict(next_states).shape = (batch_size, num_actions)
            # targets.shape = (batch_size, num_actions)
