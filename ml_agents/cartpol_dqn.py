import numpy as np

from ml_models.cartpol_model import create_CartPol_model_from_config
from collections import deque


class CartPolDQNAgent():
    def __init__(self, env, **kwargs):
    # get env info
        env_shape = env.observation_space.shape
        action_shape = env.action_space.n
            # environment input and output shapes

    # load models
        self.load_models(env_shape, action_shape, file=kwargs.get('model_file'))
        self.target_update_count = 0
            # load models:
            # self.model is our main model used to make predictions
            # input => state of environment
            #   ex. [x, y, velocity, ect]
            # output => an array with reward predictions for each 
            #   ex. [reward if action 0 taken, reward if action 1 taken, ...]

            # target model is used  

    # init replay memory
        self.min_experiences = kwargs.get('min_experiences', 1000)
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


    def load_models(self, env_shape, action_shape, file=None):
        if file:
            self.model = load_model(file)
        else:
            self.model = create_CartPol_model_from_config(env_shape, action_shape)

        # a copy of our model that we measure against
        if file:
            self.target_model = load_model(file)
        else:
            self.target_model = create_CartPol_model_from_config(env_shape, action_shape)
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state, explore=False):
        if explore and np.random.rand() <= self.exploration_rate:
            # take random action
            return random.randrange(self.action_size)
        return np.argmax(self.model.predict(state))

    def train():
        pass

    def update_replay_memory():
        pass



class MemoryItem:
    def __init__(self, s, a, r, s2, d):
        self.state = s
        self.action = a
        self.reward = r
        self.next_state = s2
        self.done = d