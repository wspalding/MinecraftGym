import gym
import wandb
import numpy as np

from ml_agents import cartpol_dqn
from ml_models import cartpol_model

# create env
env = gym.make('CartPole-v0')
env.reset()

# init wandb and connect it to gym
wandb.init(project='cartpol_test_1',
            monitor_gym=True)
config = wandb.config

# set config values
env_shape = env.observation_space.shape
action_shape = env.action_space.n


agent = cartpol_dqn.CartPolDQNAgent(env)


# tracked values
run_video = []
total_reward = 0

observation = env.reset()

# main environment loop
for _ in range(1000):
    #render screen
    screen = env.render(mode='rgb_array')
    run_video.append(screen)
    
    observation = np.reshape(observation, [1, env_shape[0]])
        # need to reshape in order to "inform" model 
        # it is only being given 1 data sample
    
    # take a random action
    # action = env.action_space.sample()
    action = np.argmax(test_model.predict(observation))
    new_observation, reward, done, _ = env.step(action)

    observation = new_observation

    # log action
    wandb.log({"actions": action})
    total_reward += reward

# close env
env.close()

# log values and video
run = np.array(run_video).transpose(0,3,1,2)
wandb.log({"run": wandb.Video(run, fps=30, format="gif")})
wandb.log({"total reward": total_reward})