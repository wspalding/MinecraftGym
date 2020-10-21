import gym
import wandb
import numpy as np

from ml_agents import cartpol_dqn
from ml_models import cartpol_model

# create env
env = gym.make('CartPole-v1')
env.reset()

# init wandb and connect it to gym
wandb.init(project='cartpol_test_1',
            monitor_gym=True)
config = wandb.config
# set config variable
config.epochs = 5000
config.checkpoint_interval = 1000
config.batch_size = 32
config.training_epochs = 1

env_shape = env.observation_space.shape
action_shape = env.action_space.n


agent = cartpol_dqn.CartPolDQNAgent(env)


# main environment loop
for i in range(config.epochs + 1):
    print(i)

    # tracked values
    run_video = []
    total_reward = 0
    episode_step = 0

    # reset env
    observation = env.reset()
    done = False
    while not done:
        #render screen
        screen = env.render(mode='rgb_array')
        run_video.append(screen)
        
        observation = np.reshape(observation, [1, env_shape[0]])
            # need to reshape in order to "inform" model 
            # it is only being given 1 data sample
        
        action = agent.get_action(observation, explore=True)

        new_observation, reward, done, info = env.step(action)

        agent.remember((observation, action, reward, new_observation, done))

        observation = new_observation

        # log action
        if(i % config.checkpoint_interval == 0):
            wandb.log({"actions_{}".format(i): action})
        total_reward += reward
        episode_step += 1

    # log values and video
    if(i % config.checkpoint_interval == 0):
        run = np.array(run_video).transpose(0,3,1,2)
        wandb.log({"run_{}".format(i): wandb.Video(run, fps=30, format="gif")})
    wandb.log({"total reward": total_reward})
    print("total reward: ", total_reward)

    # preform memory replay to train
    agent.memory_replay(config.batch_size, epochs=config.training_epochs, ce=i)


# close env
env.close()

