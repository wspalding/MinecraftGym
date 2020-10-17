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
# set config variable
config.epochs = 1000
config.checkpoint_interval = 100
config.batch_size = 18
config.training_epochs = 100

env_shape = env.observation_space.shape
action_shape = env.action_space.n


agent = cartpol_dqn.CartPolDQNAgent(env)


# main environment loop
for i in range(config.epochs + 1):
    print(i)

    # tracked values
    run_video = []
    total_reward = 0

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
        
        # take a random action
        # action = env.action_space.sample()
        action = agent.get_action(observation, explore=True)

        new_observation, reward, done, info = env.step(action)

        agent.remember(cartpol_dqn.MemoryItem(observation,
                                                action,
                                                reward,
                                                new_observation,
                                                done))

        observation = new_observation

        # log action
        if(i % config.checkpoint_interval == 0):
            wandb.log({"actions_{}".format(i): action})
        total_reward += reward
    
    # log values and video
    if(i % config.checkpoint_interval == 0):
        run = np.array(run_video).transpose(0,3,1,2)
        wandb.log({"run_{}".format(i): wandb.Video(run, fps=30, format="gif")})
    wandb.log({"total reward": total_reward})
    print("total reward: ", total_reward)

    # preform memory replay to train
    agent.memory_replay(config.batch_size, epochs=config.training_epochs)


# close env
env.close()

