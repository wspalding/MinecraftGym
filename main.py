import gym
import wandb
import numpy as np

from ml_agents import general_dqn
from ml_agents import cartpol_A3C

from ml_models import space_invaders_model
from ml_models import cartpol_model

# create env
env = gym.make('CartPol-v1')
env.reset()

# init wandb and connect it to gym
wandb.init(project='CartPol_2',
            monitor_gym=True)
config = wandb.config
# set config variable
config.epochs = 5000
config.checkpoint_interval = 250
config.batch_size = 32
config.training_epochs = 1

env_shape = env.observation_space.shape
action_shape = env.action_space.n

agent = cartpol_A3C(cartpol_model.create_CartPol_model_from_config, env)

model_save_file = '/space_invaders_1.h5'

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
        
        observation = np.reshape(observation, [1, *env_shape])
            # need to reshape in order to "inform" model 
            # it is only being given 1 data sample

        action = agent.get_action(observation, explore=True)

        new_observation, reward, done, info = env.step(action)

        agent.remember((observation, action, reward, new_observation, done))

        observation = new_observation

        # log action
        # if(i % config.checkpoint_interval == 0):
        #     wandb.log({"actions_{}".format(i): action})

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

    agent.model.save('saved_models' + model_save_file, save_format='h5')

# close env
env.close()

