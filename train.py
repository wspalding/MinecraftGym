import gym
import wandb
import numpy as np
import os
from wandb.keras import WandbCallback

from ml_agents import general_dqn
from ml_agents import cartpol_A3C

from ml_models import space_invaders_model
from ml_models import cartpol_model

from ml_callbacks import wandb_logging_callback


def train():
    # create env
    env = gym.make('SpaceInvaders-v0')
    # env = gym.make('CartPole-v1')
    env.reset()

    config_defaults = {
        'learning_rate': 0.01,
        'epochs': 10,
        'batch_size': 32,
        'training_epochs': 1,
        'loss_function': 'huber',
        'optimizer': 'adam'
    }

    # init wandb and connect it to gym
    wandb.init(project='space_inavders_lstm',
                monitor_gym=True,
                config=config_defaults)
    config = wandb.config
    # set config variable
    config.checkpoint_interval = 250
    # config.epochs = 10
    # config.batch_size = 32
    # config.training_epochs = 1

    env_shape = env.observation_space.shape
    action_shape = env.action_space.n


    # create model and agent
    model = space_invaders_model.create_SpaceInvaders_lstm(env_shape)
    agent = general_dqn.GeneralDQNAgent(model, env,
                                        callbacks=[WandbCallback()],
                                        explration_rate_min=0.001,
                                        explration_rate_decay=0.999,
                                        LR=config.learning_rate,
                                        loss_function=config.loss_function,
                                        metrics=None)


    model_save_file = '/space_invaders_1.h5'

    best_run = 0
    average_total_reward = 0

    # main environment loop
    for i in range(config.epochs + 1):
        print(i)
        log_dict = {}

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

            total_reward += reward
            episode_step += 1

        log_video = False

        if total_reward >= best_run:
            best_run = total_reward
            log_video = True
        if(i % config.checkpoint_interval == 0):
            log_video = True

        # log values and video
        if(log_video):
            run = np.array(run_video).transpose(0,3,1,2)
            log_dict["run_{}".format(i)] = wandb.Video(run, fps=30, format="gif")

            if(not os.path.isdir('saved_models')):
                os.makedirs('saved_models')

            agent.model.save('saved_models' + model_save_file, save_format='h5')

        average_total_reward = (( average_total_reward * i) + total_reward)/ (i + 1)

        log_dict["average total reward"] = average_total_reward
        log_dict["total reward"] = total_reward

        # preform memory replay to train
        agent.memory_replay(config.batch_size, epochs=config.training_epochs, ce=i)

        wandb.log(log_dict)
        print("total reward: {}, average reward: {:4.4f}".format(total_reward, average_total_reward))

    # close env
    env.close()


if(__name__ == '__main__'):
    train()