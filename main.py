import gym
import wandb
import numpy as np

# init wandb and connect it to gym
wandb.init(project='cartpol_test_1',
            monitor_gym=True)

# create env
env = gym.make('CartPole-v0')
env.reset()

# tracked values
run_video = []
total_reward = 0


# main environment loop
for _ in range(1000):
    #render screen
    screen = env.render(mode='rgb_array')
    run_video.append(screen)

    # take a random action
    action = env.action_space.sample()
    new_observation, reward, done, _ = env.step(action) 

    # log action
    wandb.log({"actions": action})
    total_reward += reward

# close env
env.close()
run = np.array(run_video).transpose(0,3,1,2)

wandb.log({"run": wandb.Video(run, fps=30, format="gif")})
wandb.log({"total reward": total_reward})