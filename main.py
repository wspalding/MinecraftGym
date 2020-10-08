import gym
import wandb

# init wandb and connect it to gym
wandb.init(project='cartpol_test_1',
            monitor_gym=True)

# create env
env = gym.make('CartPole-v0')
env.reset()

# main environment loop
for _ in range(1000):
    env.render()
    # take a random action
    action = env.action_space.sample()
    env.step(action) 
    # log action
    wandb.log({"actions": action})

# close env
env.close()