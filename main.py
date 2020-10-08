import gym


# create env
env = gym.make('CartPole-v0')
env.reset()

# main environment loop
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action

# close env
env.close()