import gym
import gym.spaces
import numpy as np
import skimage.measure


class Environment2D:

    def __init__(self,
                 environment_name=None):
        self.pool_shape  = (8,8)
        self.environment = gym.make(environment_name)
        self.action_space = self.environment.action_space

    def reset(self):
        observation = self.environment.reset()
        observation = self.reshape(observation)
        return observation


    def reshape(self,observation):
        if len(observation.shape) == 1:
            return observation
        else:
            observation = np.mean( observation, axis=2 )
            observation = skimage.measure.block_reduce( observation, self.pool_shape, np.mean)
            return observation / 255.

    def step(self,action):
        observation, reward, done, info = self.environment.step(action)
        return (self.reshape(observation), reward, done, info)

    def render(self):
        self.environment.render()
