import gym
import numpy as np

env = gym.make("TendonOneSegmentEnv-v0")
s = env.reset()

while True:
   s, r, d, info = env.step(env.action_space.sample())
   env.render()
   if d:
      print("One segment environment successfully tested.")
      break

env = gym.make("TendonTwoSegmentEnv-v0")
s = env.reset()

while True:
   s, r, d, info = env.step(env.action_space.sample())
   env.render(mode="human")
   if d:
      print("Two segment environment successfully tested.")
      break

