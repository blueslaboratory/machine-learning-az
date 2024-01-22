# 22/01/2024
# Self-Driving Car (Deep Q-Learning) (UNTESTED)

# Install related libraries:
# pip install gym
# apt install swig cmake libopenmpi-dev zlib1g-dev
# pip install stable-baselines[mpi]==2.8.0 box2d box2d-kengz


# Import the libraries
import gym
from baselines.common.atari_wrappers import make_atari, wrap_deepmind

# Create the environment
env = gym.make('SelfDrivingCar-v0')

# Wrap the environment in the DeepMind wrapper to enable frame skipping and color rendering
env = wrap_deepmind(env)

# Set the number of actions
num_actions = env.action_space.n

# Train the DQN model
model = DQN(env=env, num_actions=num_actions)
model.learn(total_timesteps=2000000)

# Test the DQN model
env.reset()
done = False

while not done:
  action, _states = model.predict(obs)
  obs, reward, done, info = env.step(action)
  env.render()

# Close the environment
env.close()