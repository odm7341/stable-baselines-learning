from stable_baselines3.common.env_checker import check_env
from gridworld import GridWorldEnv as Gridworld
from easy21 import Easy21

#env = Gridworld(size=4, render_mode="rgb_array")
env = Easy21()
check_env(env, warn=True)

episodes = 20

for episode in range(episodes):
    done = False
    obs = env.reset()
    while not done:
        env.render()
        action = env.action_space.sample()
        if action == 0:
            print("action: hit")
        else:
            print("action: stick")
        obs, reward, done, truncated, info = env.step(action)
        print("reward:", reward)
        if done:
            env.render()
            print("------")