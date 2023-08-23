# from stable_baselines3.common.env_checker import check_env

from gridworld import GridWorldEnv as Gridworld

env = Gridworld(size=4, render_mode="rgb_array")

# print(check_env(env, warn=True))

episodes = 20

for episode in range(episodes):
    done = False
    obs = env.reset()
    while not done:
        action = env.action_space.sample()
        print("action:", action)
        obs, reward, done, truncated, info = env.step(action)
        print("obs:", obs)
        print("reward:", reward)
        env.render()