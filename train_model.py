from stable_baselines3 import A2C
from easy21 import Easy21

env = Easy21()
env.reset()

model = A2C("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=10000)

episodes = 10
for episode in range(episodes):
    done = False
    obs = env.reset()
    while not done:
        env.render()
        action, _states = model.predict(obs)
        if action == 0:
            print("action: hit")
        else:
            print("action: stick")
        obs, reward, done, truncated, info = env.step(action)
        print("reward:", reward)
        if done:
            env.render()
            print("------")