from stable_baselines3 import PPO, A2C
from easy21 import Easy21

env = Easy21()
env.reset()

model = A2C("MlpPolicy", env, verbose=1, device="cpu", tensorboard_log="./tb_logs/")

model.learn(total_timesteps=10000)

episodes = 30
wins = 0
for episode in range(episodes):
    done = False
    obs, info = env.reset()
    while not done:
        env.render()
        print("obs:", obs)
        action, _states = model.predict(obs)
        #action = env.action_space.sample()
        if action == 0:
            print("action: hit")
        else:
            print("action: stick")
        obs, reward, done, truncated, info = env.step(action)
        print("reward:", reward)
        if done:
            env.render()
            if reward > 0:
                wins += 1
            print("------")

print("Win Rate:", wins / episodes)