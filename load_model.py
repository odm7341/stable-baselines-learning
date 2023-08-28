

from stable_baselines3 import DQN, PPO
from easy21 import Easy21

env = Easy21()
env.reset()

model_dir_dqn = "./modelsDQN"
model_dir_ppo = "./modelsPPO"

default_dqn = f"{model_dir_dqn}/20230825-093423-33of100"
small_dqn = f"{model_dir_dqn}/20230825-113205-53of100"
default_ppo = f"{model_dir_ppo}/20230824-152451-37of100"

models = [DQN.load(default_dqn), DQN.load(small_dqn), PPO.load(default_ppo)]
model_results = []

seed = 2

for model in models:
    model.set_random_seed(seed)
    episodes = 100
    wins = 0
    total_ep_len = 0
    for episode in range(episodes):
        done = False
        obs, info = env.reset(seed=episode + seed)
        ep_len = 0
        while not done:
            env.render()
            ep_len += 1
            total_ep_len += 1
            #print("obs:", obs)
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
                print("Episode Length:", ep_len)
                if reward > 0:
                    wins += 1
                    print("WIN", end="")
                print("------")
                
    model_results.append((total_ep_len / episodes, wins / episodes))
    print("Average Episode Length:", total_ep_len / episodes)
    print("Win Rate:", wins / episodes)
    print("***************************************************")

print("Default DQN:", f"\tAvg Episode len: {model_results[0][0]},\tWin Rate: {model_results[0][1]}")
print("Small DQN:", f"\tAvg Episode len: {model_results[1][0]},\tWin Rate: {model_results[1][1]}")
print("Default PPO:", f"\tAvg Episode len: {model_results[2][0]},\tWin Rate: {model_results[2][1]}")