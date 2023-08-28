from stable_baselines3 import PPO, A2C, DQN
from easy21 import Easy21
import time
from stable_baselines3.common.vec_env import DummyVecEnv

#env = Easy21()
num_cpu = 20
env = DummyVecEnv([(lambda: Easy21()) for i in range(num_cpu)])

tensorboard_dir = "./tb_logs/"
models_dir = "./modelsDQN/"
model = DQN("MlpPolicy", env, 
            policy_kwargs={"net_arch": [32, 32]},
            verbose=1, 
            tensorboard_log=tensorboard_dir, 
            device="cpu")

timestamp = time.strftime("%Y%m%d-%H%M%S")
timesteps_per_epoch = 100_000
epochs = 100
for epoch in range(epochs):
    model.learn(total_timesteps=timesteps_per_epoch, reset_num_timesteps=False, tb_log_name=timestamp)
    model.save(f"./{models_dir}/{timestamp}-{epoch}of{epochs}")
    print("Epoch:", epoch)
    print("Timesteps:", (epoch + 1) * timesteps_per_epoch)
    print("------")

# episodes = 100
# wins = 0
# total_ep_len = 0
# for episode in range(episodes):
#     done = False
#     obs, info = env.reset(seed=episode)
#     ep_len = 0
#     while not done:
#         env.render()
#         ep_len += 1
#         total_ep_len += 1
#         print("obs:", obs)
#         action, _states = model.predict(obs)
#         #action = env.action_space.sample()
#         if action == 0:
#             print("action: hit")
#         else:
#             print("action: stick")
#         obs, reward, done, truncated, info = env.step(action)
#         print("reward:", reward)
#         if done:
#             env.render()
#             print("Episode Length:", ep_len)
#             if reward > 0:
#                 wins += 1
#                 print("WIN", end="")
#             print("------")
            
# print("Average Episode Length:", total_ep_len / episodes)
# print("Win Rate:", wins / episodes)