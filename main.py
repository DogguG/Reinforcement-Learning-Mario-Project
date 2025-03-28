import torch
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from agent import Agent
from nes_py.wrappers import JoypadSpace
from wrappers import apply_wrappers
import os
from utils import *
import csv

# In thư mục làm việc hiện tại
print("Current working directory:", os.getcwd())

# Hàm để ghi dữ liệu vào file CSV
def log_to_csv(episode, total_reward, epsilon, replay_buffer_size, learn_step_counter, filename=os.path.join("D:/Project/Seerminar/Super-Mario-Bros-RL", "training_log.csv")):
    print(f"Attempting to write to: {os.path.abspath(filename)}")
    file_exists = os.path.isfile(filename)
    try:
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Episode", "Total Reward", "Epsilon", "Replay Buffer Size", "Learn Step Counter"])
            writer.writerow([episode, total_reward, epsilon, replay_buffer_size, learn_step_counter])
        print(f"Successfully logged to CSV: Episode {episode}, Total Reward {total_reward}")
    except Exception as e:
        print(f"Error writing to CSV: {e}")

model_path = os.path.join("models", get_current_date_time_string())
os.makedirs(model_path, exist_ok=True)

if torch.cuda.is_available():
    print("Using CUDA device:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available")

ENV_NAME = 'SuperMarioBros-1-1-v0'
SHOULD_TRAIN = True
DISPLAY = True  # Tắt render để giảm file tạm
CKPT_SAVE_INTERVAL = 10000  # Tăng khoảng cách lưu checkpoint
NUM_OF_EPISODES = 50_000

env = gym_super_mario_bros.make(ENV_NAME, render_mode='human' if DISPLAY else 'rgb', apply_api_compatibility=True)
env = JoypadSpace(env, RIGHT_ONLY)
env = apply_wrappers(env)

agent = Agent(input_dims=env.observation_space.shape, num_actions=env.action_space.n)

if not SHOULD_TRAIN:
    folder_name = ""
    ckpt_name = ""
    agent.load_model(os.path.join("models", folder_name, ckpt_name))
    agent.epsilon = 0.2
    agent.eps_min = 0.0
    agent.eps_decay = 0.0

env.reset()
next_state, reward, done, trunc, info = env.step(action=0)

for i in range(NUM_OF_EPISODES):    
    print("Episode:", i)
    done = False
    state, _ = env.reset()
    total_reward = 0
    try:
        while not done:
            a = agent.choose_action(state)
            new_state, reward, done, truncated, info = env.step(a)
            total_reward += reward

            if SHOULD_TRAIN:
                agent.store_in_memory(state, a, reward, new_state, done)
                agent.learn()

            state = new_state
    except Exception as e:
        print(f"Error in episode {i}: {e}")
        continue

    print("Total reward:", total_reward, "Epsilon:", agent.epsilon, "Size of replay buffer:", len(agent.replay_buffer), "Learn step counter:", agent.learn_step_counter)
    log_to_csv(
        episode=i,
        total_reward=total_reward,
        epsilon=agent.epsilon,
        replay_buffer_size=len(agent.replay_buffer),
        learn_step_counter=agent.learn_step_counter
    )

    if SHOULD_TRAIN and (i + 1) % CKPT_SAVE_INTERVAL == 0:
        checkpoint_path = os.path.join(model_path, "model_" + str(i + 1) + "_iter.pt")
        agent.save_model(checkpoint_path)
        # Xóa checkpoint cũ (giữ lại 2 file mới nhất)
        checkpoints = sorted([f for f in os.listdir(model_path) if f.endswith(".pt")], key=lambda x: int(x.split("_")[1]))
        while len(checkpoints) > 2:
            os.remove(os.path.join(model_path, checkpoints.pop(0)))

    print("Total reward:", total_reward)

env.close()