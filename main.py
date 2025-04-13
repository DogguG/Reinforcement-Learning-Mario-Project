import torch
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from agent import Agent
from nes_py.wrappers import JoypadSpace
from wrappers import apply_wrappers
from lime import lime_image
import os
from utils import *
import csv
import numpy as np
import matplotlib.pyplot as plt

# Hàm để ghi dữ liệu vào file CSV
def log_to_csv(episode, step, action, predict_value, interception, right_value, x_pos, total_reward, epsilon, replay_buffer_size, learn_step_counter, filename=os.path.join("D:\DH\Serminar\Reinforcement-Learning-Mario-Project", "output_mario.csv")):
    file_exists = os.path.isfile(filename)
    try:
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Episode", "Step", "Action", "Predict Value", "Interception", "Right Value", "X-Position", "Total Reward", "Epsilon", "Replay Buffer Size", "Learn Step Counter"])
            writer.writerow([episode, step, ACTION_NAMES[action], predict_value, interception, right_value, x_pos, total_reward, epsilon, replay_buffer_size, learn_step_counter])
    except Exception as e:
        print(f"Error writing to CSV: {e}")

# Ánh xạ hành động sang tên dễ hiểu
ACTION_NAMES = {
    0: 'noop',                    # ['NOOP']
    1: 'move_right',              # ['right']
    2: 'move_right_jump',         # ['right', 'A']
    3: 'move_right_speed',        # ['right', 'B']
    4: 'move_right_jump_speed',   # ['right', 'A', 'B']
}

# Hàm vẽ biểu đồ Q-value (sửa để làm nổi bật cột có Q-value cao nhất)
def plot_q_values(q_values, chosen_action, episode, step, save_path="q_values"):
    actions = [ACTION_NAMES[i] for i in range(len(RIGHT_ONLY))]  # Dùng tên hành động mới
    plt.figure(figsize=(8, 6))
    bars = plt.bar(actions, q_values, color='lightblue')
    
    # Tìm hành động có Q-value cao nhất
    max_q_value_action = np.argmax(q_values)
    bars[max_q_value_action].set_color('orange')  # Làm nổi bật cột có Q-value cao nhất
    
    plt.title(f"Q-Values for Actions (Episode {episode}, Step {step})")
    plt.xlabel("Actions")
    plt.ylabel("Q-Value")
    plt.xticks(rotation=45)  # Xoay nhãn trục x để dễ đọc
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"q_values_episode_{episode}_step_{step}.png"))
    plt.close()

# Hàm vẽ biểu đồ tổng reward qua các bước
def plot_reward_over_time(rewards, episode, save_path="rewards"):
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label="Total Reward", color='blue')
    plt.title(f"Total Reward Over Time (Episode {episode})")
    plt.xlabel("Step")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.savefig(os.path.join(save_path, f"reward_episode_{episode}.png"))
    plt.close()

# Hàm tính "interception" (khoảng cách đến chướng ngại vật gần nhất)
def calculate_interception(x_pos, info):
    obstacle_positions = [200, 500, 800, 1200]  # Ví dụ
    future_obstacles = [pos for pos in obstacle_positions if pos > x_pos]
    if future_obstacles:
        interception = min(future_obstacles) - x_pos  # Khoảng cách đến chướng ngại vật gần nhất
    else:
        interception = float('inf')  # Không có chướng ngại vật phía trước
    return interception

# Hàm tính "right value" (độ tin cậy: chênh lệch giữa Q-value cao nhất và nhì cao)
def calculate_right_value(q_values):
    sorted_q_values = np.sort(q_values)[::-1]  # Sắp xếp giảm dần
    if len(sorted_q_values) >= 2:
        right_value = sorted_q_values[0] - sorted_q_values[1]  # Chênh lệch giữa Q-value cao nhất và nhì cao
    else:
        right_value = 0.0  # Trường hợp chỉ có 1 hành động (hiếm)
    return right_value

model_path = os.path.join("models", get_current_date_time_string())
os.makedirs(model_path, exist_ok=True)

# Tạo thư mục để lưu biểu đồ
q_values_path = os.path.join("plots", "q_values")
rewards_path = os.path.join("plots", "rewards")
os.makedirs(q_values_path, exist_ok=True)
os.makedirs(rewards_path, exist_ok=True)

if torch.cuda.is_available():
    print("Using CUDA device:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available")

ENV_NAME = 'SuperMarioBros-1-1-v0'
SHOULD_TRAIN = True
DISPLAY = True
CKPT_SAVE_INTERVAL = 10000
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

# Thêm hệ số ngẫu nhiên cho tốc độ (từ yêu cầu trước)
def adjust_action_with_random_speed(action):
    speed_factor = np.random.uniform(0.5, 1.5)
    if speed_factor < 0.8:
        return 1  # ['right']
    elif speed_factor > 1.2:
        if action == 2:
            return 4  # ['right', 'A', 'B']
        return 3  # ['right', 'B']
    return action

for i in range(NUM_OF_EPISODES):    
    print("Episode:", i)
    done = False
    state, _ = env.reset()
    total_reward = 0
    step = 0
    reward_history = []

    try:
        while not done:
            a = agent.choose_action(state)
            a = adjust_action_with_random_speed(a)

            # Lấy Q-value của tất cả hành động
            state_tensor = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(agent.online_network.device)
            with torch.no_grad():
                q_values = agent.online_network(state_tensor).cpu().numpy()[0]

            # Tính predict value (Q-value của hành động được chọn)
            predict_value = q_values[a]

            # Tính interception (khoảng cách đến chướng ngại vật gần nhất)
            interception = calculate_interception(info['x_pos'], info)

            # Tính right value (độ tin cậy)
            right_value = calculate_right_value(q_values)

            # In 3 giá trị (mỗi 100 bước để tránh làm terminal quá dài)
            if step % 100 == 0:
                print(f"Step {step}: Action chosen = {ACTION_NAMES[a]} - Predict Value: {predict_value:.4f} - Interception: {interception:.2f} - Right Value: {right_value:.4f}")

            # Thực hiện hành động
            new_state, reward, done, truncated, info = env.step(a)
            total_reward += reward
            reward_history.append(total_reward)

            # Vẽ biểu đồ Q-value mỗi 100 bước
            if step % 100 == 0:
                plot_q_values(q_values, a, i, step, save_path=q_values_path)

            # Ghi log tại mỗi bước (hoặc mỗi 100 bước để tránh file CSV quá lớn)
            if step % 100 == 0:
                log_to_csv(
                    episode=i,
                    step=step,
                    action=a,
                    predict_value=predict_value,
                    interception=interception,
                    right_value=right_value,
                    x_pos=info['x_pos'],
                    total_reward=total_reward,
                    epsilon=agent.epsilon,
                    replay_buffer_size=len(agent.replay_buffer),
                    learn_step_counter=agent.learn_step_counter
                )

            if SHOULD_TRAIN:
                agent.store_in_memory(state, a, reward, new_state, done)
                agent.learn()

            state = new_state
            step += 1

    except Exception as e:
        print(f"Error in episode {i}: {e}")
        continue

    print("Total reward:", total_reward, "Epsilon:", agent.epsilon, "Size of replay buffer:", len(agent.replay_buffer), "Learn step counter:", agent.learn_step_counter)

    # Vẽ biểu đồ tổng reward cho episode
    plot_reward_over_time(reward_history, i, save_path=rewards_path)

    if SHOULD_TRAIN and (i + 1) % CKPT_SAVE_INTERVAL == 0:
        checkpoint_path = os.path.join(model_path, "model_" + str(i + 1) + "_iter.pt")
        agent.save_model(checkpoint_path)
        checkpoints = sorted([f for f in os.listdir(model_path) if f.endswith(".pt")], key=lambda x: int(x.split("_")[1]))
        while len(checkpoints) > 2:
            os.remove(os.path.join(model_path, checkpoints.pop(0)))

    print("Total reward:", total_reward)

env.close()
