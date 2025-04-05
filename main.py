import torch
import threading
import numpy as np
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from agent import Agent
from nes_py.wrappers import JoypadSpace
from gym import Wrapper
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
import os
import csv
from datetime import datetime

# Tối ưu hóa CUDA
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# In thư mục làm việc hiện tại
print("Current working directory:", os.getcwd())

# Hàm để ghi dữ liệu vào file CSV
def log_to_csv(episode, total_reward, epsilon, replay_buffer_size, learn_step_counter, filename=os.path.join("D:/Project/Seerminar/Super-Mario-Bros-RL", "training_log.csv")):
    file_exists = os.path.isfile(filename)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Episode", "Total Reward", "Epsilon", "Replay Buffer Size", "Learn Step Counter", "Timestamp"])
            writer.writerow([episode, total_reward, epsilon, replay_buffer_size, learn_step_counter, current_time])
    except Exception as e:
        print(f"Error writing to CSV: {e}")

# Wrapper để thêm phần thưởng tùy chỉnh
class RewardWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_x_pos = 0
        self.prev_y_pos = 0
        self.prev_score = 0
        self.stagnation_counter = 0
        self.stagnation_threshold = 50
        self.max_x_pos = 0

    def step(self, action):
        state, reward, done, trunc, info = self.env.step(action)
        x_pos = info['x_pos']
        y_pos = info['y_pos']
        score = info['score']
        
        if info.get('flag_get', False):
            reward += 2000
        
        if x_pos > self.prev_x_pos + 50:
            reward += 200  # Tăng phần thưởng khi đi xa
        
        if y_pos > self.prev_y_pos + 20:
            reward += 30
        
        if score > self.prev_score:
            reward += 50
        
        if x_pos > self.max_x_pos:
            reward += 30  # Tăng phần thưởng khi đạt vị trí xa nhất
            self.max_x_pos = x_pos
        
        if x_pos == self.prev_x_pos:
            self.stagnation_counter += 1
            if self.stagnation_counter >= self.stagnation_threshold:
                reward -= 2  # Giảm phạt khi đứng yên
        else:
            self.stagnation_counter = 0

        if done and not info.get('flag_get', False) and x_pos < 1000:
            reward -= 300  # Tăng phạt khi chết sớm

        self.prev_x_pos = x_pos
        self.prev_y_pos = y_pos
        self.prev_score = score
        return state, reward, done, trunc, info

# Wrapper để lưu frame và hành động
class SkipFrame(Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip
        self.frames_log = []
        self.actions_log = []
    
    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self._skip):
            next_state, reward, done, trunc, info = self.env.step(action)
            self.frames_log.append(next_state.copy())
            self.actions_log.append(action)
            total_reward += reward
            if done:
                break
        return next_state, total_reward, done, trunc, info
    
    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)
        self.frames_log = [state.copy()]
        self.actions_log = [0]
        return state, info

# Hàm apply_wrappers
def apply_wrappers(env):
    env = SkipFrame(env, skip=4)
    env = ResizeObservation(env, shape=84)
    env = GrayScaleObservation(env)
    env = FrameStack(env, num_stack=4, lz4_compress=True)
    env = RewardWrapper(env)
    return env

# Định nghĩa CBAM và AgentNN
class CBAM(torch.nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_gate = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(channels, channels // reduction, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(channels // reduction, channels, 1),
            torch.nn.Sigmoid()
        )
        self.spatial_gate = torch.nn.Sequential(
            torch.nn.Conv2d(2, 1, kernel_size=7, padding=3),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        channel_weight = self.channel_gate(x)
        x = x * channel_weight
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        spatial_input = torch.cat([max_pool, avg_pool], dim=1)
        spatial_weight = self.spatial_gate(spatial_input)
        x = x * spatial_weight
        return x

class AgentNN(torch.nn.Module):
    def __init__(self, input_shape, n_actions, freeze=False):
        super().__init__()
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            CBAM(32),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            CBAM(64),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            CBAM(64),
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.network = torch.nn.Sequential(
            self.conv_layers,
            torch.nn.Flatten(),
            torch.nn.Linear(conv_out_size, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, n_actions)
        )

        if freeze:
            self._freeze()
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(self.device)

    def forward(self, x):
        return self.network(x)

    def _get_conv_out(self, shape):
        o = self.conv_layers(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def _freeze(self):        
        for p in self.network.parameters():
            p.requires_grad = False

# Cấu hình huấn luyện
model_path = os.path.join("models", "mario_model")
os.makedirs(model_path, exist_ok=True)

if torch.cuda.is_available():
    print("Using CUDA device:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available")

ENV_NAME = 'SuperMarioBros-1-1-v0'
SHOULD_TRAIN = True
DISPLAY = True
CKPT_SAVE_INTERVAL = 500
NUM_OF_EPISODES = 10000

env = gym_super_mario_bros.make(ENV_NAME, render_mode='human' if DISPLAY else 'rgb_array', apply_api_compatibility=True)
env = JoypadSpace(env, RIGHT_ONLY)
env = apply_wrappers(env)

# Tạo agent với AgentNN
agent = Agent(input_dims=env.observation_space.shape, 
              num_actions=env.action_space.n,
              lr=0.0001,
              batch_size=256,
              replay_buffer_capacity=500_000,  # Tăng dung lượng replay buffer
              eps_decay=0.9995,
              eps_min=0.15,
              network_class=AgentNN)

# Tải mô hình đã huấn luyện tại episode 1000 để tiếp tục
checkpoint_path = "D:\Project\Seerminar\Super-Mario-Bros-RL\models\mario_model\model_1500_iter.pt"
if os.path.exists(checkpoint_path):
    print(f"Loading model from {checkpoint_path}")
    agent.load_model(checkpoint_path)
    agent.epsilon = 0.5  # Tăng epsilon để khám phá thêm
else:
    print(f"Checkpoint {checkpoint_path} not found! Starting from scratch.")
    agent.epsilon = 1.0

if not SHOULD_TRAIN:
    folder_name = ""
    ckpt_name = ""
    agent.load_model(os.path.join("models", folder_name, ckpt_name))
    agent.epsilon = 0.5
    agent.eps_min = 0.0
    agent.eps_decay = 0.0

env.reset()
next_state, reward, done, trunc, info = env.step(action=0)

# Hàm để học nhiều lần trong mỗi bước
def learn_multiple_times(agent, times=12):  # Tăng số lần học
    for _ in range(times):
        agent.learn()

# Tạo hàm prefetch để tối ưu hóa việc lưu trữ dữ liệu
def prefetch_experiences(agent, states, actions, rewards, next_states, dones):
    for i in range(len(states)):
        agent.store_in_memory(states[i], actions[i], rewards[i], next_states[i], dones[i])

# Chuẩn bị buffer cho prefetch
prefetch_states = []
prefetch_actions = []
prefetch_rewards = []
prefetch_next_states = []
prefetch_dones = []
prefetch_threshold = 64

# Tiếp tục huấn luyện từ episode 1230
START_EPISODE = 1508
print(f"Starting training from episode {START_EPISODE}")
for i in range(START_EPISODE, NUM_OF_EPISODES + 1):
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
                prefetch_states.append(state)
                prefetch_actions.append(a)
                prefetch_rewards.append(reward)
                prefetch_next_states.append(new_state)
                prefetch_dones.append(done)
                
                if len(prefetch_states) >= prefetch_threshold:
                    threading.Thread(
                        target=prefetch_experiences, 
                        args=(agent, 
                              prefetch_states.copy(), 
                              prefetch_actions.copy(), 
                              prefetch_rewards.copy(), 
                              prefetch_next_states.copy(), 
                              prefetch_dones.copy())
                    ).start()
                    
                    prefetch_states.clear()
                    prefetch_actions.clear()
                    prefetch_rewards.clear()
                    prefetch_next_states.clear()
                    prefetch_dones.clear()
                
                learn_multiple_times(agent, times=8)
                
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
        checkpoint_path = os.path.join(model_path, f"model_{i + 1}_iter.pt")
        agent.save_model(checkpoint_path)
        
        checkpoints = sorted([f for f in os.listdir(model_path) if f.endswith(".pt")], key=lambda x: int(x.split("_")[1]))
        while len(checkpoints) > 2:
            os.remove(os.path.join(model_path, checkpoints.pop(0)))
            
    print("Total reward:", total_reward)

env.close()