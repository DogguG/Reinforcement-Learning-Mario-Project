import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
from agent import Agent
from gym import Wrapper
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
import os
import cv2
import torch
import numpy as np
from lime import lime_image
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries

# Modified SkipFrame wrapper to log frames and actions
class SkipFrame(Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip
        self.counter = 0
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

def apply_wrappers(env):
    env = SkipFrame(env, skip=4)
    env = ResizeObservation(env, shape=84)
    env = GrayScaleObservation(env)
    env = FrameStack(env, num_stack=4, lz4_compress=True)
    return env

# Hàm để chuyển trạng thái thành hình ảnh RGB (cho LIME)
def state_to_rgb(state):
    frame = state[3]  # Khung hình cuối cùng
    frame = np.repeat(frame[:, :, np.newaxis], 3, axis=2)  # Chuyển thành RGB
    return frame.astype(np.uint8)

# Hàm dự đoán cho LIME
def predict_fn(images, agent):
    processed_states = []
    for img in images:
        img_gray = np.mean(img, axis=2).astype(np.float32)
        img_stack = np.stack([img_gray] * 4, axis=0)  # Shape: (4, 84, 84)
        processed_states.append(img_stack)
    
    processed_states = np.array(processed_states)  # Shape: (batch_size, 4, 84, 84)
    states_tensor = torch.tensor(processed_states, dtype=torch.float32).to(agent.online_network.device)
    
    with torch.no_grad():
        q_values = agent.online_network(states_tensor)  # Shape: (batch_size, n_actions)
    return q_values.cpu().numpy()

# Hàm giải thích bằng LIME với Q-values
def explain_action(state, action, agent, episode, frame_idx, output_dir):
    # Chuyển trạng thái thành hình ảnh RGB
    state_rgb = state_to_rgb(state)
    
    # Lấy Q-values cho trạng thái hiện tại
    state_tensor = torch.tensor(np.array([state]), dtype=torch.float32).to(agent.online_network.device)
    with torch.no_grad():
        q_values = agent.online_network(state_tensor).cpu().numpy()[0]  # Shape: (n_actions,)
    
    # Tạo explainer
    explainer = lime_image.LimeImageExplainer()
    
    # Giải thích
    explanation = explainer.explain_instance(
        state_rgb,
        lambda x: predict_fn(x, agent),
        top_labels=1,
        num_samples=1000
    )
    
    # Trực quan hóa
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=5
    )
    
    # Xóa đồ thị cũ
    plt.clf()
    
    # Vẽ đồ thị mới
    plt.figure(figsize=(12, 5))  # Tăng kích thước để có chỗ cho Q-values
    plt.subplot(1, 2, 1)
    plt.imshow(state_rgb)
    plt.title(f"Original State (Episode {episode}, Frame {frame_idx})")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(mark_boundaries(state_rgb, mask))
    # Thêm Q-values vào tiêu đề
    q_values_str = ", ".join([f"Q{i}={q:.2f}" for i, q in enumerate(q_values)])
    plt.title(f"LIME Explanation: Action {action}\nQ-values: {q_values_str}")
    plt.axis('off')
    
    # Lưu hình ảnh
    plt.savefig(os.path.join(output_dir, f"episode_{episode}_frame_{frame_idx}.png"), bbox_inches='tight')
    plt.close()

# Thiết lập môi trường
ENV_NAME = 'SuperMarioBros-1-1-v0'
NUM_OF_EPISODES = 5  # Giảm số episode để kiểm tra nhanh

env = gym_super_mario_bros.make(ENV_NAME, render_mode='rgb_array', apply_api_compatibility=True)
env = JoypadSpace(env, RIGHT_ONLY)
env = apply_wrappers(env)

# Khởi tạo agent
agent = Agent(input_dims=env.observation_space.shape, num_actions=env.action_space.n)



# Tạo thư mục lưu hình ảnh LIME
output_dir = "lime_explanations"
os.makedirs(output_dir, exist_ok=True)

# Chạy agent và giải thích bằng LIME
for i in range(NUM_OF_EPISODES):
    done = False
    state, _ = env.reset()
    rewards = 0
    frame_idx = 0
    while not done:
        action = agent.choose_action(state)
        new_state, reward, done, truncated, info = env.step(action)
        rewards += reward

        # Giải thích hành động bằng LIME (chỉ làm cho một vài frame để tránh chậm)
        if frame_idx % 100 == 0 or action == 1:  # Giải thích mỗi 100 frame hoặc khi nhảy (action=1)
            try:
                explain_action(state, action, agent, i, frame_idx, output_dir)
                print(f"Saved LIME explanation for Episode {i}, Frame {frame_idx}")
            except Exception as e:
                print(f"Error generating LIME explanation for Episode {i}, Frame {frame_idx}: {e}")

        state = new_state
        frame_idx += 1

        if done:
            print(f"Episode: {i}, Reward: {rewards}")
            if info["flag_get"]:
                print(f"Agent reached the flag in Episode {i}!")
                os.makedirs(os.path.join("games", f"game_{i}"), exist_ok=True)
                frame_skip_env = env.env.env.env  # Unwrapping the environment to get the SkipFrame wrapper
                frames_log = frame_skip_env.frames_log
                actions_log = frame_skip_env.actions_log
                for j, (frame, action) in enumerate(zip(frames_log, actions_log)):
                    scaling_factor = 10
                    new_dims = (frame.shape[1] * scaling_factor, frame.shape[0] * scaling_factor)
                    frame = np.array(frame).astype(np.uint8)
                    if frame.ndim == 2:  # Nếu frame là grayscale, chuyển thành RGB
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                    frame = cv2.resize(frame, new_dims, interpolation=cv2.INTER_NEAREST)
                    cv2.imwrite(os.path.join("games", f"game_{i}", f"frame_{j}.png"), frame)

env.close()

# Tạo video từ các hình ảnh LIME
video_path = os.path.join(output_dir, "lime_explanation_video.mp4")
frame_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".png")])
if frame_files:
    frame = cv2.imread(os.path.join(output_dir, frame_files[0]))
    height, width, _ = frame.shape
    video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
    for frame_file in frame_files:
        frame = cv2.imread(os.path.join(output_dir, frame_file))
        video_writer.write(frame)
    video_writer.release()
    print(f"Video saved at {video_path}")

