import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
from agent import Agent
from gym import Wrapper
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
import os
from PIL import Image
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
    # state có shape (4, 84, 84) (4 khung hình grayscale)
    # Lấy khung hình cuối cùng (index 3) và chuyển thành RGB
    frame = state[3]  # Khung hình cuối cùng
    frame = np.repeat(frame[:, :, np.newaxis], 3, axis=2)  # Chuyển thành RGB
    return frame.astype(np.uint8)

# Hàm dự đoán cho LIME
def predict_fn(images, agent):
    # images: List các hình ảnh RGB (84, 84, 3)
    # Chuyển thành định dạng phù hợp với online_network (4, 84, 84)
    processed_states = []
    for img in images:
        # Chuyển RGB thành grayscale
        img_gray = np.mean(img, axis=2).astype(np.float32)
        # Tạo stack 4 khung hình (dùng cùng khung hình cho đơn giản)
        img_stack = np.stack([img_gray] * 4, axis=0)  # Shape: (4, 84, 84)
        processed_states.append(img_stack)
    
    processed_states = np.array(processed_states)  # Shape: (batch_size, 4, 84, 84)
    states_tensor = torch.tensor(processed_states, dtype=torch.float32).to(agent.online_network.device)
    
    with torch.no_grad():
        q_values = agent.online_network(states_tensor)  # Shape: (batch_size, n_actions)
    return q_values.cpu().numpy()

# Hàm giải thích bằng LIME
def explain_action(state, action, agent, episode, frame_idx):
    # Chuyển trạng thái thành hình ảnh RGB
    state_rgb = state_to_rgb(state)
    
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
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(state_rgb)
    plt.title(f"Original State (Episode {episode}, Frame {frame_idx})")
    plt.subplot(1, 2, 2)
    plt.imshow(mark_boundaries(state_rgb, mask))
    plt.title(f"LIME Explanation: Action {action}")
    plt.show()

ENV_NAME = 'SuperMarioBros-1-1-v0'
NUM_OF_EPISODES = 1_000
# controllers = [Image.open(f"controllers/{i}.png") for i in range(5)]

env = gym_super_mario_bros.make(ENV_NAME, render_mode='rgb_array', apply_api_compatibility=True)
env = JoypadSpace(env, RIGHT_ONLY)
env = apply_wrappers(env)

agent = Agent(input_dims=env.observation_space.shape, num_actions=env.action_space.n)

# agent.load_model("models/folder_name/ckpt_name")

for i in range(NUM_OF_EPISODES):
    done = False
    state, _ = env.reset()
    rewards = 0
    frame_idx = 0
    while not done:
        action = agent.choose_action(state)
        frame = env.render()
        new_state, reward, done, truncated, info = env.step(action)
        rewards += reward

        # Giải thích hành động bằng LIME (chỉ làm cho một vài frame để tránh chậm)
        if frame_idx % 100 == 0:  # Giải thích mỗi 100 frame
            explain_action(state, action, agent, i, frame_idx)

        state = new_state
        frame_idx += 1

        if done:
            print(f"Episode: {i}, Reward: {rewards}")
            if info["flag_get"]:
                os.makedirs(os.path.join("games", f"game_{i}"), exist_ok=True)
                frame_skip_env = env.env.env.env  # Unwrapping the environment to get the SkipFrame wrapper
                frames_log = frame_skip_env.frames_log
                actions_log = frame_skip_env.actions_log
                for j, (frame, action) in enumerate(zip(frames_log, actions_log)):
                    scaling_factor = 10
                    new_dims = (frame.shape[1] * scaling_factor, frame.shape[0] * scaling_factor)
                    frame = Image.fromarray(frame).resize(new_dims, Image.NEAREST)
                    frame.save(os.path.join("games", f"game_{i}", f"frame_{j}.png"))
                    # controllers[action].save(os.path.join("games", f"game_{i}", f"controller_{j}.png"))
        
        # if i % 5000 == 0 and i > 0:
        #     agent.save_model(os.path.join("models", f"model_{i}_iter.pt"))

env.close()