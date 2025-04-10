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
from skimage import segmentation, feature, measure

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

def detect_obstacles(state_rgb):
    img_gray = np.mean(state_rgb, axis=2).astype(np.uint8)
    edges = feature.canny(img_gray, sigma=1.0)
    obstacle_mask = np.zeros_like(img_gray, dtype=bool)
    obstacle_mask[40:70, :] = True
    obstacle_edges = edges & obstacle_mask
    segments = segmentation.slic(state_rgb, n_segments=50, compactness=10, sigma=1)
    obstacle_segment_mask = np.zeros_like(segments, dtype=bool)
    for region in measure.regionprops(segments + 1):
        region_id = region.label - 1
        region_coords = region.coords
        if np.any(obstacle_edges[region_coords[:, 0], region_coords[:, 1]]):
            if np.all(region_coords[:, 0] > 70):
                continue
            obstacle_segment_mask[segments == region_id] = True
    return obstacle_segment_mask

def custom_segmentation(image):
    img_gray = np.mean(image, axis=2).astype(np.uint8)
    edges = feature.canny(img_gray, sigma=1.0)
    obstacle_mask = np.zeros_like(img_gray, dtype=bool)
    obstacle_mask[40:70, :] = True
    obstacle_edges = edges & obstacle_mask
    segments = segmentation.slic(image, n_segments=50, compactness=10, sigma=1)
    obstacle_segment_mask = np.zeros_like(segments, dtype=bool)
    for region in measure.regionprops(segments + 1):
        region_id = region.label - 1
        region_coords = region.coords
        if np.any(obstacle_edges[region_coords[:, 0], region_coords[:, 1]]):
            if np.all(region_coords[:, 0] > 70):
                continue
            obstacle_segment_mask[segments == region_id] = True
    labels = np.zeros_like(segments, dtype=np.int32)
    labels[obstacle_segment_mask] = 1
    return labels

def state_to_rgb(state):
    frame = state[3]
    frame = np.repeat(frame[:, :, np.newaxis], 3, axis=2)
    return frame.astype(np.uint8)

def predict_fn(images, agent):
    processed_states = []
    for img in images:
        img_gray = np.mean(img, axis=2).astype(np.float32)
        img_stack = np.stack([img_gray] * 4, axis=0)
        processed_states.append(img_stack)
    processed_states = np.array(processed_states)
    states_tensor = torch.tensor(processed_states, dtype=torch.float32).to(agent.online_network.device)
    with torch.no_grad():
        q_values = agent.online_network(states_tensor)
    return q_values.cpu().numpy()

def explain_action(state, action, agent, episode, frame_idx, output_dir):
    state_rgb = state_to_rgb(state)
    explainer = lime_image.LimeImageExplainer()
    # Dự đoán Q-value trước khi giải thích
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(agent.online_network.device)
    with torch.no_grad():
        q_values = agent.online_network(state_tensor).cpu().numpy()[0]
    print(f"Episode {episode}, Frame {frame_idx} - Q-values: {q_values}")  # Sửa lỗi cú pháp

    explanation = explainer.explain_instance(
        state_rgb,
        lambda x: predict_fn(x, agent),
        top_labels=1,
        num_samples=2000,
        segmentation_fn=custom_segmentation
    )
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=5
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(state_rgb)
    ax1.set_title(f"Original State (Episode {episode}, Frame {frame_idx})")
    ax1.axis('off')
    ax2.imshow(mark_boundaries(state_rgb, mask))
    ax2.set_title(f"LIME Explanation: Action {action}, Q-value: {q_values[action]:.4f}")
    ax2.axis('off')
    plt.savefig(os.path.join(output_dir, f"episode_{episode}_frame_{frame_idx}.png"), bbox_inches='tight')
    plt.close(fig)

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

ENV_NAME = 'SuperMarioBros-1-1-v0'
NUM_OF_EPISODES = 5

env = gym_super_mario_bros.make(ENV_NAME, render_mode='rgb_array', apply_api_compatibility=True)
env = JoypadSpace(env, RIGHT_ONLY)
env = apply_wrappers(env)

agent = Agent(input_dims=env.observation_space.shape, 
              num_actions=env.action_space.n,
              network_class=AgentNN)

model_path = "D:/Project/Seerminar/Super-Mario-Bros-RL/models/mario_model/model_1500_iter.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file {model_path} not found!")
agent.load_model(model_path)
agent.epsilon = 0.0

output_dir = "lime_explanations"
os.makedirs(output_dir, exist_ok=True)

for i in range(NUM_OF_EPISODES):
    done = False
    state, _ = env.reset()
    rewards = 0
    frame_idx = 0
    while not done:
        action = agent.choose_action(state)
        new_state, reward, done, truncated, info = env.step(action)
        rewards += reward

        if frame_idx % 50 == 0 or action == 1:
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
                frame_skip_env = env.env.env.env
                frames_log = frame_skip_env.frames_log
                actions_log = frame_skip_env.actions_log
                for j, (frame, action) in enumerate(zip(frames_log, actions_log)):
                    scaling_factor = 10
                    new_dims = (frame.shape[1] * scaling_factor, frame.shape[0] * scaling_factor)
                    frame = np.array(frame).astype(np.uint8)
                    if frame.ndim == 2:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                    frame = cv2.resize(frame, new_dims, interpolation=cv2.INTER_NEAREST)
                    cv2.imwrite(os.path.join("games", f"game_{i}", f"frame_{j}.png"), frame)

env.close()

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