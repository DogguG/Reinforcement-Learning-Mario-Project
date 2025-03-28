import torch
import gym_super_mario_bros
import lime
import skimage
import matplotlib
import numpy

print("NumPy version:", numpy.__version__)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Gym-Super-Mario-Bros version:", gym_super_mario_bros.__version__)
print("LIME installed:", lime.__version__ if hasattr(lime, '__version__') else "Yes")
print("Scikit-Image installed:", skimage.__version__)
print("Matplotlib installed:", matplotlib.__version__)