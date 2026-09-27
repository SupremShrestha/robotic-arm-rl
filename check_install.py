import pybullet
import gymnasium
import stable_baselines3
import numpy
import matplotlib
import torch

print("PyBullet:", pybullet.getAPIVersion())
print("Gymnasium:", gymnasium.__version__)
print("Stable-Baselines3:", stable_baselines3.__version__)
print("NumPy:", numpy.__version__)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

print("\nAll imports successful.")