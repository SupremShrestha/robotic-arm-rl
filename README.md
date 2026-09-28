# Robotic Arm RL

Reinforcement Learning Based Robotic Arm for Target Reaching Using PyBullet
(KUKA iiwa, Gymnasium environment, Stable-Baselines3 PPO).

## Setup (Windows, PowerShell)
    py -3.11 -m venv venv
    venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    python check_install.py

## Train
    python -m training.train
