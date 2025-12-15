# Enhancing Waypoint-Based Reinforcement Learning with Self-Supervised Dense Reward Models 

This repository provides our implementation of Waypoint-Based RL.

## System requirement
I did this project on Anaconda environments in Windows 11. The requirement package was listed in requirement.txt

##  Window Installation
- Anaconda: https://www.anaconda.com/
- Robosuite: https://robosuite.ai/docs/installation.html
    + Setting up a Conda environment by installing **Anaconda** and running **conda create -n robosuite python=3.8**
    + pip install robosuite
    + More information on how to install robosuite on Windows can be refers to this link https://robosuite.ai/docs/installation.html
    + If you encounter issue installing with **egl**, you refer to this link https://github.com/huggingface/lerobot/issues/105. I found solution is pip install   
      https://github.com/mhandb/egl_probe/archive/fix_windows_build.zip
- After you're done installation with **robosuite**, you can install **robomimic** in the env name "robosuite" by using pip install robomimic
- You might need to install MuJoCo 2.1 to local machine to able to run the experiment on Windows
    + Download the MuJoCo version 2.1 binaries for Linux or OSX.
    + Extract the downloaded mujoco210 directory into ~/.mujoco/mujoco210.
- You might need to install Visual Studio Professional 2019, to make it compatible with Mujuco 2.1 and other softwares to make it run on Window.
## Project Installation
Clone the repository using 
```bash
https://github.com/ahntun20/COMP765.git
```

## Implementation of rl-waypoints
Navigate to the rl-waypoints repository using 
```bash
cd rl-waypoints
cd rlwp
```

To train the robot for manipulation tasks in the Robosuite simulation environment, run the following commands
```bash
python main.py train=True task=<task_name>
```
The complete set of arguments can be found in the \cfg folder. The tasks for the training can be from the following: {Lift, Stack}

## Testing the trained models
We provide a trained model for each task for our approach. The trained models can be evaluated as follows
```bash
python main.py test=True task=<task_name> run_name=test render=True
```
