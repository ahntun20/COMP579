# Enhancing Waypoint-Based Reinforcement Learning with Self-Supervised Dense Reward Models 

This repository provides our implementation of Waypoint-Based RL.

## Installation
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
