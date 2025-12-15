**Experiment Reproducibility — Lift & Stack**

This document describes the exact experimental setup used to evaluate the SORS-style waypoint planning system on the `Lift` and `Stack` tasks. It explains simulator details, robot and observation information, key hyperparameters, how to run training and evaluation, and the evaluation metrics so other researchers can reproduce results.

**Repository entry**: run experiments from the repository root. The primary training code lives in `rlwp/` and uses the task configs in `rlwp/cfg/task/`.

- **Train command:**
``powershell
python main.py train=True task=Lift run_name=sors_lift_<seed>
python main.py train=True task=Stack run_name=sors_stack_<seed>
``

**Simulator & Robot**
- **Simulator:** `robosuite` (see `requirements.txt` / `setup.py` in repo). The experiments use the `Panda` robot.
- **Controller:** `OSC_POSE` controller loaded via `load_controller_config(default_controller="OSC_POSE")` in `rlwp/train.py`.
- **Control frequency:** `10 Hz` (passed as `control_freq=10` in environment constructor).
- **Rendering:** off by default (`render=False` in config). Enable `render=True` in `rlwp/cfg/config.yaml` for visual debugging.
- **Reward shaping:** `reward_shaping=False` in `train.py` to use the sparse task reward. The code adds a small shaped grasp bonus (`shaped_grasp_reward`) to help the reward model observe successful lifts.

**Tasks & Observation Details**
- Tasks used for experiments: `Lift` and `Stack` (configs under `rlwp/cfg/task/`). Use the config names `Lift` and `Stack` when calling `main.py`.
- `Lift`:
  - `state_dim: 4` (end-effector x,y,z and gripper channel)
  - `objs` vector: `obs['cube_pos']` (x,y,z)
  - `wp_steps: 50`, `gripper_steps: 10`
- `Stack`:
  - `state_dim: 4` (per waypoint)
  - `objs` vector: `np.concatenate((obs['cubeA_pos'], obs['cubeB_pos']))`
  - `wp_steps: 50`, `gripper_steps: 15`

**SORS algorithm specifics implemented**
- **Ensemble size (N):** configured via `ensemble_size` in task YAMLs (default used in experiments: `10`). Networks are `RNetwork` MLPs with `hidden_dim=128`.
- **Global replay buffer Dτ:** `MyMemory` (in `rlwp/memory.py`). Trajectories are stored as `np.concatenate((traj_full, objs))` together with the sparse episode return.
- **Global episode counter:** training uses `global_episode` from `1 .. epoch_wp * num_wp` and passes it to `Method.traj_opt()` so all phases use a global time reference.
- **Waypoint phases:** (parameters in task YAML)
  - `exploration_epoch`: purely stochastic exploration for current waypoint.
  - `ensemble_sampling_epoch`: single-model sampling (random ensemble member) used for optimization.
  - after `ensemble_sampling_epoch`: ensemble mean optimization.
  - `averaging_noise_epoch`: optional averaging/noise phase used to encourage exploration around averaged solutions.
- **SORS periodic training schedule:** parameters `Pr` (period in episodes) and `Nr` (number of minibatch updates). When `(global_episode % Pr == 0)` and `len(Dτ) > batch_size`, run `Nr` minibatch updates across the current ensemble (implemented in `train.py`).
- **Diversity reset:** with probability `0.05` per episode, up to `rand_reset_epoch`, randomly reinitialize one ensemble member to preserve diversity.

**Key hyperparameters (as used in our experiments)**
- `num_wp`: 2 (use the repo default in `rlwp/cfg/config.yaml`)
- `n_inits`: 5 (SLSQP restarts per waypoint)
- `ensemble_size`: 10
- Reward network: `hidden_size=128`, optimizer `Adam(lr=1e-3)`
- Planner: SLSQP with `eps=1e-6`, `maxiter=1e6` (practically capped by runtime)
- SORS schedule examples (present in task YAMLs):
  - `Pr: 10`, `Nr: 50`  (update every 10 episodes, 50 minibatch steps)
  - `batch_size`: task-specific (30 in default configs)
  - `epoch_wp`: per-task (e.g., 200 for Lift, 300 for Stack in defaults)

**Reproducible experiment protocol**
1. For each task (`Lift`, `Stack`) and each random seed (we recommend at least 5 seeds):
   - Set `run_name` to a unique string (e.g., `sors_lift_seed1`).
   - Run training: `python main.py train=True task=Lift run_name=sors_lift_seed1`.
2. Evaluation during training:
   - The training loop logs `reward` (episode return), `model/critic_loss`, and prints `Predicted` (ensemble mean prediction) vs `Reward_full` every episode.
   - Optionally evaluate fixed-policy performance every K episodes by running `main.py` in evaluation mode (see `main.py` / `rlwp/eval.py`).
3. Aggregation:
   - For each run, save `models/<save_name>/` and `models/<save_name>/data.pkl` (reward curves).
   - Compute mean ± std of episode return at checkpoints (e.g., every 10 or 50 episodes) across seeds.

**Evaluation metrics**
- **Primary:** Episode return (`episode_reward`) — sparse environment reward (plus the optional small shaped grasp bonus used to reveal lifts to the reward learner). Higher is better.
- **Secondary:** Predicted reward (`Predicted`) computed by `agent.get_avg_reward(traj)` — used to measure model fidelity / how well learned reward aligns with env return.
- **Per-waypoint:** `train_reward` (reward accumulated during final waypoint segment) — used to select and initialize best trajectories for the next waypoint.

**Model & checkpoint layout**
- Saved models: `models/<save_name>/wp_<wp_id>/model_<i>.pt` for ensemble member `i` and waypoint `wp_id`.
- Reward curve: `models/<save_name>/data.pkl` (pickled dict with `episode` and `reward`).
- TensorBoard logs: `runs/ours_<run_name>_<time>/`.

**Quick tips for reproducibility**
- Make sure `robosuite` and dependencies are installed in the environment used for experiments. Follow the instructions in the repo root `README` and `robosuite` docs.
- Fix random seeds in `main.py` if you want bitwise reproducibility across runs (not currently enforced by default).
- If you see training instability, shorten `epoch_wp` and run a smoke test for a few episodes to check for runtime errors.

**Contact / further help**
If you want I can add:
- a small `run_experiments.sh` / PowerShell script to run multiple seeds and aggregate results,
- a small script to produce plots from `models/<save_name>/data.pkl` (e.g., `rlwp/plot_rewards.py` exists and can be adapted), or
- an option in `main.py` to toggle `reward_shaping` and a `--dry_run` mode to exercise one episode.

---
This README is intentionally concise but contains the exact commands, file locations, and hyperparameters you need to reproduce the Lift and Stack experiments reported in the code. If you'd like, I can also add an example `results/` folder with an example run and a short plotting script.
