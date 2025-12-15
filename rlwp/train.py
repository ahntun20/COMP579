import numpy as np
import gym
import datetime
from torch.utils.tensorboard import SummaryWriter
import robosuite as suite
from robosuite.controllers import load_controller_config
import pickle
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import time
import os, sys

from memory import MyMemory
from method import Method

# Cross-platform non-blocking keypress check. On Windows use msvcrt; otherwise use select on stdin.
try:
    import msvcrt
    def exit_key_pressed():
        """Return True if user pressed 'q', 'Q' or ESC (non-blocking)."""
        if msvcrt.kbhit():
            ch = msvcrt.getwch()
            if ch in ('q', 'Q', '\x1b'):
                return True
        return False
except Exception:
    import select, tty, termios
    def exit_key_pressed():
        """Unix fallback: non-blocking check on stdin for 'q' or ESC. May require terminal focus."""
        dr, _, _ = select.select([sys.stdin], [], [], 0)
        if dr:
            ch = sys.stdin.read(1)
            if ch in ('q', 'Q', '\x1b'):
                return True
        return False


class train:
    def __init__(self, config):
        self.config = config
        self.env = config['task']['name']
        self.num_wp = config['num_wp']
        self.render = config['render']
        self.n_inits = config['n_inits']
        self.run_name = config['run_name']
        self.object = None if config['object']=='' else config['object']

        self.robot = config['task']['env']['robot']
        self.state_dim = config['task']['env']['state_dim']
        self.gripper_steps = config['task']['env']['gripper_steps']
        self.wp_steps = config['task']['env']['wp_steps']
        self.use_latch = config['task']['env']['use_latch']

        # shaped reward parameters (help agent observe successful grasps)
        self.shaped_grasp_reward = config['task']['task'].get('shaped_grasp_reward', 5.0)
        self.lift_threshold = config['task']['task'].get('lift_threshold', 0.02)

        # optional debug flags (can set in config later)
        self.debug_gripper = config.get('debug', {}).get('debug_gripper', False)
        self.force_grasp = config.get('debug', {}).get('force_grasp', False)

        self.batch_size = config['task']['task']['batch_size']
        self.epoch_wp = config['task']['task']['epoch_wp']
        self.rand_reset_epoch = config['task']['task']['rand_reset_epoch']

        # NEW: SORS update schedule (required in config)
        self.Pr = config['task']['task']['Pr']   # update period in episodes
        self.Nr = config['task']['task']['Nr']   # number of gradient steps per update

        self.train()

    def reset_env(self, env, get_objs=False):
        obs = env.reset()
        if get_objs:
            if self.env == 'Lift':
                objs = obs['cube_pos']
            elif self.env == 'Stack':
                objs = np.concatenate((obs['cubeA_pos'], obs['cubeB_pos']), axis=-1) 
            elif self.env == 'NutAssembly':
                nut = 'RoundNut'
                objs = obs[nut + '_pos']
            elif self.env == 'PickPlace':
                objs = obs[self.object+'_pos']
            elif self.env == 'Door':
                objs = np.concatenate((obs['door_pos'], obs['handle_pos']), axis=-1)
            return obs, objs
        return obs

    def get_state(self, obs):
        if self.env == 'Door':
            robot_pos = obs['robot0_eef_pos']
            robot_ang = R.from_quat(obs['robot0_eef_quat']).as_euler('xyz', degrees=False)
            state = np.concatenate((robot_pos, robot_ang), axis=-1)
            return state

        state = obs['robot0_eef_pos']
        return state

    def get_action(self, env, wp_idx, state, traj_mat, gripper_mat, time_s, timestep):
        error = traj_mat[wp_idx, :] - state
        if timestep < 10:
            full_action = np.array(list(10.*error) + [0.]*(6-len(state)) +[-1.])
        elif time_s >= self.wp_steps - self.gripper_steps:
            # gripper commands from trajectory may be small; clamp and optionally force full close
            g = float(np.array(gripper_mat[wp_idx]).flatten()[0]) if np.array(gripper_mat[wp_idx]).size>0 else 0.0
            # optionally force full close (1.0) for debug testing
            if self.force_grasp:
                g = 1.0
            # clamp to valid range
            g = float(np.clip(g, -1.0, 1.0))
            full_action = np.array([0.]*6 + [g])
        else:
            full_action = np.array(list(10.*error)  +[0.]*(6-len(state)) + [0.])

        if self.debug_gripper:
            # try to log the last element (gripper) and return it for inspection
            try:
                sys.stdout.write(f"t:{timestep} wp:{wp_idx} gripper_cmd:{full_action[-1]:.3f}\n")
            except Exception:
                pass

        return full_action

    def train(self):
        save_data = {'episode': [], 'reward': []}

        if self.object is None:
            save_name = self.env + '/' + self.run_name
            if self.env == 'Door' and self.use_latch:
                save_name = self.env + '/with_latch/' + self.run_name
            elif self.env == 'Door' and not self.use_latch:
                save_name = self.env + 'without_latch/' + self.run_name
        else:
            save_name = self.env + '/' + self.object + '/' + self.run_name

        controller_config = load_controller_config(default_controller="OSC_POSE")

        env = suite.make(
            env_name=self.env,
            robots=self.robot,
            controller_configs=controller_config,
            has_renderer=self.render,
            reward_shaping=True,   # R_env (can be made sparse if desired)
            control_freq=10,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            initialization_noise=None,
            # single_object_mode=2,
            # object_type=self.object,
            # use_latch=self.use_latch,
        )

        # Start from first waypoint
        wp_id = 1
        obs, objs = self.reset_env(env, get_objs=True)
        agent = Method(
            state_dim=self.state_dim,
            objs=objs,
            wp_id=wp_id,
            save_name=save_name,
            config=self.config
        )

        # Global buffer D_tau (persistent across waypoints)
        # Pre-compute maximum stored trajectory vector length so samples can stack
        max_traj_len = self.num_wp * self.state_dim + agent.objs_len
        memory = MyMemory(traj_size=max_traj_len)

        run_name = 'runs/ours_' + self.run_name + datetime.datetime.now().strftime("%H-%M")
        writer = SummaryWriter(run_name)

        # Total number of episodes = epoch_wp per waypoint
        EPOCHS = self.epoch_wp * self.num_wp

        total_steps = 0

        for global_episode in tqdm(range(1, EPOCHS + 1)):
            # Optional diversity: random reset of one ensemble member (SORS: rand() < 0.05)
            if (
                np.random.rand() < 0.05
                and global_episode < self.rand_reset_epoch
                and global_episode > 1
            ):
                agent.reset_model(np.random.randint(agent.n_models))

            episode_reward = 0.0
            done, truncated = False, False
            obs, objs = self.reset_env(env, get_objs=True)

            # Build trajectory using current SORS-style selection rules
            traj_full = agent.traj_opt(global_episode, objs)

            state = self.get_state(obs)
            traj_mat = np.reshape(traj_full, (wp_id, self.state_dim))[:, :self.state_dim - 1] + state
            gripper_mat = np.reshape(traj_full, (wp_id, self.state_dim))[:, self.state_dim - 1:]

            time_s = 0
            train_reward = 0.0

            for timestep in range(wp_id * self.wp_steps):
                if self.render:
                    env.render()

                state = self.get_state(obs)
                wp_idx = timestep // 50

                action = self.get_action(self.env, wp_idx, state, traj_mat, gripper_mat, time_s, timestep)

                time_s += 1
                if time_s >= 50:
                    time_s = 1

                obs, reward, done, _ = env.step(action)
                episode_reward += reward

                # Reward attributed to final waypoint segment
                if timestep // 50 == wp_id - 1:
                    train_reward += reward

                total_steps += 1

            # D_tau: store (trajectory, env return R_env)
            memory.push(np.concatenate((traj_full, objs)), episode_reward)

            save_data['episode'].append(global_episode)
            save_data['reward'].append(episode_reward)

            # SORS-style periodic reward updates:
            # if global_episode mod Pr == 0 and |D_tau| > batch_size:
            #       repeat Nr times: sample minibatch & update ensemble
            if (global_episode % self.Pr == 0) and (len(memory) > self.batch_size):
                for _ in range(self.Nr):
                    critic_loss = agent.update_parameters(memory, self.batch_size)
                    writer.add_scalar('model/critic_loss', critic_loss, total_steps)

            # Track best trajectory for this waypoint
            if train_reward > agent.best_reward:
                agent.set_init(traj_full, train_reward)
                agent.save_model(save_name)

            writer.add_scalar('reward', episode_reward, global_episode)
            tqdm.write(
                "wp_id: {}, GlobalEp: {}, Reward_full: {}; Reward: {}, Predicted: {}".format(
                    wp_id,
                    global_episode,
                    round(episode_reward, 2),
                    round(train_reward, 2),
                    round(agent.get_avg_reward(traj_full), 2),
                )
            )

            # Save reward curve (ensure directory exists)
            save_dir = os.path.join('models', save_name)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            pickle.dump(save_data, open(os.path.join(save_dir, 'data.pkl'), 'wb'))

            # At the *end* of each waypoint phase, save ensemble and move to next waypoint
            if (global_episode % self.epoch_wp == 0) and (wp_id < self.num_wp):
                agent.save_model(save_name)
                wp_id += 1
                # Rebuild agent for next waypoint (R buffer grows via learned_models)
                agent = Method(
                    state_dim=self.state_dim,
                    objs=objs,
                    wp_id=wp_id,
                    save_name=save_name,
                    config=self.config
                )

        # Final save for last waypoint
        agent.save_model(save_name)
        exit()


