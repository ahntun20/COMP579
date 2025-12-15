import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize, LinearConstraint
import os
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize, LinearConstraint
import os
import random
from models import RNetwork
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize, LinearConstraint
import os
import random
from models import RNetwork
from tqdm import tqdm

class Method(object):
    def __init__(self, state_dim, objs, wp_id, save_name, config):
        # device-aware
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state_dim = state_dim
        self.wp_id = wp_id
        self.best_wp = []
        self.n_inits = 5
        self.lr = 1e-3
        self.hidden_size = 128
        # ensemble size: allow override from config
        self.n_models = config['task']['task'].get('ensemble_size', 10)
        self.models = []                # ensemble for the current waypoint (critic, optim)
        self.learned_models = []        # list of ensembles from previous waypoints
        self.n_eval = 100

        # length of object vector (used to build network input dim)
        self.objs_len = len(objs)

        # scheduling params (phases)
        self.action_dim = config['task']['task']['action_space']
        self.exploration_epoch = config['task']['task']['exploration_epoch']
        self.ensemble_sampling_epoch = config['task']['task']['ensemble_sampling_epoch']
        self.averaging_noise_epoch = config['task']['task']['averaging_noise_epoch']

        # initialize ensemble for current waypoint
        for _ in range(self.n_models):
            traj_dim = self.wp_id * self.state_dim + self.objs_len
            critic = RNetwork(traj_dim, hidden_dim=self.hidden_size).to(device=self.device)
            optimizer = torch.optim.Adam(critic.parameters(), lr=self.lr)
            self.models.append((critic, optimizer))
        # Save name for loading/saving per-waypoint models
        self.save_name = save_name

        # Load previously-saved waypoint ensembles (SORS-style storage)
        self.load_previous_models()

        # best trajectory buffer used for initialization between waypoint updates
        # keep as flat vector (length = wp_id * state_dim) for compatibility with
        # other code that slices by flat indices. If the state includes a gripper
        # channel (index >= 3), initialize that channel per-waypoint with
        # uniform random values in [-1, 1] so gripper targets vary across WPs.
        self.best_traj = self.action_dim * (np.random.rand(self.state_dim * self.wp_id) - 0.5)
        if self.state_dim > 3 and self.wp_id > 0:
            try:
                bt = self.best_traj.reshape(self.wp_id, self.state_dim)
                bt[:, 3] = np.random.uniform(-1, 1, size=self.wp_id)
                self.best_traj = bt.flatten()
            except Exception:
                # if reshape fails for any reason, leave the flat initialization as-is
                pass
        self.best_reward = -np.inf
        self.lincon = LinearConstraint(np.eye(self.state_dim), -self.action_dim, self.action_dim)

    def traj_opt(self, global_episode, objs):
        """
        Build waypoint set using prior learned ensembles and the phased selection rules
        global_episode: integer episode counter (global across training)
        objs: raw object vector from environment (numpy)
        Returns: flattened trajectory for all waypoints up to self.wp_id
        """
        # choose a model index for single-model sampling phase
        self.reward_idx = random.choice(range(self.n_models))
        self.objs = torch.FloatTensor(objs).to(device=self.device)
        self.curr_episode = global_episode

        self.traj = []

        # iterate through waypoints 1..wp_id
        for idx in range(1, self.wp_id + 1):
            min_cost = np.inf

            self.load_model = True if idx != self.wp_id else False
            self.curr_wp = idx - 1

            # exploration-only initialization for the current waypoint
            if idx == self.wp_id and global_episode <= self.exploration_epoch:
                # random pose initialization
                self.best_wp = self.action_dim * (np.random.rand(self.state_dim) - 0.5)
                # force the gripper channel (if present) to a close command with
                # some probability so the environment occasionally experiences
                # grasp attempts during early exploration. This helps the reward
                # model observe successful lifts and break the feedback loop.
                if self.state_dim > 3 and np.random.rand() < 0.5:
                    # set gripper target to +1 (close)
                    try:
                        self.best_wp[3] = 1.0
                    except Exception:
                        pass

            else:
                # optimize waypoint with SLSQP using the appropriate reward (prior ensembles are averaged)
                for t_idx in range(self.n_inits):
                    start_idx = self.curr_wp * self.state_dim
                    xi0 = np.copy(self.best_traj[start_idx:start_idx + self.state_dim])
                    if t_idx != 0:
                        xi0 += np.random.normal(0, 0.1, size=self.state_dim)

                    res = minimize(self.get_cost, xi0, method='SLSQP', constraints=self.lincon,
                                   options={'eps': 1e-6, 'maxiter': 1e6})
                    if res.success and res.fun < min_cost:
                        min_cost = res.fun
                        self.best_wp = res.x

                # for the current waypoint, apply small averaging/noise depending on phase
                if idx == self.wp_id:
                    # single-model sampling or ensemble mean selection is handled in get_reward()
                    # optionally add small noise during early averaging phase (config-defined)
                    if global_episode < self.averaging_noise_epoch:
                        # small pose noise
                        if np.random.rand() < 0.5:
                            self.best_wp[:3] += np.random.normal(0, 0.02, 3)
                        # small gripper noise
                        if np.random.rand() < 0.2:
                            self.best_wp[-1] += np.random.normal(0, 0.02)

            self.traj.append(self.best_wp)

        return np.array(self.traj).flatten()

    def set_init(self, traj, reward):
        self.best_reward = reward
        self.best_traj = np.copy(traj)

    def get_cost(self, traj):
        """
        xi: numpy array for the current waypoint (last waypoint in the prefix)

        We build the full trajectory vector exactly as during training:
            full_traj = [all previous best wps, xi, objs]
            length = idx * state_dim + len(objs)
        """
        xi_t = torch.FloatTensor(traj).to(device=self.device)

        # previous waypoints already chosen in this traj_opt call
        if len(self.traj) > 0:
            traj_learnt = torch.FloatTensor(np.array(self.traj).flatten()).to(device=self.device)
            full = torch.cat((traj_learnt, xi_t), dim=0)
        else:
            full = xi_t

        # append object features
        full = torch.cat((full, self.objs), dim=0)

        # full is now 1D tensor with correct length for the critic
        reward = self.get_reward(full)
        return -reward

    def get_reward(self, full_traj_tensor, global_episode=None):
        """SORS-style get_reward that assumes `traj` is already the full 1D vector
        of shape (idx * state_dim + len(objs)).
        """
        traj = full_traj_tensor.to(device=self.device).unsqueeze(0)

        # ---------------------------------------------
        # Case 1: previous waypoints (load pretrained)
        # ---------------------------------------------
        if self.load_model:
            # self.curr_wp is idx-1, previous waypoint id is idx
            wp_id = self.curr_wp + 1
            models = None
            if hasattr(self, 'prev_reward_models'):
                models = self.prev_reward_models.get(wp_id, None)
            # fallback for older storage format
            if models is None and len(self.learned_models) > self.curr_wp:
                try:
                    models = self.learned_models[self.curr_wp]
                except Exception:
                    models = None

            if models is None or len(models) == 0:
                return 0.0

            val = 0.0
            for critic in models:
                with torch.no_grad():
                    val += critic(traj).item()
            return val / len(models)

        # ---------------------------------------------
        # Case 2: current waypoint (online ensemble)
        # exploration handled in traj_opt (we won't call get_cost then)
        # ---------------------------------------------
        if getattr(self, 'curr_episode', 0) < self.ensemble_sampling_epoch:
            critic, _ = self.models[self.reward_idx]
            with torch.no_grad():
                return critic(traj).item()

        # ensemble mean
        val = 0.0
        for critic, _ in self.models:
            with torch.no_grad():
                val += critic(traj).item()
        return val / len(self.models)

    def get_avg_reward(self, traj):
        reward = 0.0
        traj = torch.FloatTensor(traj).to(device=self.device)
        traj = torch.cat((traj, self.objs))
        traj = traj.unsqueeze(0)
        for critic, _ in self.models:
            with torch.no_grad():
                reward += critic(traj).item()
        return reward / self.n_models

    def load_previous_models(self):
        """
        Loads all reward ensembles for all previous waypoints into:
            self.prev_reward_models[wp_id] = [model_1, ..., model_N]
        """
        self.prev_reward_models = {}

        # nothing to load for wp 1
        if self.wp_id == 1:
            return

        base = os.path.join('models', self.save_name)
        for w in range(1, self.wp_id):
            path = os.path.join(base, f'wp_{w}')
            models_w = []
            if not os.path.exists(path):
                continue
            for n in range(self.n_models):
                model_path = os.path.join(path, f'model_{n}.pt')
                if not os.path.exists(model_path):
                    continue
                traj_dim = w * self.state_dim + self.objs_len
                critic = RNetwork(traj_dim, hidden_dim=self.hidden_size).to(device=self.device)
                state_dict = torch.load(model_path, map_location=self.device)
                critic.load_state_dict(state_dict)
                critic.eval()
                models_w.append(critic)
            if len(models_w) > 0:
                self.prev_reward_models[w] = models_w

        print(f"[INFO] Loaded previous reward models up to wp {self.wp_id - 1}")

    def update_parameters(self, memory, batch_size):
        """Run one minibatch update for every model in the current ensemble and return mean loss."""
        loss = np.zeros((self.n_models,))
        for idx, (critic, optim) in enumerate(self.models):
            loss[idx] = self.update_critic(critic, optim, memory, batch_size)
        return float(np.mean(loss))

    def update_critic(self, critic, optim, memory, batch_size):
        trajs, rewards = memory.sample(batch_size)
        trajs = torch.FloatTensor(trajs).to(device=self.device)
        rewards = torch.FloatTensor(rewards).to(device=self.device).unsqueeze(1)

        # Critics were created with input dim depending on current wp_id. Memory stores
        # padded trajectories (max length). Slice the batch to the critic's expected input size.
        try:
            in_dim = critic.linear1.in_features
        except Exception:
            # fallback: assume trajs already match
            in_dim = trajs.shape[1]

        if trajs.shape[1] > in_dim:
            trajs = trajs[:, :in_dim]

        rhat = critic(trajs)
        loss = F.mse_loss(rhat, rewards)
        optim.zero_grad()
        loss.backward()
        optim.step()
        return loss.item()

    def reset_model(self, idx):
        traj_dim = self.wp_id * self.state_dim + getattr(self, 'objs_len', 0)
        critic = RNetwork(traj_dim, hidden_dim=self.hidden_size).to(device=self.device)
        optimizer = torch.optim.Adam(critic.parameters(), lr=self.lr)
        self.models[idx] = (critic, optimizer)
        tqdm.write("RESET MODEL {}".format(idx))

    def save_model(self, save_name: str = None):
        """
        Save reward ensemble for the current waypoint:
            models/<save_name>/wp_<wp_id>/model_i.pt
        """
        if save_name is None:
            save_name = self.save_name
        path = os.path.join('models', save_name, f'wp_{self.wp_id}')
        os.makedirs(path, exist_ok=True)

        for i, (critic, _) in enumerate(self.models):
            torch.save(critic.state_dict(), os.path.join(path, f'model_{i}.pt'))

        print(f"[SAVE] Saved reward ensemble for wp {self.wp_id} into {path}")

