import argparse
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt


def moving_average(x, w):
    if w <= 1:
        return x
    return np.convolve(x, np.ones(w) / w, mode='valid')


def load_data(pickle_path):
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(pickle_path)
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    # Expecting dict with 'episode' and 'reward'
    if isinstance(data, dict):
        if 'episode' in data and 'reward' in data:
            episodes = np.array(data['episode'])
            rewards = np.array(data['reward'])
            return episodes, rewards
        # some code uses list of tuples or other formats
        # try to find obvious keys
        for k in ('ep', 'eps', 'episodes'):
            if k in data:
                episodes = np.array(data[k])
                rewards = np.array(data.get('reward', data.get('rewards')))
                return episodes, rewards

    # fallback: try to interpret as sequence of (episode,reward) pairs
    try:
        arr = np.array(data)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return arr[:, 0], arr[:, 1]
    except Exception:
        pass

    raise ValueError('Unrecognized data format in {}'.format(pickle_path))


def plot_rewards(episodes, rewards, output_path=None, smooth=1, show=False):
    # Sort by episode in case it's out of order
    order = np.argsort(episodes)
    eps = episodes[order]
    r = rewards[order]

    plt.figure(figsize=(8, 4.5))
    plt.plot(eps, r, alpha=0.4, label='reward')

    if smooth and smooth > 1:
        r_smooth = moving_average(r, smooth)
        # align x for smoothed series
        eps_smooth = eps[(smooth - 1) :]
        plt.plot(eps_smooth, r_smooth, color='C1', label=f'moving_avg({smooth})')

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward vs Episode')
    plt.grid(alpha=0.3)
    plt.legend()

    if output_path:
        out_dir = os.path.dirname(output_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        print('Saved plot to', output_path)

    if show:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot reward vs episode from saved pickle')
    parser.add_argument('--file', '-f', required=True, help='Path to pickle file (data.pkl or eval_data.pkl)')
    parser.add_argument('--output', '-o', default=None, help='Output PNG path (optional)')
    parser.add_argument('--smooth', '-s', type=int, default=1, help='Moving average window (default 1 = no smoothing)')
    parser.add_argument('--show', action='store_true', help='Show plot window')

    args = parser.parse_args()

    episodes, rewards = load_data(args.file)
    if args.output is None:
        # default output filename next to pickle
        base = os.path.splitext(args.file)[0]
        args.output = base + '_reward_plot.png'

    plot_rewards(episodes, rewards, output_path=args.output, smooth=args.smooth, show=args.show)


if __name__ == '__main__':
    main()
