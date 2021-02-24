import os
import gym
import torch
import pickle
import pprint
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from tianshou.policy import DQNPolicy
from tianshou.env import DummyVectorEnv
from tianshou.utils.net.common import Net
from tianshou.trainer import offpolicy_trainer
from tianshou.data import Collector, ReplayBuffer, PrioritizedReplayBuffer

from gym import utils
from gym.envs.robotics import fetch_env

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'pick_and_place.xml')

class EnvSpec:
    def __init__(self, reward_threshold: float=100):
        self.reward_threshold = reward_threshold

class FetchPickAndPlaceEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', t_max: int=1000):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)
        
        # Customized Variables
        self.previous_achieved_goal = None
        self.t_max = t_max
        self.t_current = 0
        self.spec = EnvSpec(
            reward_threshold=100
        )

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = fetch_env.goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        elif self.reward_type == 'dense':
            return -d
        elif self.reward_type == 'dense_on_move':
            was_moved = False
            if self.previous_achieved_goal is None:
                pass
            else:
                displacement = fetch_env.goal_distance(achieved_goal, self.previous_achieved_goal)
                if displacement > 0.0:
                    was_moved = True
            self.previous_achieved_goal = achieved_goal
            if d <= self.distance_threshold and was_moved:
                return -d
            elif d <= self.distance_threshold and not was_moved:
                return -d
            elif d > self.distance_threshold and was_moved:
                return -d * 0.5
            elif d > self.distance_threshold and not was_moved:
                return -d * 2.0
        elif self.reward_type == 'dense_on_move_else_sparse':
            was_moved = False
            if self.previous_achieved_goal is None:
                pass
            else:
                displacement = fetch_env.goal_distance(achieved_goal, self.previous_achieved_goal)
                if displacement > 0.0:
                    was_moved = True
            self.previous_achieved_goal = achieved_goal
            if was_moved:
                return -d * 0.5
            else:
                return -(d > self.distance_threshold).astype(np.float32) * 10.0
        else:
            raise ValueError(f'Invalid reward_type: {self.reward_type}')

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        self.t_current += 1
        # print(f'{self.t_current}/{self.t_max}')
        done = False if self.t_current < self.t_max else True
        if done:
            self.t_current = 0
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        # return obs, reward, done, info
        return obs['observation'], reward, done, info

def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--task', type=str, default='CartPole-v0')
    parser.add_argument('--seed', type=int, default=1626)
    parser.add_argument('--eps-test', type=float, default=0.05)
    parser.add_argument('--eps-train', type=float, default=0.1)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--target-update-freq', type=int, default=320)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--step-per-epoch', type=int, default=1000)
    parser.add_argument('--collect-per-step', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--hidden-sizes', type=int,
                        nargs='*', default=[128, 128, 128, 128])
    parser.add_argument('--training-num', type=int, default=8)
    parser.add_argument('--test-num', type=int, default=100)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument('--prioritized-replay',
                        action="store_true", default=False)
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--beta', type=float, default=0.4)
    # parser.add_argument(
    #     '--save-buffer-name', type=str,
    #     default="./expert_DQN_CartPole-v0.pkl")
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_known_args()[0]
    return args


def test_dqn(args=get_args()):
    reward_type = 'dense_on_move_else_sparse'
    env = FetchPickAndPlaceEnv(reward_type=reward_type)
    args.state_shape = env.observation_space['observation'].shape or env.observation_space['observation'].n
    # TODO: env.observation_space isn't the same as env.observation_space["observation"], and this is breaking the API.
    # Figure out how to fix this.
    args.action_shape = env.action_space.shape or env.action_space.n
    # you can also use tianshou.env.SubprocVectorEnv
    train_envs = DummyVectorEnv(
        [lambda: FetchPickAndPlaceEnv(reward_type=reward_type) for _ in range(args.training_num)])
    test_envs = DummyVectorEnv(
        [lambda: FetchPickAndPlaceEnv(reward_type=reward_type) for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # Q_param = V_param = {"hidden_sizes": [128]}
    # model
    net = Net(args.state_shape, args.action_shape,
              hidden_sizes=args.hidden_sizes, device=args.device,
              # dueling=(Q_param, V_param),
              ).to(args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    policy = DQNPolicy(
        net, optim, args.gamma, args.n_step,
        target_update_freq=args.target_update_freq)
    # buffer
    if args.prioritized_replay:
        buf = PrioritizedReplayBuffer(
            args.buffer_size, alpha=args.alpha, beta=args.beta)
    else:
        buf = ReplayBuffer(args.buffer_size)
    # collector
    train_collector = Collector(policy, train_envs, buf)
    test_collector = Collector(policy, test_envs)
    # policy.set_eps(1)
    train_collector.collect(n_step=args.batch_size)
    # log
    log_path = os.path.join(args.logdir, 'pick_and_place', 'dqn')
    writer = SummaryWriter(log_path)

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def stop_fn(mean_rewards):
        print(f'mean_rewards: {mean_rewards}')

        return mean_rewards >= env.spec.reward_threshold

    def train_fn(epoch, env_step):
        # eps annnealing, just a demo
        if env_step <= 10000:
            policy.set_eps(args.eps_train)
        elif env_step <= 50000:
            eps = args.eps_train - (env_step - 10000) / \
                40000 * (0.9 * args.eps_train)
            policy.set_eps(eps)
        else:
            policy.set_eps(0.1 * args.eps_train)

    def test_fn(epoch, env_step):
        policy.set_eps(args.eps_test)

    # trainer
    result = offpolicy_trainer(
        policy, train_collector, test_collector, args.epoch,
        args.step_per_epoch, args.collect_per_step, args.test_num,
        args.batch_size, train_fn=train_fn, test_fn=test_fn,
        stop_fn=stop_fn, save_fn=save_fn, writer=writer)

    # assert stop_fn(result['best_reward'])
    if stop_fn(result['best_reward']):
        print('Success!')
    else:
        print(f"Failure: result['best_reward'] == {result['best_reward']} < {env.spec.reward_threshold}")


    if __name__ == '__main__':
        pprint.pprint(result)
        # Let's watch its performance!
        env = FetchPickAndPlaceEnv(reward_type=reward_type)
        policy.eval()
        policy.set_eps(args.eps_test)
        collector = Collector(policy, env)
        result = collector.collect(n_episode=1, render=args.render)
        print(f'Final reward: {result["rew"]}, length: {result["len"]}')

    # save buffer in pickle format, for imitation learning unittest
    buf = ReplayBuffer(args.buffer_size)
    collector = Collector(policy, test_envs, buf)
    collector.collect(n_step=args.buffer_size)
    pickle.dump(buf, open('FetchPickAndPlaceEnv.pkl', "wb"))


def test_pdqn(args=get_args()):
    args.prioritized_replay = True
    args.gamma = .95
    args.seed = 1
    test_dqn(args)


if __name__ == '__main__':
    test_dqn(get_args())