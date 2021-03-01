import time
import numpy as np
import matplotlib.pyplot as plt
import psutil
import torch
from typing import List
from mpi4py import MPI
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy as dc
from .base import DDPG_HER_Base

class DDPG_HER_Trainer(DDPG_HER_Base):
    def __init__(
        self,
        weight_path: str,
        env_name: str="FetchPickAndPlace-v1",
        memory_size: float=7e+5//50,
        batch_size: int=256,
        actor_lr: float=1e-3,
        critic_lr: float=1e-3,
        gamma: float=0.98,
        tau: float=0.05,
        k_future: int=4
    ):
        super().__init__(
            weight_path=weight_path,
            env_name=env_name,
            memory_size=memory_size,
            batch_size=batch_size,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            gamma=gamma,
            tau=tau,
            k_future=k_future, # k_future = 4 -> future_p = 1 - (1. / (1 + k_future)) = 0.8
            is_train=True
        )
        self.t_success_rate = None
        self.total_ac_loss = None
        self.total_cr_loss = None

    def reset_metadata(self):
        self.t_success_rate = []
        self.total_ac_loss = []
        self.total_cr_loss = []

    def save_plot(
        self, save_path: str='plot.png',
        include: List[str]=['success', 'actor', 'critic']
    ):
        assert len(include) > 0
        plt.style.use('ggplot')
        plt.figure()
        num_subplots = len(include)
        base_plot_idx = 100 * num_subplots + 10 + 1
        mode_data_map = {
            'success': self.t_success_rate,
            'actor': self.total_ac_loss,
            'critic': self.total_cr_loss
        }
        data_len_list = [len(mode_data_map[mode]) for mode in include]
        unique_len_list = list(set(data_len_list))
        if len(unique_len_list) > 1:
            raise Exception('Plot data has different lengths.')
        num_epochs = unique_len_list[0]
        for i, mode in enumerate(include):
            if num_subplots > 1:
                plot_idx = base_plot_idx + i
                plt.subplot(plot_idx)
            if mode == 'success':
                plt.plot(np.arange(0, num_epochs), self.t_success_rate)
                plt.title("Success rate")
            elif mode == 'actor':
                plt.plot(np.arange(0, num_epochs), self.total_ac_loss)
                plt.title("Actor loss")
            elif mode == 'critic':
                plt.plot(np.arange(0, num_epochs), self.total_cr_loss)
                plt.title("Critic loss")
            else:
                raise ValueError(f'Invalid mode: {mode}')
        
        plt.savefig(save_path)

    def _reset_until_nontrivial_goal(self, env, thresh: float=0.05) -> dict:
        """Reset environment until a non-trivial goal is achieved.
        Triviality is determined by the distance between the
        achieved goal and desired goal.
        A non-trivial goal is considered to be a distance above a
        given threshold. e.g. 0.05 would be above 5 cm.

        Args:
            env: gym environment
            thresh (float, optional):
                Distance threshold.
                All goal distances lower than this are trivial and will be skipped.
                Defaults to 0.05.

        Returns:
            dict: environment dictionary
        """
        def calc_goal_distance(env_dict: dict) -> float:
            achieved_goal = env_dict["achieved_goal"]
            desired_goal = env_dict["desired_goal"]
            return np.linalg.norm(achieved_goal - desired_goal)
        
        env_dict = self.env.reset()
        while calc_goal_distance(env_dict) <= thresh:
            env_dict = self.env.reset()
        
        return env_dict

    def _get_episode_dict(self, nontrivial_goal_thresh: float=0.05, n_timesteps: int=50) -> dict:
        env_dict = self._reset_until_nontrivial_goal(env=self.env, thresh=nontrivial_goal_thresh) # Ensures non-trivial goal distance
        episode_dict = { # states, actions, and goals are accumulated for each episode
            "state": [],
            "action": [],
            "info": [],
            "achieved_goal": [],
            "desired_goal": [],
            "next_state": [],
            "next_achieved_goal": []
        }
        state = env_dict["observation"]
        achieved_goal = env_dict["achieved_goal"]
        desired_goal = env_dict["desired_goal"]
        for t in range(n_timesteps): # each episode is divided into timesteps that define each state
            action = self.agent.choose_action(state, desired_goal)
            next_env_dict, reward, done, info = self.env.step(action)

            next_state = next_env_dict["observation"]
            next_achieved_goal = next_env_dict["achieved_goal"]
            next_desired_goal = next_env_dict["desired_goal"]

            episode_dict["state"].append(state.copy())
            episode_dict["action"].append(action.copy())
            episode_dict["achieved_goal"].append(achieved_goal.copy())
            episode_dict["desired_goal"].append(desired_goal.copy())

            state = next_state.copy()
            achieved_goal = next_achieved_goal.copy()
            desired_goal = next_desired_goal.copy()

        episode_dict["state"].append(state.copy())
        episode_dict["achieved_goal"].append(achieved_goal.copy())
        episode_dict["desired_goal"].append(desired_goal.copy())

        # Remove first state and copy to next_state list
        episode_dict["next_state"] = episode_dict["state"][1:]
        # Remove first achieved goal and copy to next_achieved_goal list
        episode_dict["next_achieved_goal"] = episode_dict["achieved_goal"][1:]
        
        return episode_dict

    def _eval_agent(self, env_, agent_, nontrivial_goal_thresh: float=0.05, n_timesteps: int=50):
        total_success_rate = []
        running_r = []
        for ep in range(10):
            per_success_rate = []
            env_dictionary = self._reset_until_nontrivial_goal(env=env_, thresh=nontrivial_goal_thresh)
            s = env_dictionary["observation"]
            ag = env_dictionary["achieved_goal"]
            g = env_dictionary["desired_goal"]
            ep_r = 0
            for t in range(n_timesteps):
                with torch.no_grad():
                    a = agent_.choose_action(s, g, train_mode=False)
                observation_new, r, _, info_ = env_.step(a)
                s = observation_new['observation']
                g = observation_new['desired_goal']
                per_success_rate.append(info_['is_success'])
                ep_r += r
            total_success_rate.append(per_success_rate)
            if ep == 0:
                running_r.append(ep_r)
            else:
                running_r.append(running_r[-1] * 0.99 + 0.01 * ep_r)
        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate[:, -1])
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        return global_success_rate / MPI.COMM_WORLD.Get_size(), running_r, ep_r

    def train(
        self, max_epochs: int=50, max_cycles: int=50,
        max_episodes: int=2, num_updates: int=40,
        n_timesteps: int=50,
        nontrivial_goal_thresh: float=0.05,
        penalize_actions: bool=True,
        plot_save_path: str='plot.png',
        plot_include: List[str]=['success', 'actor', 'critic'],
        log_dir: str='logs'
    ):
        to_gb = lambda in_bytes: in_bytes / 1024 / 1024 / 1024
        log_writer = SummaryWriter(log_dir) if log_dir is not None else None

        self.reset_metadata()
        for epoch in range(max_epochs):
            start_time = time.time()
            epoch_actor_loss = 0
            epoch_critic_loss = 0
            for cycle in range(0, max_cycles): # multiple cycles per epoch
                mb = [] # Minibatch (refer to HER article pseudocode)

                # Standard Experience Replay
                # Store the transition (s_t||g, a_t, r_t, s_t+1||g) in R
                for episode in range(max_episodes): # multiple episodes per cycle
                    episode_dict = self._get_episode_dict(
                        nontrivial_goal_thresh=nontrivial_goal_thresh,
                        n_timesteps=n_timesteps
                    )
                    mb.append(dc(episode_dict))
                self.agent.store(mb)

                cycle_actor_loss = 0
                cycle_critic_loss = 0
                for n_update in range(num_updates): # For each update, the replay buffer is sampled to train the actor and critic.
                    actor_loss, critic_loss = self.agent.train(penalize_actions=penalize_actions) # HER comes into play when sampling from R in agent.train()
                    cycle_actor_loss += actor_loss
                    cycle_critic_loss += critic_loss

                epoch_actor_loss += cycle_actor_loss / num_updates
                epoch_critic_loss += cycle_critic_loss /num_updates
                self.agent.update_networks()

            ram = psutil.virtual_memory()
            success_rate, running_reward, episode_reward = self._eval_agent(
                self.env, self.agent,
                nontrivial_goal_thresh=nontrivial_goal_thresh,
                n_timesteps=n_timesteps
            )
            self.total_ac_loss.append(epoch_actor_loss)
            self.total_cr_loss.append(epoch_critic_loss)
            end_time = time.time()
            time_elapsed = end_time - start_time
            if MPI.COMM_WORLD.Get_rank() == 0:
                self.t_success_rate.append(success_rate)
                print(f"Epoch:{epoch}| "
                    f"Running_reward:{running_reward[-1]:.3f}| "
                    f"EP_reward:{episode_reward:.3f}| "
                    f"Memory_length:{len(self.agent.memory)}| "
                    f"Duration:{time_elapsed:.3f}| "
                    f"Actor_Loss:{actor_loss:.3f}| "
                    f"Critic_Loss:{critic_loss:.3f}| "
                    f"Success rate:{success_rate:.3f}| "
                    f"{to_gb(ram.used):.1f}/{to_gb(ram.total):.1f} GB RAM")
                self.agent.save_weights(self.weight_path)
                if plot_save_path is not None:
                    self.save_plot(
                        save_path=plot_save_path,
                        include=plot_include
                    )
                if log_writer is not None:
                    log_writer.add_scalar("main/success_rate", success_rate, epoch)
                    log_writer.add_scalar("main/actor_loss", epoch_actor_loss, epoch)
                    log_writer.add_scalar("main/critic_loss", epoch_critic_loss, epoch)
                    log_writer.add_scalar("reward/running_reward", running_reward[-1], epoch)
                    log_writer.add_scalar("reward/episode_reward", episode_reward, epoch)
                    log_writer.add_scalar("memory/memory_length", len(self.agent.memory), epoch)
                    log_writer.add_scalar("memory/gb_ram_usage", to_gb(ram.used), epoch)
                    log_writer.add_scalar("time_elapsed", time_elapsed, epoch)
