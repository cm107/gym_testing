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
            k_future=k_future,
            is_train=True
        )
        self.t_success_rate = None
        self.total_ac_loss = None
        self.total_cr_loss = None

    def reset_metadata(self):
        self.t_success_rate = []
        self.total_ac_loss = []
        self.total_cr_loss = []

    def _eval_agent(self, env_, agent_):
        total_success_rate = []
        running_r = []
        for ep in range(10):
            per_success_rate = []
            env_dictionary = env_.reset()
            s = env_dictionary["observation"]
            ag = env_dictionary["achieved_goal"]
            g = env_dictionary["desired_goal"]
            while np.linalg.norm(ag - g) <= 0.05:
                env_dictionary = env_.reset()
                s = env_dictionary["observation"]
                ag = env_dictionary["achieved_goal"]
                g = env_dictionary["desired_goal"]
            ep_r = 0
            for t in range(50):
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

    def train(
        self, max_epochs: int=50, max_cycles: int=50,
        max_episodes: int=2, num_updates: int=40,
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
            for cycle in range(0, max_cycles):
                mb = []
                cycle_actor_loss = 0
                cycle_critic_loss = 0
                for episode in range(max_episodes):
                    episode_dict = {
                        "state": [],
                        "action": [],
                        "info": [],
                        "achieved_goal": [],
                        "desired_goal": [],
                        "next_state": [],
                        "next_achieved_goal": []}
                    env_dict = self.env.reset()
                    state = env_dict["observation"]
                    achieved_goal = env_dict["achieved_goal"]
                    desired_goal = env_dict["desired_goal"]
                    while np.linalg.norm(achieved_goal - desired_goal) <= 0.05:
                        env_dict = self.env.reset()
                        state = env_dict["observation"]
                        achieved_goal = env_dict["achieved_goal"]
                        desired_goal = env_dict["desired_goal"]
                    for t in range(50):
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
                    episode_dict["next_state"] = episode_dict["state"][1:]
                    episode_dict["next_achieved_goal"] = episode_dict["achieved_goal"][1:]
                    mb.append(dc(episode_dict))

                self.agent.store(mb)
                for n_update in range(num_updates):
                    actor_loss, critic_loss = self.agent.train()
                    cycle_actor_loss += actor_loss
                    cycle_critic_loss += critic_loss

                epoch_actor_loss += cycle_actor_loss / num_updates
                epoch_critic_loss += cycle_critic_loss /num_updates
                self.agent.update_networks()

            ram = psutil.virtual_memory()
            success_rate, running_reward, episode_reward = self._eval_agent(self.env, self.agent)
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
