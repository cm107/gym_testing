from __future__ import annotations
import os
import gym
import cv2
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm
import time
from typing import List, Dict, cast

from streamer.recorder.stream_writer import StreamWriter
from common_utils.base.basic import BasicLoadableObject

from ..util import get_device, make_dir_if_not_exists
from ..env_util import make_env, make_env_list
from ..multiprocessing_env import SubprocVecEnv
from ..network.actor_critic import ActorCritic, VisualActorCritic

class ExperienceSegment:
    def __init__(self):
        self.log_probs: List[torch.Tensor] = []
        self.values: List[torch.Tensor]    = []
        self.rewards: List[torch.Tensor]   = []
        self.masks: List[torch.Tensor]     = []
        self.entropy: float = 0
        self.next_state: torch.Tensor = None

        # PPO Related Buffers
        self.states = []
        self.actions = []

class MetaData(BasicLoadableObject['MetaData']):
    def __init__(self):
        super().__init__()
        self.frame_idx_list = []
        self.sampled_frame_idx_list = []
        self.test_rewards = []
        self.train_losses = []
        self.avg_train_losses = []
        self.train_actor_losses = []
        self.avg_train_actor_losses = []
        self.train_critic_losses = []
        self.avg_train_critic_losses = []
        self.entropies = []
        self.avg_entropies = []
        self.advantages = []
        self.lr_buffer = []
        self.clip_param_buffer = []

        self.start_time = time.time()
        self.timestamp = self.start_time
        self.time_elapsed_per_frame = []
        self.avg_time_elapsed_per_frame = []
        self.time_elapsed = []
        self.sampled_time_elapsed = []

        self.working_clip_param = None
        

class TrainConfig(BasicLoadableObject['TrainConfig']):
    def __init__(
        self,
        lr: float=3e-4, # Hyperparameters
        horizon: int=5,
        gamma: float=0.99,
        tau: float=0.95,
        num_envs: int=16,
        ppo_epochs: int=4,
        mini_batch_size: int=5,
        vf_coeff: float=0.5,
        entropy_coeff: float=0.001,
        clip_param: float=0.2,
        max_frames: int=200000, # Train Duration Related
        save_step_size: int=10,
        reward_threshold: float=None,
        max_train_time: float=None,
        use_gae: bool=True, # Flags
        use_ppo: bool=True
    ):
        super().__init__()

        # Hyperparameters
        self.lr = lr
        self.horizon = horizon
        self.gamma = gamma
        self.tau = tau
        self.num_envs = num_envs
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.vf_coeff = vf_coeff
        self.entropy_coeff = entropy_coeff
        self.clip_param = clip_param

        # Train Duration Related
        self.max_frames = max_frames
        self.save_step_size = save_step_size
        self.reward_threshold = reward_threshold
        self.max_train_time = max_train_time

        # Flags
        self.use_gae = use_gae
        self.use_ppo = use_ppo

    @classmethod
    def get_atari_config(cls) -> TrainConfig:
        """https://arxiv.org/pdf/1707.06347.pdf
        Refer to Table 5.
        """
        # TODO: Implement scaling of adam stepsize and clipping parameter that decreases from 1 to 0 over the course of the training session
        cfg = TrainConfig()
        cfg.horizon = 128
        cfg.lr = 2.5e-4
        cfg.ppo_epochs = 3
        cfg.mini_batch_size = 32 # 1/4 of horizon
        cfg.gamma = 0.99
        cfg.tau = 0.95
        cfg.num_envs = 8
        cfg.clip_param = 0.1
        cfg.vf_coeff = 1
        cfg.entropy_coeff = 0.01
        cfg.max_frames = 40000000
        return cfg

class ActorCriticWorker:
    def __init__(self, output_dir: str='output', env_name: str="CartPole-v0", run_id: str="0"):
        self.device = get_device()
        self.env_name = env_name
        # self.envs = make_env_list(env_name=self.env_name, num_envs=16) # Environment batch
        self.envs = cast(SubprocVecEnv, None) # Environment batch used for training
        self.env = make_env(env_name=self.env_name) # Test environment

        print(f"self.env.observation_space: {self.env.observation_space}")
        print(f"self.env.action_space: {self.env.action_space}")

        if not isinstance(self.env.action_space, gym.spaces.discrete.Discrete):
            num_outputs = self.env.action_space.shape[0]
            self.is_discrete_action = False
        else:
            num_outputs = self.env.action_space.n
            self.is_discrete_action = True
        if not isinstance(self.env.observation_space, gym.spaces.box.Box):
            num_inputs = self.env.observation_space.shape[0]
            self.is_image_observation = False
            self.model = ActorCritic(
                num_inputs=num_inputs,
                num_outputs=num_outputs,
                hidden_size=256,
                std=0.0,
                is_discrete_action=self.is_discrete_action
            ).to(self.device)
            self.is_image_observation = False
        else:
            # Make sure that observation is an image.
            assert len(self.env.observation_space.shape) == 3 # (h, w, c)
            assert self.env.observation_space.shape[2] == 3
            self.model = VisualActorCritic(
                obs_spec=self.env.observation_space,
                num_outputs=num_outputs,
                hidden_size=256,
                std=0.0,
                is_discrete_action=self.is_discrete_action
            ).to(self.device)
            self.is_image_observation = True
        print(f'Model Class: {type(self.model).__name__}')

        # Prepare output dir
        make_dir_if_not_exists(output_dir)
        env_dir = f"{output_dir}/{env_name}"
        make_dir_if_not_exists(env_dir)
        self.run_id = run_id
        self.run_dir = f"{env_dir}/{run_id}"
        make_dir_if_not_exists(self.run_dir)
        self.model_path = f'{self.run_dir}/model.pth'
        self.best_model_path = f'{self.run_dir}/best.pth'
        self.train_cfg_path = f'{self.run_dir}/train_config.yaml'
    
    def _plot(self, xdata: list, ydata: list, title: str='data', save_filename: str='plot.png'):
        plt.figure(figsize=(5,5))
        plt.subplot(111)
        plt.title(title)
        plt.plot(xdata, ydata)
        plt.savefig(f"{self.run_dir}/{save_filename}")
        plt.clf()
        plt.close('all')

    def _plot_rewards(self, rewards):
        plt.figure(figsize=(5,5))
        plt.subplot(111)
        plt.title('rewards')
        plt.plot(rewards)
        plt.savefig(f"{self.run_dir}/rewards.png")
        plt.clf()
        plt.close('all')
        
    def _test_env(self, vis=False, num_frames: int=None) -> float:
        state = self.env.reset()
        if vis:
            self.env.render()
        done = False
        total_reward = 0
        frame_idx = 0
        while True:
            if not self.is_image_observation:
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            else:
                state = torch.FloatTensor(state.transpose((2, 0, 1))).unsqueeze(0).to(self.device)
            dist, _ = self.model(state)
            action = dist.sample().cpu().numpy()[0]
            next_state, reward, done, _ = self.env.step(action)
            state = next_state
            if vis:
                self.env.render()
            total_reward += reward
            frame_idx += 1
            if num_frames is not None:
                if frame_idx < num_frames:
                    pass
                else:
                    break
            else:
                if done:
                    break
        return total_reward

    def _compute_returns(self, cfg: TrainConfig, segment: ExperienceSegment, next_value):
        if not cfg.use_gae:
            # Empirical Method
            R = next_value # this is the last reward
            returns = []
            for step in reversed(range(len(rewards))):
                R = segment.rewards[step] + cfg.gamma * R * segment.masks[step] # R doesn't change when mask is 0 (done flag)
                returns.insert(0, R) # Appends R to front of list
            return returns
        else:
            # GAE Method
            values = segment.values + [next_value]
            gae = 0
            returns = []
            for step in reversed(range(len(segment.rewards))):
                delta = segment.rewards[step] + cfg.gamma * values[step + 1] * segment.masks[step] - values[step]
                gae = delta + cfg.gamma * cfg.tau * segment.masks[step] * gae
                returns.insert(0, gae + values[step])
            return returns

    def _accumulate_experience_segment(self, cfg: TrainConfig, state) -> ExperienceSegment:
        segment = ExperienceSegment()
        if isinstance(self.env.observation_space, gym.spaces.box.Box):
            state = state / self.env.observation_space.high

        for _ in range(cfg.horizon):
            if not self.is_image_observation:
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            else:
                state = torch.FloatTensor(state.transpose((0, 3, 1, 2))).to(self.device)
            dist, value = self.model(state)

            # import random
            # if random.random() < 0.10:
            #     action = dist.sample()
            #     action = torch.LongTensor([self.env.action_space.sample() for i in range(state.size()[0])]).to(self.device)
            # else:
            #     action = dist.sample() # sample the action for each environment
            action = dist.sample() # sample the action for each environment
            if self.is_discrete_action:
                segment.next_state, reward, done, _ = self.envs.step(action.cpu().numpy())
            else:
                segment.next_state, reward, done, _ = self.envs.step(action.cpu().numpy()) # np.ndarray of the next_state, reward, done for each environment
            if isinstance(self.env.observation_space, gym.spaces.box.Box):
                segment.next_state = segment.next_state / self.env.observation_space.high

            log_prob = dist.log_prob(action) # shape (num_envs,). This is the log probability of the sampled action for each env.
            segment.entropy += dist.entropy().mean() # dist.entropy() is of shape (num_envs,). This is adding the average entropy across all envs.
            
            segment.log_probs.append(log_prob)
            segment.values.append(value)
            segment.rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(self.device))
            segment.masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(self.device)) # mask: 1->not done, 0->done
            
            if cfg.use_ppo: # PPO only
                segment.states.append(state)
                segment.actions.append(action)

            state = segment.next_state
        
        return segment

    def _a2c_update(self, cfg: TrainConfig, optimizer, log_probs, advantages, metadata: MetaData=None):
        # lower prob -> more negative log_prob -> higher actor_loss
        # higher positive advantage -> higher positive actor_loss
        # higher negative advantage -> higher negative actor_loss
        actor_loss  = -(log_probs * advantages.detach()).mean()

        # higher positive advantage, higher negative advantage -> higher critic loss
        critic_loss = advantages.pow(2).mean()

        # Although it is unlikely if the training is going well, the actor loss can go negative
        # The critic loss is always positive
        loss = actor_loss + cfg.vf_coeff * critic_loss - cfg.entropy_coeff * segment.entropy
        
        if metadata is not None:
            metadata.train_losses.append(loss.detach().cpu().numpy())
            metadata.train_actor_losses.append(actor_loss.detach().cpu().numpy())
            metadata.train_critic_losses.append(critic_loss.detach().cpu().numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def _ppo_iter(self, mini_batch_size, states, actions, log_probs, returns, advantage):
        batch_size = states.size(0)
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.randint(0, batch_size, mini_batch_size)
            states0 = states[rand_ids, :]
            actions0 = actions[rand_ids, :] if not self.is_discrete_action else actions[rand_ids]
            log_probs0 = log_probs[rand_ids, :] if not self.is_discrete_action else log_probs[rand_ids]
            returns0 = returns[rand_ids, :]
            advantage0 = advantage[rand_ids, :]
            yield states0, actions0, log_probs0, returns0, advantage0

    def _ppo_update(self, cfg: TrainConfig, optimizer, states, actions, log_probs, returns, advantages, metadata: MetaData=None):
        actor_loss_list = []
        critic_loss_list = []
        loss_list = []
        clip_param = metadata.working_clip_param if metadata.working_clip_param is not None else cfg.clip_param
        metadata.clip_param_buffer.append(clip_param)
        for _ in range(cfg.ppo_epochs):
            for state, action, old_log_probs, return_, advantage in self._ppo_iter(cfg.mini_batch_size, states, actions, log_probs, returns, advantages):
                dist, value = self.model(state)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

                actor_loss  = - torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()
                loss = cfg.vf_coeff * critic_loss + actor_loss - cfg.entropy_coeff * entropy

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                actor_loss_list.append(actor_loss.detach().cpu().numpy())
                critic_loss_list.append(critic_loss.detach().cpu().numpy())
                loss_list.append(loss.detach().cpu().numpy())
        
        if metadata is not None:
            metadata.train_actor_losses.append(np.array(actor_loss_list).mean())
            metadata.train_critic_losses.append(np.array(critic_loss_list).mean())
            metadata.train_losses.append(np.array(loss_list).mean())

    def _update_plots(self, frame_idx: int, metadata: MetaData):
        metadata.avg_train_losses.append(np.average(metadata.train_losses))
        metadata.train_losses = []
        metadata.avg_train_actor_losses.append(np.average(metadata.train_actor_losses))
        metadata.train_actor_losses = []
        metadata.avg_train_critic_losses.append(np.average(metadata.train_critic_losses))
        metadata.train_critic_losses = []
        metadata.avg_entropies.append(np.average(metadata.entropies))
        metadata.entropies = []
        metadata.avg_time_elapsed_per_frame.append(np.average(metadata.time_elapsed_per_frame))
        metadata.time_elapsed_per_frame = []

        metadata.sampled_frame_idx_list.append(frame_idx)
        metadata.sampled_time_elapsed.append(metadata.time_elapsed[-1]/60)
        self._plot(
            xdata=metadata.sampled_frame_idx_list, ydata=metadata.test_rewards,
            title='rewards', save_filename='rewards.png'
        )
        self._plot(
            xdata=metadata.sampled_time_elapsed, ydata=metadata.test_rewards,
            title='rewards_vs_time', save_filename='rewards_vs_time.png'
        )
        self._plot(
            xdata=metadata.sampled_frame_idx_list, ydata=metadata.avg_train_losses,
            title='loss', save_filename='loss.png'
        )
        self._plot(
            xdata=metadata.sampled_frame_idx_list, ydata=metadata.avg_train_actor_losses,
            title='actor_loss', save_filename='actor_loss.png'
        )
        self._plot(
            xdata=metadata.sampled_frame_idx_list, ydata=metadata.avg_train_critic_losses,
            title='critic_loss', save_filename='critic_loss.png'
        )
        self._plot(
            xdata=metadata.sampled_frame_idx_list, ydata=metadata.avg_entropies,
            title='entropy', save_filename='entropy.png'
        )
        self._plot(
            xdata=metadata.sampled_frame_idx_list, ydata=metadata.avg_time_elapsed_per_frame,
            title='Average Time Elapsed Per Frame', save_filename='avg_time_elapsed_per_frame.png'
        )
        self._plot(
            xdata=metadata.frame_idx_list, ydata=metadata.advantages,
            title='advantage', save_filename='advantage.png'
        )
        self._plot(
            xdata=metadata.frame_idx_list, ydata=metadata.lr_buffer,
            title='Learning Rate', save_filename='lr.png'
        )
        if len(metadata.clip_param_buffer) > 0:
            self._plot(
                xdata=metadata.frame_idx_list, ydata=metadata.clip_param_buffer,
                title='Clipping Parameter', save_filename='clip_param.png'
            )

    def train(self, cfg: TrainConfig=None):
        if cfg is None:
            if os.path.isfile(self.train_cfg_path):
                cfg = TrainConfig.load_from_path(self.train_cfg_path)
            else:
                cfg = TrainConfig()
        elif isinstance(cfg, str):
            cfg = TrainConfig.load_from_path(cfg)
        elif isinstance(cfg, TrainConfig):
            pass
        cfg.save_to_path(self.train_cfg_path, overwrite=True)

        self.envs = make_env_list(env_name=self.env_name, num_envs=cfg.num_envs) # Environment batch
        optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr)
        assert len(optimizer.param_groups) == 1

        def schedule_lr(optimizer, lr):
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        def get_lr(optimizer) -> float:
            return optimizer.param_groups[0]['lr']

        state = self.envs.reset() # num_envs (16) x num_observations (4)
        
        metadata = MetaData()
        pbar = tqdm(total=cfg.max_frames, unit="frame(s)", leave=True)
        frame_idx = 0
        step_count = 0
        while frame_idx < cfg.max_frames:
            anneal_coeff = 1 - (frame_idx / cfg.max_frames)
            # Use fixed-length segments of experience (e.g. 20 timesteps)
            # to compute estimators of the returns and advantage function
            segment = self._accumulate_experience_segment(cfg=cfg, state=state)
            frame_idx += cfg.horizon

            next_state = torch.FloatTensor(segment.next_state).to(self.device)
            if not self.is_image_observation:
                next_state = torch.FloatTensor(segment.next_state).to(self.device)
            else:
                next_state = torch.FloatTensor(segment.next_state.transpose((0, 3, 1, 2))).to(self.device)
            _, next_value = self.model(next_state)

            # actual action-value calculated from segment sample
            returns = self._compute_returns(cfg, segment, next_value)

            # Detaching log_probs and values here seems to cause problems in non-PPO mode?
            if not cfg.use_ppo:
                log_probs = torch.cat(segment.log_probs) # shape: (num_envs*segment_length,); always negative; ranges from log(0+) to log(1)
                returns   = torch.cat(returns).detach()
                values    = torch.cat(segment.values) # estimate of action value (base)
            else:
                log_probs = torch.cat(segment.log_probs).detach() # shape: (num_envs*segment_length,); always negative; ranges from log(0+) to log(1)
                returns   = torch.cat(returns).detach()
                values    = torch.cat(segment.values).detach() # estimate of action value (base)

            # Calculation of Advantage (This is a measure of how much actual action-value better/worse than expected)
            # Advantage can be either positive or negative
            advantage = returns - values # shape: (num_envs*segment_length,1)
            metadata.advantages.append(advantage.detach().mean().cpu().numpy())
            metadata.entropies.append(segment.entropy.detach().cpu().numpy())
            
            timestamp = time.time()
            metadata.time_elapsed_per_frame.append(timestamp - metadata.timestamp)
            metadata.timestamp = timestamp

            if cfg.use_ppo:
                states = torch.cat(segment.states)
                actions = torch.cat(segment.actions)
                self._ppo_update(
                    cfg=cfg,
                    optimizer=optimizer,
                    states=states,
                    actions=actions,
                    log_probs=log_probs,
                    returns=returns,
                    advantages=advantage,
                    metadata=metadata
                )
            else:
                self._a2c_update(
                    cfg=cfg,
                    optimizer=optimizer,
                    log_probs=log_probs,
                    advantages=advantages,
                    metadata=metadata
                )
            
            pbar.set_description(f"loss: {metadata.train_losses[-1]:.2f}")
            pbar.update(cfg.horizon)

            metadata.frame_idx_list.append(frame_idx)
            current_lr = get_lr(optimizer)
            metadata.lr_buffer.append(current_lr)
            metadata.time_elapsed.append(time.time() - metadata.start_time)
            schedule_lr(optimizer, anneal_coeff * cfg.lr)
            metadata.working_clip_param = anneal_coeff * cfg.clip_param

            if step_count % cfg.save_step_size == 0:
                test_reward = np.mean([self._test_env(vis=False) for _ in range(10)])
                if len(metadata.test_rewards) == 0 or test_reward >= max(metadata.test_rewards):
                    self.model.save(self.best_model_path)
                metadata.test_rewards.append(test_reward)

                self._update_plots(frame_idx=frame_idx, metadata=metadata)
                self.model.save(self.model_path)

                if cfg.reward_threshold is not None and test_reward > cfg.reward_threshold:
                    break
                if cfg.max_train_time is not None and metadata.sampled_time_elapsed[-1] >= cfg.max_train_time:
                    # Note: This threshold is in minutes
                    break
            step_count += 1
        metadata.save_to_path(f'{self.run_dir}/metadata.json', overwrite=True)
        pbar.close()
    
    def infer(self, num_frames: int=100, delay: float=1/20, video_save: str=None, show_reward: bool=True, show_details: bool=True, use_best: bool=False):
        from common_utils.cv_drawing_utils import draw_text_rows_in_corner
        self.model.load(self.model_path if not use_best else self.best_model_path)
        self.model.eval()
        if video_save is not None:
            save_path = f'{self.run_dir}/{video_save}'
            stream_writer = StreamWriter(video_save_path=save_path, fps=1/delay)
        else:
            stream_writer = None

        env = make_env(env_name=self.env_name)
        state = env.reset()
        if stream_writer is not None:
            img = env.render(mode='rgb_array')
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            stream_writer.step(img)
        else:
            env.render()

        cumulative_reward = 0
        for i in range(num_frames):
            if not self.is_image_observation:
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            else:
                state = torch.FloatTensor(state.transpose((2, 0, 1))).unsqueeze(0).to(self.device)
            dist, _ = self.model(state)
            next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
            cumulative_reward += reward
            state = next_state
            if stream_writer is not None:
                img = env.render(mode='rgb_array')
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                if show_reward:
                    img = draw_text_rows_in_corner(
                        img=img,
                        row_text_list=[
                            f"Reward: {reward}",
                            f"Cumulative Reward: {cumulative_reward}"
                        ],
                        row_height=img.shape[0]*0.04,
                        thickness=1,
                        color=[255, 0, 0],
                        corner='topleft'
                    )
                if show_details:
                    img = draw_text_rows_in_corner(
                        img=img,
                        row_text_list=[
                            f"Env: {self.env_name}",
                            f"Run ID: {self.run_id}"
                        ],
                        row_height=img.shape[0]*0.04,
                        thickness=1,
                        color=[255, 0, 0],
                        corner='bottomleft'
                    )

                stream_writer.step(img)
            else:
                env.render()
            time.sleep(delay)
        env.close()
        if stream_writer is not None:
            stream_writer.close()