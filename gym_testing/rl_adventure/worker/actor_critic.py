import cv2
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm
import time
from typing import List

from streamer.recorder.stream_writer import StreamWriter

from ..util import get_device, make_dir_if_not_exists
from ..env_util import make_env, make_env_list
from ..network.actor_critic import ActorCritic

class ExperienceSegment:
    def __init__(self):
        self.log_probs: List[torch.Tensor] = []
        self.values: List[torch.Tensor]    = []
        self.rewards: List[torch.Tensor]   = []
        self.masks: List[torch.Tensor]     = []
        self.entropy: float = 0
        self.next_state: torch.Tensor = None

class ActorCriticWorker:
    def __init__(self, output_dir: str='output', env_name: str="CartPole-v0", run_id: str="0"):
        self.device = get_device()
        self.env_name = env_name
        self.envs = make_env_list(env_name=self.env_name, num_envs=16) # Environment batch
        self.env = make_env(env_name=self.env_name) # Test environment

        print(f"self.envs.observation_space: {self.envs.observation_space}")
        print(f"self.envs.action_space: {self.envs.action_space}")
        self.model = ActorCritic(
            num_inputs=self.envs.observation_space.shape[0],
            num_outputs=self.envs.action_space.n,
            hidden_size=256,
            std=0.0
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters())

        # Prepare output dir
        make_dir_if_not_exists(output_dir)
        env_dir = f"{output_dir}/{env_name}"
        make_dir_if_not_exists(env_dir)
        self.run_dir = f"{env_dir}/{run_id}"
        make_dir_if_not_exists(self.run_dir)
        self.model_path = f'{self.run_dir}/model.pth'
    
    def _plot(self, data: list, title: str='data', save_filename: str='plot.png'):
        plt.figure(figsize=(5,5))
        plt.subplot(111)
        plt.title(title)
        plt.plot(data)
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
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            dist, _ = self.model(state)
            next_state, reward, done, _ = self.env.step(dist.sample().cpu().numpy()[0])
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

    def _compute_returns(self, next_value, rewards, masks, gamma=0.99):
        """
        This is the straightforward way of calculating returns
        """
        R = next_value # this is the last reward
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + gamma * R * masks[step] # R doesn't change when mask is 0 (done flag)
            returns.insert(0, R) # Appends R to front of list
        return returns

    def _compute_gae(self, next_value, rewards, masks, values, gamma=0.99, tau=0.95):
        """
        This calculates returns with gae algorithm
        """
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + gamma * tau * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns

    def _accumulate_experience_segment(self, segment_length: int, state) -> ExperienceSegment:
        segment = ExperienceSegment()

        for _ in range(segment_length):
            state = torch.FloatTensor(state).to(self.device)
            dist, value = self.model(state)

            action = dist.sample() # sample the action for each environment
            segment.next_state, reward, done, _ = self.envs.step(action.cpu().numpy()) # np.ndarray of the next_state, reward, done for each environment

            log_prob = dist.log_prob(action) # shape (num_envs,). This is the log probability of the sampled action for each env.
            segment.entropy += dist.entropy().mean() # dist.entropy() is of shape (num_envs,). This is adding the average entropy across all envs.
            
            segment.log_probs.append(log_prob)
            segment.values.append(value)
            segment.rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(self.device))
            segment.masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(self.device)) # mask: 1->not done, 0->done
            
            state = segment.next_state
        
        return segment

    def train(self, lr: float=3e-4, num_steps: int=5, max_frames: int=20000, gamma: float=0.99, tau: float=0.95, use_gae: bool=False):
        state = self.envs.reset() # num_envs (16) x num_observations (4)
        
        test_rewards = []
        train_losses = []
        avg_train_losses = []
        train_actor_losses = []
        avg_train_actor_losses = []
        train_critic_losses = []
        avg_train_critic_losses = []
        entropies = []
        avg_entropies = []
        advantages = []
        pbar = tqdm(total=max_frames, unit="frame(s)", leave=True)
        for frame_idx in range(max_frames):
            # Use fixed-length segments of experience (e.g. 20 timesteps)
            # to compute estimators of the returns and advantage function
            segment = self._accumulate_experience_segment(segment_length=num_steps, state=state)
                
            if frame_idx % 1000 == 0:
                test_rewards.append(np.mean([self._test_env(vis=False) for _ in range(10)]))
                self._plot(data=test_rewards, title='rewards', save_filename='rewards.png')
                    
            next_state = torch.FloatTensor(segment.next_state).to(self.device)
            _, next_value = self.model(next_state)

            # actual action-value calculated from segment sample
            if not use_gae:
                returns = self._compute_returns(next_value, segment.rewards, segment.masks, gamma=gamma)
            else:
                returns = self._compute_gae(next_value, segment.rewards, segment.masks, segment.values, gamma=gamma, tau=tau)

            log_probs = torch.cat(segment.log_probs) # shape: (num_envs*segment_length,); always negative; ranges from log(0+) to log(1)
            returns   = torch.cat(returns).detach()
            values    = torch.cat(segment.values) # estimate of action value (base)

            # Calculation of Advantage (This is a measure of how much actual action-value better/worse than expected)
            # Advantage can be either positive or negative
            advantage = returns - values # shape: (num_envs*segment_length,1)
            advantages.append(advantage.detach().mean().cpu().numpy())

            # lower prob -> more negative log_prob -> higher actor_loss
            # higher positive advantage -> higher positive actor_loss
            # higher negative advantage -> higher negative actor_loss
            actor_loss  = -(log_probs * advantage.detach()).mean()

            # higher positive advantage, higher negative advantage -> higher critic loss
            critic_loss = advantage.pow(2).mean()

            # Although it is unlikely if the training is going well, the actor loss can go negative
            # The critic loss is always positive
            loss = actor_loss + 0.5 * critic_loss - 0.001 * segment.entropy
            train_losses.append(loss.detach().cpu().numpy())
            train_actor_losses.append(actor_loss.detach().cpu().numpy())
            train_critic_losses.append(critic_loss.detach().cpu().numpy())
            entropies.append(segment.entropy.detach().cpu().numpy())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            pbar.set_description(f"loss: {loss:.2f}")
            pbar.update()

            if frame_idx % 1000 == 0:
                avg_train_losses.append(np.average(train_losses))
                train_losses = []
                avg_train_actor_losses.append(np.average(train_actor_losses))
                train_actor_losses = []
                avg_train_critic_losses.append(np.average(train_critic_losses))
                train_critic_losses = []
                avg_entropies.append(np.average(entropies))
                entropies = []
                self._plot(data=avg_train_losses, title='loss', save_filename='loss.png')
                self._plot(data=avg_train_actor_losses, title='actor_loss', save_filename='actor_loss.png')
                self._plot(data=avg_train_critic_losses, title='critic_loss', save_filename='critic_loss.png')
                self._plot(data=avg_entropies, title='entropy', save_filename='entropy.png')
                self._plot(data=advantages, title='advantage', save_filename='advantage.png')
                self.model.save(self.model_path)
        pbar.close()
    
    def infer(self, num_frames: int=100, delay: float=1/20, video_save: str=None):
        self.model.load(self.model_path)
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
        for i in range(num_frames):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            dist, _ = self.model(state)
            next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
            state = next_state
            if stream_writer is not None:
                img = env.render(mode='rgb_array')
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                stream_writer.step(img)
            else:
                env.render()
            time.sleep(delay)
        env.close()
        if stream_writer is not None:
            stream_writer.close()