# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import torch
import bittensor as bt
from typing import List
from abc import abstractmethod

class BaseRewardModel:
    SQRT_TWO = torch.tensor([2.0])

    @property
    @abstractmethod
    def name(self) -> str: ...
    def __str__(self) -> str: return str(self.name)
    def __repr__(self) -> str: return str(self.name)

    @abstractmethod
    def get_rewards( self, prompt: str, completion: List[str], name: str ) -> torch.FloatTensor: ...

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.var = 0.0
        self.count_limit = 3000
        self.rewards_stats = {
        name: {'count': self.count, 'mean': self.mean, 'var': self.var} for name in ['rlhf_reward_model', 'relevance_filter']
        }

    def normalize_rewards(self, rewards: torch.FloatTensor, reward_model_name: str) -> torch.FloatTensor:
        # logging.debug(f"Initial rewards for {reward_model_name}: {rewards}")
        # Skip normalization for the relevance filter
        if reward_model_name == "relevance_filter":
            return rewards
        stats = self.rewards_stats[reward_model_name]
        new_count = rewards.numel()

        # Update stats only if there are new rewards.
        if 0 < new_count and 0 < stats['count'] + new_count:
            new_mean = rewards.mean()
            new_var = rewards.var(dim=0)
            new_weight = new_count / (stats['count'] + new_count)
            old_weight = stats['count'] / (stats['count'] + new_count)

            # Save the difference in means before updating the old mean.
            diff = new_mean - stats['mean']

            # Update the old mean with the new mean and weights.
            stats['mean'] = new_weight * new_mean + old_weight * stats['mean']
            # Update the old variance with the new variance and weights, and adjusting for the difference in means.
            stats['var'] = (new_weight * new_var) + (old_weight * stats['var']) + (new_weight * old_weight) * diff * diff
            # Update the old count with the new count, but don't exceed the limit.
            stats['count'] = min(self.count_limit, stats['count'] + new_count)

        # Standardize the rewards using the updated mean and variance.
        rewards = rewards - stats['mean']
        if stats['var'] > 0:
            rewards /= torch.sqrt(stats['var'])
        # Scale the standardized rewards to the range [0, 1] using the error function as a CDF.
        normalized_rewards = 0.5 * (1 + torch.erf(rewards / SQRT_TWO.to(rewards.device)))
        
        # logging.debug(f"Normalized rewards for {reward_model_name}: {normalized_rewards}")
        return normalized_rewards

    def apply(self, prompt: str, responses: List[bt.DendriteCall], name: str, reward_model_name: str) -> torch.FloatTensor:
        """ Applies the reward model across each call. Unsuccessful responses are zeroed.
        """
        # Get indices of correctly responding calls.
        successful_completions_indices = list(range(len(responses)))

        # Get all completions from responding calls.
        successful_completions: List[str] = [responses[idx].strip() for idx in successful_completions_indices]

        # Reward each completion.
        successful_rewards = self.get_rewards(prompt, successful_completions, name)

        # Softmax rewards across samples.
        successful_rewards_normalized = self.normalize_rewards(successful_rewards, reward_model_name)

        # Init zero rewards for all calls.
        filled_rewards = torch.zeros(len(responses), dtype=torch.float32)
        filled_rewards_normalized = torch.zeros(len(responses), dtype=torch.float32)

        # Fill reward tensor.
        for idx, reward, reward_normalized in zip(successful_completions_indices, successful_rewards, successful_rewards_normalized):
            filled_rewards[idx] = reward
            filled_rewards_normalized[idx] = reward_normalized

        # Return the filled rewards.
        return filled_rewards, filled_rewards_normalized

class MockRewardModel( BaseRewardModel ):

    @property
    def name(self) -> str: return self.mock_name

    def __init__(self, mock_name: str = 'MockReward'):
        super().__init__()
        self.mock_name = mock_name

    def apply( self, prompt: str, completion: List[str], name: str ) -> torch.FloatTensor: 
        mock_reward = torch.tensor( [0 for _ in completion], dtype=torch.float32 )
        return mock_reward, mock_reward

        