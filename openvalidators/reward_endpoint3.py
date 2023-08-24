import torch
import bittensor as bt
from typing import List
from openvalidators.dendrite import AsyncDendritePool
from openvalidators.config import add_args, check_config, config
from openvalidators.run import run
from openvalidators.misc import ttl_get_block
from openvalidators.reward import (
    OpenAssistantRewardModel,
    ReciprocateRewardModel,
    RelevanceRewardModel,
)
import logging
logging.basicConfig(level=logging.INFO)
SQRT_TWO = torch.tensor([2.0])

class RewardNormalizer:
    def __init__(self, reward_model_names: List[str]):
        self.rewards_stats = {
            name: {'count': 0, 'mean': 0.0, 'var': 1.0} for name in ['rlhf_reward_model', 'reciprocate_reward_model', 'relevance_filter']
        }
        self.count_limit = 1e6  # This can be adjusted based on your requirements.

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

class reward_endpoint:

    @classmethod
    def check_config(cls, config: "bt.Config"):
        check_config(cls, config)

    @classmethod
    def add_args(cls, parser):
        add_args(cls, parser)

    @classmethod
    def config(cls):
        return config(cls)

    def __init__(self, gpu_id):
        self.config = reward_endpoint.config()
        self.device = torch.device(f"cuda:{gpu_id}")
        self.reward_weights = torch.tensor([
            self.config.reward.rlhf_weight, 
            self.config.reward.reciprocate_weight, 
        ], dtype=torch.float32).to(self.device)
        self._normalizer = None

        # Ensure reward function weights sum to 1.
        assert torch.isclose(self.reward_weights.sum(), torch.tensor(1.0)), "Reward function weights do not sum to 1"

        self.reward_functions = [
            OpenAssistantRewardModel(device=self.device),
            ReciprocateRewardModel(device=self.device)
        ]

        assert len(self.reward_functions) == len(self.reward_weights), "Length of reward function weights and reward functions do not match"

        self.masking_functions = [RelevanceRewardModel(device=self.device)]

    @property
    def normalizer(self):
        if self._normalizer is None:
            self._normalizer = RewardNormalizer(['rlhf_reward_model', 'reciprocate_reward_model', 'relevance_filter'])
        return self._normalizer

    def calculate_total_reward(self, prompt, responses, name="augment"):
        all_model_scores = {}
        rewards: torch.FloatTensor = torch.zeros(len(responses), dtype=torch.float32).to(self.device)

        for weight, reward_fn in zip(self.reward_weights, self.reward_functions):
            reward_values = reward_fn.apply(prompt, responses, name).to(self.device)
            reward_values_normalized = self.normalizer.normalize_rewards(reward_values, reward_fn.name)
            # logging.debug(f"Rewards BEFORE normalization for {reward_fn.name}: {reward_values}")
            # logging.debug(f"Rewards AFTER normalization for {reward_fn.name}: {reward_values_normalized}")

            rewards += weight * reward_values_normalized
            all_model_scores[reward_fn.name] = reward_values_normalized.tolist()

        for masking_fn in self.masking_functions:
            mask_values = masking_fn.apply(prompt, responses, name).to(self.device)
            # logging.debug(f"Raw mask values from {masking_fn.name}: {mask_values}")
            mask_values_binary = (mask_values > 0.5).float()
            rewards *= mask_values_binary
            all_model_scores[masking_fn.name] = mask_values_binary.tolist()

        logging.info(f"Final reward values: {rewards}")
        return rewards, all_model_scores


from flask import Flask, request, jsonify
import traceback
import sys
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=8008, type=int, help="Authentication token")
    parser.add_argument("--gpu", default=0, type=int, help="GPU device to use")
    args = parser.parse_args()
    return args

auth_tokens= ["SjhSXuEmZoW#%SD@#nAsd123bash#$%&@n"]

app = Flask(__name__)

@app.route("/", methods=["POST"])
def chat():
    request_data = request.get_json()
    auth_token = request_data.get("verify_token")
    if auth_token not in auth_tokens:
        return jsonify({"error": "Invalid authentication token"}), 401

    prompt = request_data.get("prompt", None)
    responses = request_data.get("responses", None)
    if not (prompt and responses):
        return "No prompt or responses"

    try:
        rewards, all_model_scores = rw.calculate_total_reward(prompt, responses)
        return jsonify({"rewards": rewards.tolist(), "reward_details": all_model_scores})
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        return str(e)

if __name__ == "__main__":
    args = parse_arguments()
    rw = reward_endpoint(args.gpu)
    app.run(host="0.0.0.0", port=args.port, threaded=False, debug=False)


