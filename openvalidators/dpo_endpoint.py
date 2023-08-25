from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import bittensor as bt
from typing import List
from openvalidators.dendrite import AsyncDendritePool
from openvalidators.config import add_args, check_config, config
from openvalidators.run import run
from openvalidators.misc import ttl_get_block
from openvalidators.reward import (
    # OpenAssistantRewardModel,
    # ReciprocateRewardModel,
    # RelevanceRewardModel,
    DirectPreferenceRewardModel
)
import logging
logging.basicConfig(level=logging.INFO)
SQRT_TWO = torch.tensor([2.0])

class RewardNormalizer:
    def __init__(self, reward_model_names: List[str]):
        self.rewards_stats = {
            name: {'count': 0, 'mean': 0.0, 'var': 1.0} for name in ['rlhf_reward_model', 'relevance_filter']
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
app = Flask(__name__)

MODEL_NAME = "cerebras/btlm-3b-8k-base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device)

def reward_single(prompt, completion):
    """Compute reward for a single completion."""
    with torch.no_grad():
        combined = tokenizer(prompt + completion, return_tensors="pt").input_ids[0].to(device)
        prompt_part = tokenizer(prompt, return_tensors="pt").input_ids[0].to(device)

        if tokenizer.model_max_length <= len(prompt_part) or tokenizer.model_max_length < len(combined):
            return -11.

        labels = combined.clone()
        labels[:len(prompt_part)] = -100
        labels = labels[1:]
        loss_mask = (labels != -100)
        labels[labels == -100] = 0
        labels = labels.unsqueeze(0).unsqueeze(2)
        logits = model(combined.unsqueeze(0)).logits[:, :-1, :]

        reward = compute_reward(logits, labels, loss_mask)

        if torch.isnan(reward) or torch.isinf(reward):
            return -11.
        return reward

@app.route('/score', methods=['POST'])
def score():
    content = request.json
    prompt = content.get('prompt', '')
    completion = content.get('completion', '')
    
    if not prompt or not completion:
        return jsonify(error="Both prompt and completion are required"), 400

    reward = reward_single(prompt, completion)
    return jsonify(reward=reward)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)