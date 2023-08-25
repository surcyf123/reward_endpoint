import torch
import bittensor as bt
from typing import List
from openvalidators.dendrite import AsyncDendritePool
from openvalidators.config import add_args, check_config, config
from openvalidators.run import run
from openvalidators.misc import ttl_get_block
from reward import BaseRewardModel, MockRewardModel

from openvalidators.reward import (
    OpenAssistantRewardModel,
    # ReciprocateRewardModel,
    RelevanceRewardModel,
)
import logging
logging.basicConfig(level=logging.INFO)

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
            # self.config.reward.reciprocate_weight, 
        ], dtype=torch.float32).to(self.device)
        self._normalizer = None

        # Ensure reward function weights sum to 1.
        assert torch.isclose(self.reward_weights.sum(), torch.tensor(1.0)), "Reward function weights do not sum to 1"

        self.reward_functions = [
            OpenAssistantRewardModel(device=self.device),
            # ReciprocateRewardModel(device=self.device)
        ]
        print("Number of reward functions:", len(self.reward_functions))
        print("Number of reward weights:", len(self.reward_weights))

        assert len(self.reward_functions) == len(self.reward_weights), f"Length mismatch: {len(self.reward_functions)} reward functions but {len(self.reward_weights)} reward weights."

        self.masking_functions = [RelevanceRewardModel(device=self.device)]

    @property
    def normalizer(self):
        if self._normalizer is None:
            self._normalizer = RewardNormalizer(['rlhf_reward_model', 'relevance_filter'])
        return self._normalizer


    def calculate_total_reward(self, prompt, responses, name="augment"):
        all_model_scores = {}
        rewards: torch.FloatTensor = torch.zeros(len(responses), dtype=torch.float32).to(self.device)

        for weight, reward_fn in zip(self.reward_weights, self.reward_functions):
            # Using the apply method from the reward.py file
            reward_values, reward_values_normalized = reward_fn.apply(prompt, responses, name, reward_fn)

            # Move tensors to the desired device
            reward_values = reward_values.to(self.device)
            reward_values_normalized = reward_values_normalized.to(self.device)
            
            rewards += weight * reward_values_normalized
            all_model_scores[reward_fn.name] = reward_values_normalized.tolist()

        for masking_fn in self.masking_functions:
            mask_values, mask_values_normalized = masking_fn.apply(prompt, responses, name, masking_fn)
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


