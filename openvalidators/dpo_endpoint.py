from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from openvalidators.reward import DirectPreferenceRewardModel
import logging
import argparse
from reward import BaseRewardModel
import threading

logging.basicConfig(level=logging.INFO)
MODEL_NAME = "cerebras/btlm-3b-8k-base"
semaphore = threading.Semaphore(8)  # Limit to 8 concurrent requests
app = Flask(__name__)

class reward_endpoint:
    def __init__(self, device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True).to(self.device)
        
        self.reward_weights = torch.tensor([1.0], dtype=torch.float32).to(self.device)
        self.reward_functions = [DirectPreferenceRewardModel(device=self.device)]
        self.normalizer = BaseRewardModel()


def compute_reward(logits, labels, loss_mask):
    """Compute reward from logits and labels."""
    logits = logits.log_softmax(-1)
    per_token_logps = torch.gather(logits, dim=2, index=labels).squeeze(2)
    reward = (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)

    # Check for NaNs or Infs on GPU tensor
    if torch.any(torch.isnan(reward)) or torch.any(torch.isinf(reward)):
        reward[torch.isnan(reward) | torch.isinf(reward)] = -11.  # exp(-11)=1.67e-5 < 2e-5=1/50257 (typical vocab size)

    return reward


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=8008, type=int, help="Authentication token")
    parser.add_argument("--gpu", default=0, type=int, help="GPU device to use")
    args = parser.parse_args()
    return args
    
@app.route("/", methods=["POST"])
def chat():
    with semaphore:
        content = request.json
        prompt = content.get('prompt', '')
        completions = content.get('responses', [])

        if not prompt or not completions:
            return jsonify(error="Both prompt and responses are required"), 400

        rewards = []
        token_probabilities = []

        # Tokenize the prompt once
        prompt_ids = rw.tokenizer.encode(prompt, return_tensors="pt")[0].to(device)

        for completion in completions:
            # Tokenize only the completion
            completion_ids = rw.tokenizer.encode(completion, add_special_tokens=False, return_tensors="pt")[0].to(device)
            combined = torch.cat([prompt_ids, completion_ids])

            if rw.tokenizer.model_max_length <= len(prompt_ids) or rw.tokenizer.model_max_length < len(combined):
                rewards.append(-11.)
                token_probabilities.append({})
                continue

            labels = combined.clone()
            labels[:len(prompt_ids)] = -100
            labels = labels[1:]
            loss_mask = (labels != -100)
            labels[labels == -100] = 0
            labels = labels.unsqueeze(0).unsqueeze(2)
            logits = rw.model(combined.unsqueeze(0)).logits[:, :-1, :]

            # Convert logits to probabilities
            probs = logits.softmax(dim=2)

            # Extract individual token probabilities
            token_probs = {}
            for i, token_id in enumerate(labels.squeeze().tolist()):
                token = rw.tokenizer.decode([token_id])  # <-- Modification here
                token_prob = probs[0, i, token_id].item()
                token_probs[token] = token_prob

        reward = compute_reward(logits, labels, loss_mask)
        normalized_reward = rw.normalizer.normalize_rewards(reward, MODEL_NAME) 

        rewards.append(normalized_reward)
        token_probabilities.append(token_probs)

    return jsonify(rewards=rewards, token_probabilities=token_probabilities)


if __name__ == "__main__":
    args = parse_arguments()
    device = torch.device(f"cuda:{args.gpu}")
    rw = reward_endpoint(device)
    app.run(host="0.0.0.0", port=args.port, threaded=False, debug=False)
