from flask import Flask, request, jsonify
import csv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)

MODEL_NAME = "cerebras/btlm-3b-8k-base"
device = None
tokenizer = None
model = None


def load_model_and_tokenizer():
    """Load model and tokenizer into memory."""
    global device, tokenizer, model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device)


def compute_reward(logits, labels, loss_mask):
    """Compute reward from logits and labels."""
    logits = logits.log_softmax(-1)
    per_token_logps = torch.gather(logits, dim=2, index=labels).squeeze(2)
    reward = (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    return reward[0].cpu().detach().item()


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


def generate_and_score_completions(prompt, num_completions=5, max_length=50):
    """Generate completions and score them."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    completions = model.generate(input_ids, max_length=max_length, num_return_sequences=num_completions,
                                 pad_token_id=tokenizer.eos_token_id, do_sample=True)

    results = [
        (tokenizer.decode(completion, skip_special_tokens=True), reward_single(prompt, tokenizer.decode(completion, skip_special_tokens=True)),
         model(completion).logits.mean(dim=1).tolist())
        for completion in completions
    ]

    return sorted(results, key=lambda x: x[1], reverse=True)[:num_completions]


@app.route('/predict', methods=['POST'])
def predict():
    content = request.json
    prompt = content.get('prompt', '')
    
    if not prompt:
        return jsonify(error="Prompt not provided"), 400

    completions_data = generate_and_score_completions(prompt)

    # Extract just the completions text for response
    completions = [c[0] for c in completions_data]
    
    return jsonify(completions=completions)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
