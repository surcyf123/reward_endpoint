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
from typing import List
from .config import RewardModelType
from .reward import BaseRewardModel
from transformers import AutoTokenizer, AutoModelForCausalLM


class DirectPreferenceRewardModel(BaseRewardModel):

    reward_model_name: str = "TheBloke/Llama-2-7B-fp16"

    @property
    def name(self) -> str: return RewardModelType.dpo.value

    def __init__(self, device: str):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(DirectPreferenceRewardModel.reward_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(DirectPreferenceRewardModel.reward_model_name,
                                                          torch_dtype=torch.float16).to(self.device)

    def reward_single(self, prompt: str, completion: str, name: str) -> float:
        r""" Calculates a direct preference optimization (DPO) style reward for a completion,
        which is a reference model's average log-probability for completion tokens given a prompt.
        Uses guidance from https://github.com/eric-mitchell/direct-preference-optimization/blob/main/trainers.py.
        """
        with torch.no_grad():
            # Tokenize the combined prompt + completion.
            combined = self.tokenizer(prompt + completion, return_tensors="pt").input_ids[0].to(self.device)  # [seq_len]
            # Tokenize only the prompt, to help determine prompt token length.
            prompt_part = self.tokenizer(prompt, return_tensors="pt").input_ids[0].to(self.device)  # [prompt_len]
            # Ensure that the prompt_part tokens align with the combined tokens.
            assert (prompt_part == combined[:len(prompt_part)]).all()

            labels = combined.clone()  # [seq_len]
            # Label only each next token prediction ground-truth.
            labels = labels[1:]  # [seq_len-1]
            # Ignore prompt part for calculating reward.
            labels[1:len(prompt_part)] = -100
            loss_mask = (labels != -100)  # [seq_len-1]

            # Dummy token to allow for indexing, but loss will be ignored.
            labels[labels == -100] = 0
            # Reshape for gather operation.
            labels = labels.unsqueeze(0).unsqueeze(2)  # [batch_size=1, seq_len-1, :]

            # Forward pass to calculate logit predictions for each sequence position.
            logits = self.model(combined.unsqueeze(0)).logits  # [batch_size=1, seq_len, vocab_len]
            # Predict only where labels are available.
            logits = logits[:, :-1, :]  # [batch_size=1, seq_len-1, vocab_len]
            # Rescale via log(softmax(logits)).
            logits = logits.log_softmax(-1)
            # Calculate the model's log-probability for each actual completion token.
            per_token_logps = torch.gather(logits, dim=2, index=labels).squeeze(2)  # [batch_size=1, seq_len-1]
            # Average log-probability over completion sequence.
            reward = (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)  # [batch_size=1]
            reward = reward[0].cpu().detach()

            return reward
        
    def get_rewards(self, prompt: str, completions: List[str], name: str) -> torch.FloatTensor:
        return torch.tensor([self.reward_single(prompt, completion, name) for completion in completions],
                            dtype=torch.float32).to(self.device)
