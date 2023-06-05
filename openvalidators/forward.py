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
# DEALINGS IN
#  THE SOFTWARE.

import time
import wandb
import torch
import random
import bittensor as bt

from loguru import logger
from typing import List, Tuple
from openvalidators.misc import ttl_get_block
from openvalidators.prompts import (
    extract_score,
    followup_request_template,
    answer_scoring_template,
    followup_scoring_template,
)
from openvalidators.utils import check_uid_availability


def get_random_uids(self, k: int) -> torch.LongTensor:
    """Returns k available random uids from the metagraph.
    Args:
        k (int): Number of uids to return.
    Returns:
        uids (torch.LongTensor): Randomly sampled available uids.
    Notes:
        If `k` is larger than the number of available `uids`, set `k` to the number of available `uids`.
    """
    candidate_uids = [uid
                      for uid, ax in enumerate(self.metagraph.axons)
                      if check_uid_availability(self.metagraph, uid, self.config.neuron.vpermit_tao_limit)]

    available_uids = torch.tensor(candidate_uids, dtype=torch.int64).to(self.device)
    uids = torch.tensor(random.sample(available_uids.tolist(), k), dtype=torch.int64)
    return uids


def is_successful_completion(
    self, response: bt.DendriteCall, min_len: int = 10
) -> bool:
    """Filters out unsuccessful responses.

    Args:
        response (bitensor.DendriteCall)
            Response from the network.
        min_len (int):
            Minimum length for a completion to be considered successful.

    Returns:
        True if the response is successful, False otherwise.
    """
    len_check = len(response.completion) > min_len

    if self.config.neuron.nsfw_filter:
        filter_check = filter_message(self, response.completion)
    else:
        filter_check = True

    return len_check and filter_check


async def scoring_completions(
    self, prompt: str, scoring_template: str, responses: List[bt.DendriteCall]
) -> Tuple[torch.FloatTensor, List[List[int]], List[List[str]], List[List[int]]]:
    """Using the prompt and call responses, outsource prompt-based scoring to network,
       return scoring average for each response.

    Args:
        prompt (str):
            Prompt to use for the reward model.
        responses (List[ bittensor.DendriteCall ]):
            List of responses from the network.

    Returns:
        filled_scores (torch.FloatTensor, shape = (len(responses)) ):
            Network outsourced prompt-based scoring average for each response.
        all_scoring_uids (List[ List[ int ] ]):
            For each completion, a list of random uids prompted for scoring the completion.
        all_scoring_completions (List[ List[ str ] ]):
            For each completion, scoring response completions for each random uid queried.
        all_scoring_values (List[ List[ int ] ]):
            For each completion, extracted scores (or None) for each random uid queried.

    """

    # Scoring details for event logging.
    n_score = self.config.neuron.scoring_sample_size

    # Initialize results containers with consistent sizes.
    all_scoring_uids = [[None] * n_score for _ in responses]
    all_scoring_completions = [[""] * n_score for _ in responses]
    all_scoring_values = [[None] * n_score for _ in responses]

    # Fill scores with zeros for unsuccessful responses.
    filled_scores = torch.zeros(len(responses), dtype=torch.float32)

    # Each completion separately scored by the network.
    for i, response in enumerate(responses):
        if not is_successful_completion(self, response):
            continue

        # Scoring prompt for this completion.
        scoring_prompt = scoring_template.format(
            question=prompt, answer=response.completion.strip()
        )

        # Random uids for scoring this completion.
        scoring_uids = get_random_uids(self, k=n_score).to(self.device)
        all_scoring_uids[i] = scoring_uids.tolist()

        # Query the network with the scoring prompt to score a given prompt + completion.
        scoring_responses = await self.dendrite_pool.async_forward(
            roles=["user"],
            messages=[scoring_prompt],
            uids=scoring_uids,
            timeout=self.config.neuron.scoring_timeout,
        )

        # Scoring completions for wandb log.
        all_scoring_completions[i] = [resp.completion for resp in scoring_responses]

        # Scoring values for wandb log.
        scoring_values = [
            extract_score(resp.completion)
            if is_successful_completion(self, resp)
            else None
            for resp in scoring_responses
        ]

        all_scoring_values[i] = scoring_values

        # Scoring average for completion.
        successful_scoring_values = [value for value in scoring_values if value]
        if len(successful_scoring_values) > 0:
            filled_scores[i] = sum(successful_scoring_values) / len(
                successful_scoring_values
            )

    # Return the filled scores.
    return filled_scores, all_scoring_uids, all_scoring_completions, all_scoring_values


def reward_completions(
    self, prompt: str, responses: List[bt.DendriteCall]
) -> torch.FloatTensor:
    """Using the prompt and call responses return softmaxed rewards for each response.

    Args:
        prompt (str):
            Prompt to use for the reward model.
        responses (List[ bt.DendriteCall ]):
            List of responses from the network.

    Returns:
        filled_rewards (torch.FloatTensor, shape = (len(responses)) ):
            Softmaxed rewards for each response.
    """
    # Filters out unsuccessful responses.
    successful_completions_indices: List[int] = [
        idx
        for idx, resp in enumerate(responses)
        if is_successful_completion(self, resp)
    ]
    successful_completions: List[str] = [
        responses[idx].completion.strip() for idx in successful_completions_indices
    ]

    # Get completions with and without prompt.
    completions_with_prompt: List[str] = [
        prompt + comp for comp in successful_completions
    ]
    completions_without_prompt: List[str] = [comp for comp in successful_completions]

    # Compute the reward for each completion by taking the difference between
    # the score with prompt and the score without prompt.
    # Shift the rewards by the reward shift.
    successful_rewards: torch.FloatTensor = self.reward_model.reward(
        completions_with_prompt=completions_with_prompt,
        completions_without_prompt=completions_without_prompt,
        difference=True,
        shift=self.config.neuron.reward_shift,
    ).to(self.device)

    # Softmax the rewards
    successful_rewards = torch.nn.functional.softmax(successful_rewards, 0)

    # Fill scores with zeros for non successful responses.
    filled_rewards = torch.zeros(len(responses), dtype=torch.float32)
    for idx, reward in zip(successful_completions_indices, successful_rewards):
        filled_rewards[idx] = reward

    # Return the filled rewards.
    return filled_rewards


def filter_message(self, message, bound_score=0.5) -> bool:
    """Check if the message is related to any sexual content.

    Args:
        message (str):
            The message that we check if we should filter out.
    Returns:
        result (bool):
            True indicates we should filter out the result, false indicates the result is safe.
    """
    tokenized = self.filter_tokenizer(message)
    input_ids = tokenized["input_ids"]
    filter_out = False

    # The model can only support 512 tokens at a time, so we have to break up the check
    # if there are too many tokens in a single message.
    while len(input_ids) > 0:
        _input_ids = input_ids[:512]

        with torch.no_grad():
            output = self.filter_model(torch.tensor([_input_ids]).to(self.device))

        # Filter out if the logit score is out of bound.
        filter_out = (
            output.logits[0, 0] < bound_score or output.logits[0, 1] > bound_score
        )

        # Return if this section of the message needs to be filtered out.
        if filter_out:
            break

        input_ids = input_ids[512:]

    return filter_out


async def forward(self):
    """Performs a forward pass.

    1. Loads a base prompt from the dataset.
    2. Queries the network with the base prompt and gets a follow up question.
    3. Computes the rewards for the followup question.
    4. Gets the best followup question.
    5. Queries the network with the followup and gets answers.
    6. Computes the rewards for the answer responses.
    7. Computes forward pass rewards.
    8. Trains the gating layer.

    """
    bt.logging.info("forward()")
    start_time = time.time()

    # Load a base prompt from the dataset.
    bootstrap_prompt = next(self.dataset)["context"]

    # Query the network with the base prompt and get the question extensions.
    followup_prompt = f"{bootstrap_prompt}\n\n{followup_request_template}\n\n"
    followup_uids = get_random_uids(self, k=self.config.neuron.followup_sample_size).to(
        self.device
    )
    followup_completions = await self.dendrite_pool.async_forward(
        roles=["user"],
        messages=[followup_prompt],
        uids=followup_uids,
        timeout=self.config.neuron.followup_timeout,
    )
    # Reward model evaluation.
    followup_rewards = reward_completions(
        self, followup_prompt, followup_completions
    ).to(self.device)
    best_followup = followup_completions[
        followup_rewards.argmax(dim=0)
    ].completion.strip()

    # Prompt-based scoring via network.
    followup_scoring_result = await scoring_completions(
        self, bootstrap_prompt, followup_scoring_template, followup_completions
    )
    (
        followup_scorings,
        followup_scoring_uids,
        followup_scoring_completions,
        followup_scoring_values,
    ) = followup_scoring_result
    best_followup_scoring = followup_completions[
        followup_scorings.argmax(dim=0)
    ].completion.strip()

    # Query the network with the question and get responses.
    answer_prompt = f"{bootstrap_prompt}\n\n{best_followup}"
    answer_uids = get_random_uids(self, k=self.config.neuron.answer_sample_size).to(
        self.device
    )
    answer_completions = await self.dendrite_pool.async_forward(
        roles=["user"],
        messages=[answer_prompt],
        uids=answer_uids,
        timeout=self.config.neuron.answer_timeout,
    )
    # Reward model evaluation.
    answer_rewards = reward_completions(self, answer_prompt, answer_completions).to(
        self.device
    )
    best_answer = answer_completions[answer_rewards.argmax(dim=0)].completion.strip()

    # Prompt-based scoring via network.
    answer_scoring_result = await scoring_completions(
        self, answer_prompt, answer_scoring_template, answer_completions
    )
    (
        answer_scorings,
        answer_scoring_uids,
        answer_scoring_completions,
        answer_scoring_values,
    ) = answer_scoring_result
    best_answer_scoring = answer_completions[
        answer_scorings.argmax(dim=0)
    ].completion.strip()

    # Compute forward pass rewards.
    scattered_followup_rewards = (
        torch.zeros((self.metagraph.n), dtype=torch.float32)
        .to(self.device)
        .scatter(0, answer_uids, answer_rewards)
    )
    scattered_answer_rewards = (
        torch.zeros((self.metagraph.n), dtype=torch.float32)
        .to(self.device)
        .scatter(0, followup_uids, followup_rewards)
    )
    rewards = scattered_followup_rewards + scattered_answer_rewards
    self.moving_averaged_scores = (
        self.config.neuron.moving_average_alpha * rewards.to(self.device)
        + (1 - self.config.neuron.moving_average_alpha) * self.moving_averaged_scores.to(self.device)
    )

    # Train the gating layer.
    scores = self.gating_model(answer_prompt).to(self.device)
    gating_loss = self.gating_model.backward(
        scores=scores[answer_uids], rewards=answer_rewards
    )

    # Create event.
    # TODO: Compress wandb data better.
    event = {
        "block": ttl_get_block(self),
        "step_length": time.time() - start_time,
        "gating_loss": gating_loss.item(),
        "gating_scorings": scores[answer_uids].tolist(),
        "base_prompt": bootstrap_prompt,
        "followup_uids": followup_uids.tolist(),
        "followup_completions": [comp.completion for comp in followup_completions],
        'followup_times': [ comp.elapsed_time for comp in followup_completions ],
        "followup_rewards": followup_rewards.tolist(),
        "followup_scorings": followup_scorings,
        "followup_scoring_uids": followup_scoring_uids,
        "followup_scoring_completions": followup_scoring_completions,
        "followup_scoring_values": followup_scoring_values,
        "best_followup_scoring": best_followup_scoring,
        "best_followup": best_followup,
        "best_answer": best_answer,
        "answer_prompt": answer_prompt,
        "answer_uids": answer_uids.tolist(),
        "answer_completions": [ans.completion for ans in answer_completions],
        'answer_times': [ ans.elapsed_time for ans in answer_completions ],
        "answer_rewards": answer_rewards.tolist(),
        "answer_scorings": answer_scorings,
        "answer_scoring_uids": answer_scoring_uids,
        "answer_scoring_completions": answer_scoring_completions,
        "answer_scoring_values": answer_scoring_values,
        "best_answer_scoring": best_answer_scoring,
    }

    bt.logging.debug("step:", str(event))
    # Log to wandb.
    if not self.config.wandb.off:
        
        if ttl_get_block( self ) % self.config.wandb.weights_block_length <= self.prev_block % self.config.wandb.weights_block_length:
            event["moving_averaged_scores"] = self.moving_averaged_scores.tolist()
            bt.logging.debug("logging weights")

        self.wandb.log(event)

    # Log locally
    if not self.config.neuron.dont_save_events:
        logger.log("EVENTS", "events", **event)