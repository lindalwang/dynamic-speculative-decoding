"""
Dynamic Speculative Decoding with adaptive gamma scheduling.

Gamma (number of draft tokens) adjusts based on acceptance rate:
- All drafts accepted → increase gamma (drafter is doing well)
- Some rejected → decrease gamma (be more conservative)

Based on HuggingFace's AssistedCandidateGenerator heuristic.
"""

import torch
from torch.nn import Module
from utils.sampling_strategies import Sampler, GreedySampler
from utils.caching import prune_cache
import utils.printing as printing
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass, field


@dataclass
class DynamicGammaScheduler:
    """
    Dynamically adjusts gamma based on acceptance rate.
    
    Args:
        initial_gamma: Starting value for gamma.
        min_gamma: Minimum allowed gamma.
        max_gamma: Maximum allowed gamma.
        schedule: "heuristic" for dynamic, "constant" for static.
    """
    initial_gamma: int = 4
    min_gamma: int = 1
    max_gamma: int = 10
    schedule: str = "heuristic"
    
    # Internal state (initialized in __post_init__)
    gamma: int = field(init=False)
    _history: List[Tuple[int, int]] = field(init=False)
    
    def __post_init__(self):
        self.gamma = self.initial_gamma
        self._history = []
    
    def update(self, num_accepted: int, num_speculated: int) -> None:
        """
        Update gamma based on acceptance.
        
        Args:
            num_accepted: Number of drafts accepted (n).
            num_speculated: Number of drafts generated (gamma used).
        """
        self._history.append((num_accepted, num_speculated))
        
        if self.schedule == "constant":
            return
        
        # Heuristic schedule
        if num_accepted == num_speculated:
            # All accepted → increase gamma
            self.gamma = min(self.gamma + 2, self.max_gamma)
        else:
            # Some rejected → decrease gamma
            self.gamma = max(self.min_gamma, num_accepted + 1)
    
    def get_gamma(self) -> int:
        """Get current gamma value."""
        return self.gamma
    
    def reset(self) -> None:
        """Reset to initial state."""
        self.gamma = self.initial_gamma
        self._history = []
    
    def get_stats(self) -> dict:
        """Get statistics about gamma adjustments."""
        if not self._history:
            return {
                "avg_acceptance_rate": 0.0,
                "total_accepted": 0,
                "total_speculated": 0,
                "num_steps": 0,
                "final_gamma": self.gamma,
            }
        
        total_accepted = sum(h[0] for h in self._history)
        total_speculated = sum(h[1] for h in self._history)
        
        return {
            "avg_acceptance_rate": total_accepted / total_speculated if total_speculated > 0 else 0.0,
            "total_accepted": total_accepted,
            "total_speculated": total_speculated,
            "num_steps": len(self._history),
            "final_gamma": self.gamma,
        }


@torch.no_grad()
def dynamic_speculative_generate(
    inputs: List[int],
    drafter: Module,
    target: Module,
    scheduler: DynamicGammaScheduler,
    tokenizer=None,
    sampler: Sampler = GreedySampler(),
    max_gen_len: int = 40,
    eos_tokens_id: Union[int, List[int]] = 1,
    pad_token_id: int = 0,
    use_cache: bool = False,
    skip_sample_adjustment: bool = False,
    first_target: bool = True,
    debug: bool = False,
) -> Tuple[List[int], float, dict]:
    """
    Speculative Decoding with dynamic gamma adjustment.
    
    Args:
        inputs: Input token IDs.
        drafter: Smaller/faster draft model.
        target: Larger/slower target model.
        scheduler: DynamicGammaScheduler for adaptive gamma.
        tokenizer: Tokenizer for debug output.
        sampler: Sampling strategy.
        max_gen_len: Maximum tokens to generate.
        eos_tokens_id: End-of-sequence token ID(s).
        pad_token_id: Padding token ID.
        use_cache: Whether to use KV-cache.
        skip_sample_adjustment: Skip adjusted sampling on rejection.
        first_target: Run target model first to prefill cache.
        debug: Print debug information.
    
    Returns:
        Tuple of (generated_tokens, acceptance_rate, scheduler_stats).
    """
    scheduler.reset()
    
    # Initialize caches
    drafter_cache, target_cache = None, None

    # Prepare stop tokens
    list_tokens_id = eos_tokens_id if isinstance(eos_tokens_id, list) else [eos_tokens_id]
    stop_tokens = torch.tensor(list_tokens_id, dtype=torch.long, device=target.device).unsqueeze(1)
    
    # Get vocabulary size and sequence limits
    vocabulary_size = target.config.vocab_size
    prompt_len = len(inputs)
    
    if hasattr(target.config, 'max_position_embeddings'):
        max_seq_length = target.config.max_position_embeddings
    elif hasattr(target.config, 'max_context_length'):
        max_seq_length = target.config.max_context_length
    else:
        max_seq_length = 1024
    
    total_len = min(max_seq_length, prompt_len + max_gen_len)

    # Prepare input tensor
    input_ids = torch.full((1, total_len), pad_token_id, dtype=torch.long, device=target.device)
    input_ids[0, :prompt_len] = torch.tensor(inputs, dtype=torch.long, device=target.device)
    
    current_position = prompt_len
    
    # Prefill cache
    if first_target:
        Mp = target(
            input_ids=input_ids[..., :current_position],
            past_key_values=target_cache,
            use_cache=use_cache
        )
        target_cache = Mp.past_key_values
        p_p = sampler(Mp.logits[..., -1, :])
        t = sampler.sample(p_p)
        input_ids[0, current_position] = t
        current_position += 1
        
        if torch.isin(t, stop_tokens):
            if debug:
                printing.end_token_found(0)
            return input_ids[0, prompt_len:current_position].tolist(), 0.0, scheduler.get_stats()
        
        if debug:
            printing.initial_step(t, tokenizer)
    
    # Main loop
    while current_position < total_len:
        # Get current gamma from scheduler
        gamma = scheduler.get_gamma()
        corrected_gamma = min(gamma, total_len - current_position - 1)
        
        if corrected_gamma <= 0:
            break

        q = torch.zeros((1, corrected_gamma, vocabulary_size), device=target.device)
        input_ids = input_ids.to(drafter.device)
        
        for k in range(corrected_gamma):
            Mq = drafter(
                input_ids=input_ids[..., :current_position + k],
                past_key_values=drafter_cache,
                use_cache=use_cache
            )
            drafter_cache = Mq.past_key_values
            draft_logits = Mq.logits[..., -1, :]
            draft_probs = sampler(draft_logits)
            q[0, k] = draft_probs.to(target.device)
            xi = sampler.sample(draft_probs)
            input_ids[0, current_position + k] = xi
        
        # Verify with target
        input_ids = input_ids.to(target.device)
        
        Mp = target(
            input_ids=input_ids[..., :current_position + corrected_gamma],
            past_key_values=target_cache,
            use_cache=use_cache
        )
        target_cache = Mp.past_key_values
        
        # Extract logits for draft token positions
        # With cache: logits only has new positions, so slice from start
        # Without cache: logits has all positions, slice at current_position - 1
        if use_cache:
            draft_logits = Mp.logits[..., :corrected_gamma, :]
        else:
            draft_logits = Mp.logits[..., current_position - 1:current_position + corrected_gamma - 1, :]
        p = sampler(draft_logits)
        
        # Rejection sampling
        r = torch.rand(corrected_gamma, device=target.device)
        fractions = p / q
        n = corrected_gamma
        for i in range(corrected_gamma):
            if r[i] > fractions[0, i, input_ids[0, current_position + i]]:
                n = i
                break
        
        scheduler.update(num_accepted=n, num_speculated=corrected_gamma)
        
        if debug:
            print(f"  γ={corrected_gamma}, accepted={n}, new_γ={scheduler.get_gamma()}")
        
        # Check for stop token in accepted drafts
        stop_locations = torch.nonzero(torch.eq(input_ids[..., current_position:current_position + n], stop_tokens))
        if stop_locations.shape[0] > 0:
            stop_location = stop_locations[0, 1].item()
            if debug:
                printing.end_token_found(stop_location)
            stats = scheduler.get_stats()
            return input_ids[0, prompt_len:current_position + stop_location + 1].tolist(), stats["avg_acceptance_rate"], stats

        # Sample next token
        if n == corrected_gamma:
            # All drafts accepted, get bonus token from position after last draft
            if use_cache:
                p_p = Mp.logits[..., corrected_gamma, :]
            else:
                p_p = Mp.logits[..., current_position + corrected_gamma - 1, :]
            p_p = sampler(p_p)
        else:
            if use_cache:
                drafter_cache = prune_cache(drafter_cache, corrected_gamma - n)
                target_cache = prune_cache(target_cache, corrected_gamma - n + 1)
            
            if not skip_sample_adjustment:
                diff = p[..., n, :] - q[0, n, :]
                diff_max = torch.where(diff > 0, diff, torch.zeros_like(diff))
                diff_max_sum = torch.sum(diff_max, dim=-1, keepdim=True)
                p_p = diff_max / diff_max_sum
            else:
                p_p = p[..., n, :]

        x = sampler.sample(p_p)
        
        if debug:
            generated = input_ids.clone().detach()
        
        input_ids[0, current_position + n:current_position + corrected_gamma] = pad_token_id
        input_ids[0, current_position + n] = x
        
        if debug:
            printing.speculative_step(tokenizer, generated, input_ids, n, prompt_len, current_position, corrected_gamma)
        
        current_position += n + 1
        
        if torch.isin(x, stop_tokens):
            if debug:
                printing.end_token_found(n)
            stats = scheduler.get_stats()
            return input_ids[0, prompt_len:current_position].tolist(), stats["avg_acceptance_rate"], stats
    
    stats = scheduler.get_stats()
    return input_ids[0, prompt_len:].tolist(), stats["avg_acceptance_rate"], stats

