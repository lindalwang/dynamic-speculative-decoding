"""
Static Speculative Decoding.

Based on: https://arxiv.org/pdf/2211.17192.pdf
"""

import torch
from torch.nn import Module
from utils.sampling_strategies import Sampler, GreedySampler
from utils.caching import prune_cache
import utils.printing as printing
from typing import List, Tuple, Union


@torch.no_grad()
def speculative_generate(inputs: List[int], fast_drafter: Module, slow_target: Module, tokenizer = None, gamma: int = 5, sampler: Sampler = GreedySampler(), max_gen_len: int = 40, eos_tokens_id: Union[int, List[int]] = 1, pad_token_id: int = 0, use_cache: bool = False, skip_sample_adjustment: bool = False, first_target: bool = True) -> Tuple[List[int], float]:
    """
    Static Speculative Decoding with fixed gamma.
    
    Args:
        inputs: Input token IDs.
        drafter: Smaller/faster draft model.
        target: Larger/slower target model.
        tokenizer: Tokenizer for debug output.
        gamma: Number of draft tokens per step (fixed).
        sampler: Sampling strategy.
        max_gen_len: Maximum tokens to generate.
        eos_tokens_id: End-of-sequence token ID(s).
        pad_token_id: Padding token ID.
        use_cache: Whether to use KV-cache.
        skip_sample_adjustment: Skip adjusted sampling on rejection.
        first_target: Run target model first to prefill cache.
        debug: Print debug information.
    
    Returns:
        Tuple of (generated_tokens, acceptance_rate).
    """
    # Initialize caches
    drafter_cache, target_cache = None, None

    # Prepare stop tokens
    list_tokens_id = eos_tokens_id if isinstance(eos_tokens_id, list) else [eos_tokens_id]
    stop_tokens = torch.tensor(list_tokens_id, dtype=torch.long, device=target.device).unsqueeze(1)
    
    # Counters
    drafts_accepted = 0
    drafts_speculated = 0
    
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
    
    # Initial target forward pass
    if first_target:
        Mp = slow_target(input_ids=input_ids[..., :current_position], past_key_values=slow_target_cache, use_cache=use_cache)
        slow_target_cache = Mp.past_key_values
        p_p = sampler(Mp.logits[..., -1, :])
        t = sampler.sample(p_p)
        input_ids[0, current_position] = t
        current_position += 1
        
        if torch.isin(t, stop_tokens):
            if debug:
                printing.end_token_found(0)
            return input_ids[0, prompt_len:current_position].tolist(), 0.0
        
        if debug:
            printing.initial_step(t, tokenizer)
    
    # Main loop
    while current_position < total_len:
        corrected_gamma = min(gamma, total_len - current_position - 1)
        
        if corrected_gamma <= 0:
            break

        # Store draft probabilities
        q = torch.zeros((1, corrected_gamma, vocabulary_size), device=target.device)
        
        # Generate drafts
        input_ids = input_ids.to(drafter.device)
        
        for k in range(corrected_gamma):
            # run fast drafter on the input ids
            Mq = fast_drafter(input_ids=input_ids[..., :current_position+k], past_key_values=fast_drafter_cache, use_cache=use_cache)
            # update the fast drafter cache
            fast_drafter_cache = Mq.past_key_values
            
            # get the logits of the last token
            draft_logits = Mq.logits[...,-1,:] 
            
            # process the logits using the specified logits processor
            draft_probs = sampler(draft_logits)
            
            # store the logits in q tensor
            q[0, k] = draft_probs.to(slow_target.device)
            
            # sample a token from the processed logits using the specified logits processor
            xi = sampler.sample(draft_probs)

            # update input ids
            input_ids[0, current_position + k] = xi
        
        drafts_speculated += corrected_gamma
        
        # run target model on the draft tokens
        # result = logits of the previous tokens + one more token
        Mp = slow_target(input_ids=input_ids[..., :current_position + corrected_gamma], past_key_values=slow_target_cache, use_cache=use_cache)

        # update the slow target cache
        slow_target_cache = Mp.past_key_values
        draft_logits = Mp.logits[..., current_position - 1:current_position + corrected_gamma - 1, :] # [1, corrected_gamma, vocab_size]
        p = sampler(draft_logits) # [1, gamma, vocab_size]
        
        Mp = target(
            input_ids=input_ids[..., :current_position + corrected_gamma],
            past_key_values=target_cache,
            use_cache=use_cache
        )
        target_cache = Mp.past_key_values
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
        
        drafts_accepted += n
        
        # Check for stop token
        stop_locations = torch.nonzero(torch.eq(input_ids[..., current_position:current_position + n], stop_tokens))
        if stop_locations.shape[0] > 0:
            stop_location = stop_locations[0, 1].item()
            if debug:
                printing.end_token_found(stop_location)
            accept_rate = drafts_accepted / drafts_speculated if drafts_speculated > 0 else 0
            return input_ids[0, prompt_len:current_position + stop_location + 1].tolist(), accept_rate

        # Sample next token
        if n == corrected_gamma:
            p_p = Mp.logits[..., current_position + corrected_gamma - 1, :]
            p_p = sampler(p_p)
        # otherwise, we use the n-th token of Mp
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

        # sample a token from the processed logits using the specified logits processor
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
            accept_rate = drafts_accepted / drafts_speculated if drafts_speculated > 0 else 0
            return input_ids[0, prompt_len:current_position].tolist(), accept_rate
    
    accept_rate = drafts_accepted / drafts_speculated if drafts_speculated > 0 else 0
    return input_ids[0, prompt_len:].tolist(), accept_rate
