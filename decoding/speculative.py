import torch
from torch.nn import Module
from utils.sampling_strategies import Sampler, GreedySampler
from transformers.cache_utils import DynamicCache
from utils.caching import prune_cache
import utils.printing as printing
from typing import List, Tuple

@torch.no_grad()
def speculative_generate(inputs: List[int], fast_drafter: Module, slow_target: Module, tokenizer = None, gamma: int = 5, logits_processor: Sampler = GreedySampler(), max_gen_len: int = 40, eos_tokens_id: int | List[int] = 1, pad_token_id: int = 0, use_cache: bool = False, skip_sample_adjustment: bool = False, first_target: bool = True) -> Tuple[List[int], float]:
    """
    Implementation of Speculative Decoding based on https://arxiv.org/pdf/2211.17192.pdf.

    Returns the generated sequence (List[int]) and the acceptance ratio (float).
    """
    
    # initialize caches for drafter and target models
    fast_drafter_cache, slow_target_cache = None, None

    # prepare stop tokens
    list_tokens_id = eos_tokens_id if isinstance(eos_tokens_id, list) else [eos_tokens_id] # list of stop tokens
    stop_tokens = torch.tensor(list_tokens_id, dtype=torch.long, device=slow_target.device).unsqueeze(1) # [1, num_stop_tokens]
    
    # counters for accepted and speculated draft tokens
    drafts_accepted = 0
    drafts_speculated = 0
    
    # get vocabulary size
    vocabulary_size = slow_target.config.vocab_size    
        
    # get length of the input sequence
    prompt_len = len(inputs)

    # get max sequence length
    # if max_position_embeddings is not defined, use max_context_length from the config of the slow target model, otherwise use 1024
    if hasattr(slow_target.config, 'max_position_embeddings'):
        max_seq_length = slow_target.config.max_position_embeddings
    elif hasattr(slow_target.config, 'max_context_length'):
        max_seq_length = slow_target.config.max_context_length
    else:
        max_seq_length = 1024
    
    # get total length
    total_len = min(max_seq_length, prompt_len + max_gen_len)

    # prepare input ids
    input_ids = torch.full((1, total_len), pad_token_id, dtype=torch.long, device=slow_target.device) # [1, total_len]
    input_ids[0, :prompt_len] = torch.tensor(inputs, dtype=torch.long, device=slow_target.device) # [1, prompt_len]
    
    # initialize current position
    current_position = prompt_len
    
    # if first_target is True, run the target model before the speculative algorithm. This allows us to prefetch the kvcache and get a first token.
    if first_target:
        Mp = slow_target(input_ids=input_ids[..., :current_position], past_key_values=slow_target_cache, use_cache=use_cache)
        slow_target_cache = Mp.past_key_values
        p_p = logits_processor(Mp.logits[..., -1, :])
        t = logits_processor.sample(p_p)
        input_ids[0, current_position] = t
        current_position += 1
        
        if torch.isin(t, stop_tokens):
            printing.end_token_found(0) # debugging
            return input_ids[0, prompt_len:current_position].tolist(), 0
        
        printing.initial_step(t, tokenizer) # debugging
    
    # speculative decoding loop
    while current_position < total_len:
        # correct gamma to ensure we don't generate more drafts than possible
        corrected_gamma = min(gamma, total_len-current_position-1)

        # initialize q tensor to store the logits of the drafts (shape: [1, gamma, vocab_size])
        q = torch.zeros((1, corrected_gamma, vocabulary_size), device=slow_target.device)
        
        # move input ids to fast drafter device
        input_ids = input_ids.to(fast_drafter.device)
        
        # generate gamma (corrected_gamma) drafts
        for k in range(corrected_gamma):
            # run fast drafter on the input ids
            Mq = fast_drafter(input_ids=input_ids[..., :current_position+k], past_key_values=fast_drafter_cache, use_cache=use_cache)
            # update the fast drafter cache
            fast_drafter_cache = Mq.past_key_values
            
            # get the logits of the last token
            draft_logits = Mq.logits[...,-1,:] 
            
            # process the logits using the specified logits processor
            draft_probs = logits_processor(draft_logits)
            
            # store the logits in q tensor
            q[0, k] = draft_probs.to(slow_target.device)
            
            # sample a token from the processed logits using the specified logits processor
            xi = logits_processor.sample(draft_probs)

            # update input ids
            input_ids[0, current_position + k] = xi
            
        # keep track of the number of speculated tokens (for acceptance ratio)
        drafts_speculated += corrected_gamma

        # move input ids to slow target device
        input_ids = input_ids.to(slow_target.device)
        
        # run target model on the draft tokens
        # result = logits of the previous tokens + one more token
        Mp = slow_target(input_ids=input_ids[..., :current_position + corrected_gamma], past_key_values=slow_target_cache, use_cache=use_cache)

        # update the slow target cache
        slow_target_cache = Mp.past_key_values
        draft_logits = Mp.logits[..., current_position - 1:current_position + corrected_gamma - 1, :] # [1, corrected_gamma, vocab_size]
        p = logits_processor(draft_logits) # [1, gamma, vocab_size]
        
        # perform rejection sampling to compute the last accepted draft position
        # initialize 'corrected_gamma' random numbers
        r = torch.rand(corrected_gamma, device=slow_target.device) 
        # compute the fractions for speculative sampling (section 2.3 in the paper)
        fractions = p / q
        n = corrected_gamma
        for i in range(corrected_gamma):
            if r[i] > fractions[0, i, input_ids[0, current_position + i]]:
                n = i
                break

        # update the number of accepted drafts for recording acceptance ratio
        drafts_accepted += n
        
        # check if the end token is in the drafts
        # if so, return the accepted tokens
        stop_locations = torch.nonzero(torch.eq(input_ids[..., current_position:current_position + n], stop_tokens))
        if stop_locations.shape[0] > 0:
            stop_location = stop_locations[0, 1].item()
            printing.end_token_found(stop_location) # debugging
            return input_ids[0, prompt_len:current_position + stop_location + 1].tolist(), drafts_accepted / drafts_speculated

        # adjust the distribution from Mp
        # if n == gamma, we use the last token of Mp
        if n == corrected_gamma:
            p_p = Mp.logits[..., current_position + corrected_gamma - 1, :]
            p_p = logits_processor(p_p)
        # otherwise, we use the n-th token of Mp
        else:
            # prune the cache
            if use_cache:
                fast_drafter_cache = prune_cache(fast_drafter_cache, corrected_gamma - n)
                slow_target_cache = prune_cache(slow_target_cache, corrected_gamma - n + 1)
            # adjust the distribution from Mp 
            # if skip_sample_adjustment is True, we use the n-th token of Mp
            if not skip_sample_adjustment:
                diff = p[..., n, :] - q[0, n, :] # get the difference between p and q
                diff_max = torch.where(diff > 0, diff, torch.zeros_like(diff)) # get the maximum of diff
                diff_max_sum = torch.sum(diff_max, dim=-1, keepdim=True) # get the sum of diff_max
                p_p = diff_max / diff_max_sum # adjust the distribution
            # otherwise, we use the n-th token of p
            else:
                p_p = p[..., n, :]

        # sample a token from the processed logits using the specified logits processor
        x = logits_processor.sample(p_p)
        
        # for debugging
        generated = input_ids.clone().detach()
        
        # update input ids
        input_ids[0, current_position + n:current_position + corrected_gamma] = pad_token_id
        input_ids[0, current_position + n] = x
        
        printing.speculative_step(tokenizer, generated, input_ids, n, prompt_len, current_position, corrected_gamma) # debugging
        
        # update current position
        current_position += n + 1
        
        # check if the end token is in the drafts
        # if so, return the accepted tokens
        if torch.isin(x, stop_tokens):
            printing.end_token_found(n) # debugging
            return input_ids[0, prompt_len:current_position].tolist(), drafts_accepted / drafts_speculated
    
    return input_ids[0, prompt_len:].tolist(), drafts_accepted / drafts_speculated