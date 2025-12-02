from math import inf
import torch
from torch.nn import Module
from utils.logits_processor import LogitsProcessor, GreedyProcessor
import utils.printing as printing
from typing import List

# Baseline decoding algorithms for comparison

def _length_penalty_fn(length, alpha, min_length):
    """
    Length penalty function for beam search decoding.
    """
    return ((min_length + length) / (min_length + 1)) ** alpha

@torch.no_grad()
def autoregressive_generate(inputs: List[int], model: Module, max_gen_len: int = 40, logits_processor: LogitsProcessor = GreedyProcessor(), eos_tokens_id: int | List[int] = 1, pad_token_id: int = 0, use_cache: bool = False) -> List[int]:
    """
    Autoregressive decoding algorithm for baseline comparison.

    Returns the generated sequence (List[int]).
    """
    # initialize cache
    cache = None
    # get length of the input sequence
    prompt_len = len(inputs)
    # get max sequence length
    # if max_position_embeddings is not defined, use max_context_length from the config of the slow target model, otherwise use 1024
    if hasattr(model.config, 'max_position_embeddings'):
        max_seq_length = model.config.max_position_embeddings
    elif hasattr(model.config, 'max_context_length'):
        max_seq_length = model.config.max_context_length
    else:
        max_seq_length = 1024
    # get total length of the sequence
    total_len = min(max_seq_length, prompt_len + max_gen_len)
    # initialize input ids
    input_ids = torch.full((1, total_len), pad_token_id, dtype=torch.long, device=model.device) # [1, total_len]
    input_ids[0, :prompt_len] = torch.tensor(inputs, dtype=torch.long, device=model.device) # [1, prompt_len]

    # initialize list of end tokens
    list_tokens_id = (
        eos_tokens_id if isinstance(eos_tokens_id, list) else [eos_tokens_id]
    )
    # convert end tokens to tensor
    stop_tokens = torch.tensor(list_tokens_id, dtype=torch.long, device=model.device)

    # autoregressive generation loop
    for curr in range(prompt_len, total_len):
        # get model output
        output = model(input_ids[..., :curr], past_key_values=cache, use_cache=use_cache)
        # get logits
        logits = output.logits[..., -1, :]  # [1, vocab_size]
        # process and sample logits using the specified logits processor
        probs = logits_processor(logits)  # [1, vocab_size]
        x = logits_processor.sample(probs)  # [1, 1]
        # update input ids
        input_ids[0, curr] = x
        # update cache
        cache = output.past_key_values

        # check for end token
        if torch.isin(x, stop_tokens):
            printing.end_token_found(curr) # debugging
            break
    
    # return generated sequence
    return input_ids[0, prompt_len : curr + 1].tolist()


@torch.no_grad()
def beam_search_generate(inputs: List[int], model: Module, max_gen_len: int = 40, num_beams: int = 4, top_k: int = 3, min_length: float = 5.0, alpha: float = 1.2, eos_tokens_id: int | List[int] = 1, pad_token_id: int = 0, tokenizer=None) -> List[int]:
    """
    Beam search decoding algorithm for baseline comparison.

    Returns the generated sequence (List[int]).
    """
    # get length of the input sequence
    prompt_len = len(inputs)
    # get max sequence length
    # if max_position_embeddings is not defined, use max_context_length from the config of the slow target model, otherwise use 1024
    if hasattr(model.config, 'max_position_embeddings'):
        max_seq_length = model.config.max_position_embeddings
    elif hasattr(model.config, 'max_context_length'):
        max_seq_length = model.config.max_context_length
    else:
        max_seq_length = 1024
    
    # make sure the prompt length does not exceed the maximum sequence length
    assert prompt_len < max_seq_length, "Prompt length exceeds maximum sequence length."

    # get total length of the sequence
    total_len = min(max_seq_length, prompt_len + max_gen_len)
    # initialize input ids
    input_ids = torch.full((num_beams, total_len), pad_token_id, dtype=torch.long, device=model.device)
    input_ids[:, :prompt_len] = torch.tensor(inputs, dtype=torch.long, device=model.device)
    # initialize probs, initialized to the minimum possible value
    probs = torch.full((num_beams, total_len), torch.finfo(torch.float).min, dtype=torch.float, device=model.device)
    # initialize beams_probs, initialized to the minimum possible value
    beams_probs = torch.full((num_beams,), torch.finfo(torch.float).min, dtype=torch.float, device=model.device)
    # initialize last_indexes to -1
    last_indexes = torch.full((num_beams,), -1, dtype=torch.long, device=model.device)

    # convert end tokens to tensor  
    stop_tokens = torch.tensor((eos_tokens_id if isinstance(eos_tokens_id, list) else [eos_tokens_id]), dtype=torch.long, device=model.device)

    # prefill step
    # initialize probs and beams_probs to 1.0 
    probs[:, :prompt_len] = 1.0
    beams_probs[:] = 1.0
    
    # get model output
    output = model(input_ids[:, :prompt_len])
    
    # get log-probabilities
    curr_prob = torch.nn.functional.log_softmax(output.logits[0, -1, :], dim=-1)
    
    # get top k tokens for each beam
    top_probs, top_tokens = torch.topk(curr_prob, num_beams, dim=-1)
    input_ids[:, prompt_len] = top_tokens

    # update probs and beams_probs
    probs[:, prompt_len] = probs[:, prompt_len - 1] + top_probs
    beams_probs[:] = probs[:, prompt_len] / _length_penalty_fn(1, alpha, min_length)
    
    # autoregressive generation loop using top k tokens (beam search)
    for curr in range(prompt_len + 1, total_len):
        output = model(input_ids[:, :curr])
        logits = output.logits[:, -1, :]
        probs_curr = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # get top k tokens for each beam
        top_probs, top_tokens = torch.topk(probs_curr, top_k, dim=-1)

        # for each beam, generate all possible next tokens
        possibilities = []
        for beam in range(num_beams):
            # skip a beam if it has already finished (reached end token)
            if last_indexes[beam] != -1:
                prob_vec = probs[beam].detach().clone()
                input_vec = input_ids[beam].detach().clone()
                possibilities.append((beams_probs[beam], input_vec, prob_vec, last_indexes[beam]))
                continue
            
            # for each possibility, generate all possible next tokens
            for possibility in range(top_k):
                # get new probability
                new_prob = probs[beam, curr - 1] + top_probs[beam, possibility]
                # get length penalty
                lp = _length_penalty_fn(curr - prompt_len, alpha, min_length)
                # update probability vector
                prob_vec = probs[beam].detach().clone()
                prob_vec[curr] = new_prob
                input_vec = input_ids[beam].detach().clone()
                input_vec[curr] = top_tokens[beam, possibility]
                last_token_idx = -1
                # if the token is end token or pad token, update last_token_idx
                if torch.isin(input_vec[curr], stop_tokens) or input_vec[curr] == pad_token_id:
                    last_token_idx = curr
                
                # check if the input vector is already in possibilities
                already_in = False
                for p in possibilities:
                    if torch.equal(p[1], input_vec):
                        already_in = True
                        break
                # if not already in, add to possibilities
                if not already_in:
                    possibilities.append((new_prob / (lp if lp != 0 else 1), input_vec, prob_vec, last_token_idx))

        # sort possibilities by probability 
        possibilities.sort(key=lambda x: x[0], reverse=True)

        printing.beam_search_step(possibilities, curr, tokenizer) # debugging

        # select top k possibilities
        possibilities = possibilities[:num_beams]

        # for each beam, update beams_probs, input_ids, probs, and last_indexes
        for beam in range(num_beams):
            beams_probs[beam] = possibilities[beam][0]
            input_ids[beam] = possibilities[beam][1]
            probs[beam] = possibilities[beam][2]
            last_indexes[beam] = possibilities[beam][3]

        # if all beams have finished, break out of the decoding loop
        if torch.all(last_indexes != -1):
            printing.end_token_found(curr) # debugging
            break
    
    # set last indexes to total length if they are still -1
    last_indexes[last_indexes == -1] = total_len - 1

    # return the generated sequence
    return input_ids[0, prompt_len : last_indexes[0] + 1].tolist()