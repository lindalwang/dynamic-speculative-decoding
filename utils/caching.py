"""
KV-Cache pruning utilities for speculative decoding.

When draft tokens are rejected during speculative decoding, the KV-cache
must be pruned to remove the rejected token positions.

"""

from typing import Tuple, Union
from torch import Tensor
from transformers.cache_utils import DynamicCache


def prune_cache(
    cache: Union[Tuple[Tuple[Tensor, Tensor]], DynamicCache, None], 
    num_tokens_to_discard: int
) -> Union[Tuple[Tuple[Tensor, Tensor]], DynamicCache, None]:
    """
    Prune the KV-cache by removing tokens from the end.
    
    This is called when draft tokens are rejected during speculative decoding,
    and we need to remove those positions from the cache before continuing.

    Args:
        cache: The KV cache to be pruned (tuple format or DynamicCache).
        num_tokens_to_discard: Number of tokens to remove from the end.

    Returns:
        The pruned KV cache, or None if input was None.
        
    Raises:
        ValueError: If cache type is not supported.
    """
    if cache is None:
        return None
    
    if num_tokens_to_discard <= 0:
        return cache  
    
    if isinstance(cache, DynamicCache):
        return prune_dynamic_cache(cache, num_tokens_to_discard)
    elif isinstance(cache, tuple):
        return prune_tuple_cache(cache, num_tokens_to_discard)
    else:
        raise ValueError(f"Unsupported cache type: {type(cache)}. "
                        f"Expected DynamicCache or tuple.")


def prune_dynamic_cache(cache: DynamicCache, num_tokens_to_discard: int) -> DynamicCache:
    """
    Prune a DynamicCache by removing tokens from the end.
    
    Works with Llama 3.2 and other models using HuggingFace's DynamicCache.
    Handles Grouped Query Attention (GQA) correctly since DynamicCache
    abstracts the KV head count.

    Args:
        cache: The DynamicCache to prune.
        num_tokens_to_discard: Number of tokens to remove from the end.

    Returns:
        The pruned DynamicCache (modified in place).
    """
    if cache is None:
        return None
    
    seq_length = cache.get_seq_length()
    
    if num_tokens_to_discard >= seq_length:
        raise ValueError(
            f"Cannot discard {num_tokens_to_discard} tokens from cache "
            f"with only {seq_length} tokens."
        )
    
    new_length = seq_length - num_tokens_to_discard
    
    if hasattr(cache, 'crop'):
        cache.crop(new_length)
        return cache
    
    # Fallback: Manual pruning for older HuggingFace versions
    # Cache tensors shape: (batch, num_kv_heads, seq_len, head_dim)
    for layer_idx in range(len(cache.key_cache)):
        cache.key_cache[layer_idx] = cache.key_cache[layer_idx][
            :, :, :new_length, :
        ].contiguous()
        cache.value_cache[layer_idx] = cache.value_cache[layer_idx][
            :, :, :new_length, :
        ].contiguous()
    
    if hasattr(cache, '_seen_tokens'):
        cache._seen_tokens = new_length
    
    return cache


def prune_tuple_cache(
    cache: Tuple[Tuple[Tensor, ...], ...], 
    num_tokens_to_discard: int
) -> Tuple[Tuple[Tensor, ...], ...]:
    """
    Prune a tuple-format KV-cache by removing tokens from the end.
    
    Works with older models that return past_key_values as nested tuples.
    Expected format: Tuple of (key, value) tuples per layer, where each
    tensor has shape (batch, num_heads, seq_len, head_dim).

    Args:
        cache: Tuple of tuples containing key/value tensors per layer.
        num_tokens_to_discard: Number of tokens to remove from the end.

    Returns:
        New tuple cache with pruned tensors.
    """
    if cache is None:
        return None
    
    if len(cache) == 0:
        return cache
    
   
    first_tensor = cache[0][0] if cache[0] is not None else None
    if first_tensor is not None:
        seq_length = first_tensor.shape[2]  # (batch, heads, seq_len, head_dim)
        if num_tokens_to_discard >= seq_length:
            raise ValueError(
                f"Cannot discard {num_tokens_to_discard} tokens from cache "
                f"with only {seq_length} tokens."
            )

    new_cache = []
    for layer_cache in cache:
        if layer_cache is None:
            new_cache.append(None)
            continue

        new_layer = []
        for tensor in layer_cache:
            if tensor is None:
                new_layer.append(None)
            else:
                new_length = tensor.shape[2] - num_tokens_to_discard
                new_tensor = tensor[:, :, :new_length, :].contiguous()
                new_layer.append(new_tensor)
        
        new_cache.append(tuple(new_layer))

    return tuple(new_cache)
