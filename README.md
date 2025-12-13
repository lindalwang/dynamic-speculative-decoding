# Dynamic Speculative Decoding

A PyTorch implementation of speculative decoding for accelerating LLM inference.

## Setup

```bash
pip install torch transformers termcolor pyyaml
```

## Usage

```bash
# Interactive mode
python inference.py

# Single prompt
python inference.py --prompt "What is machine learning?"

# Custom config
python inference.py --config config/base.yaml
```

## Configuration

Edit `config/base.yaml` to change:
- Target/drafter models
- Generation parameters (gamma, max_length, use_cache)
- Sampling strategy (greedy, multinomial, top-k, nucleus, top-k + nucleus)


## References

- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) (Leviathan et al., 2023)
- [Accelerating LLM Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) (Chen et al., 2023)

