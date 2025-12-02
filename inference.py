import argparse
import random
import numpy as np
import torch
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from sampling import autoregressive_generate, speculative_generate
from utils.sampling_strategies import GreedySampler, MultinomialSampler, TopKSampler, NucleusSampler, TopKNucleusSampler
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    QuantoConfig,
)
import time
from termcolor import colored


@dataclass
class ModelConfig:
    """Configuration for a model."""
    name: str
    quantization: Optional[str] = None
    
    def get_quantization_config(self):
        """Convert quantization string to QuantoConfig."""
        if self.quantization is None:
            return None
        return QuantoConfig(weights=self.quantization)


@dataclass
class GenerationConfig:
    """Configuration for generation parameters."""
    gamma: int = 4
    max_length: int = 35
    use_cache: bool = False
    chat_mode: bool = True


@dataclass 
class SamplingConfig:
    """Configuration for sampling strategy."""
    sampler: str = "greedy"
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    
    def build_sampler(self):
        """Build the sampler based on config."""
        samplers = {
            "greedy": lambda: GreedySampler(temperature=self.temperature),
            "multinomial": lambda: MultinomialSampler(temperature=self.temperature),
            "topk": lambda: TopKSampler(temperature=self.temperature, top_k=self.top_k),
            "nucleus": lambda: NucleusSampler(temperature=self.temperature, top_p=self.top_p),
            "topknucleus": lambda: TopKNucleusSampler(
                temperature=self.temperature, top_k=self.top_k, top_p=self.top_p
            ),
        }
        if self.sampler not in samplers:
            raise ValueError(f"Unknown sampler: {self.sampler}. Available: {list(samplers.keys())}")
        return samplers[self.sampler]()


@dataclass
class InferenceModesConfig:
    """Configuration for which inference modes to run."""
    speculative: bool = True
    target_autoregressive: bool = True
    drafter_autoregressive: bool = False


@dataclass
class DebugConfig:
    """Configuration for debug settings."""
    enabled: bool = False
    seed: int = 42


@dataclass
class Config:
    """Main configuration container."""
    target_model: ModelConfig
    drafter_model: ModelConfig
    generation: GenerationConfig
    sampling: SamplingConfig
    inference_modes: InferenceModesConfig
    debug: DebugConfig
    device: str = "cuda"
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from a YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls(
            target_model=ModelConfig(
                name=data['models']['target']['name'],
                quantization=data['models']['target'].get('quantization'),
            ),
            drafter_model=ModelConfig(
                name=data['models']['drafter']['name'],
                quantization=data['models']['drafter'].get('quantization'),
            ),
            generation=GenerationConfig(
                gamma=data['generation'].get('gamma', 4),
                max_length=data['generation'].get('max_length', 35),
                use_cache=data['generation'].get('use_cache', False),
                chat_mode=data['generation'].get('chat_mode', True),
            ),
            sampling=SamplingConfig(
                sampler=data['sampling'].get('sampler', 'greedy'),
                temperature=data['sampling'].get('temperature', 1.0),
                top_k=data['sampling'].get('top_k', 50),
                top_p=data['sampling'].get('top_p', 0.9),
            ),
            inference_modes=InferenceModesConfig(
                speculative=data['inference_modes'].get('speculative', True),
                target_autoregressive=data['inference_modes'].get('target_autoregressive', True),
                drafter_autoregressive=data['inference_modes'].get('drafter_autoregressive', False),
            ),
            debug=DebugConfig(
                enabled=data['debug'].get('enabled', False),
                seed=data['debug'].get('seed', 42),
            ),
            device=data.get('device', 'cuda'),
        )
    
    @classmethod
    def default(cls) -> "Config":
        """Create default configuration."""
        return cls(
            target_model=ModelConfig(name="meta-llama/Llama-3.2-3B-Instruct", quantization="int8"),
            drafter_model=ModelConfig(name="meta-llama/Llama-3.2-1B-Instruct", quantization="int8"),
            generation=GenerationConfig(),
            sampling=SamplingConfig(),
            inference_modes=InferenceModesConfig(),
            debug=DebugConfig(),
        )


class SpeculativeDecodingInference:
    """Speculative decoding inference runner."""

    def __init__(self, config: Config):
        print(
            colored("Speculative Decoding", "red"),
            colored("Inference", on_color="on_red", color="white"),
            "\n",
        )
        self.config = config
        self.device = config.device
        
        # Build sampler from config
        self.sampler = config.sampling.build_sampler()

        self._load_models()
        self._print_config()

    def _load_models(self):
        """Load target and drafter models based on configuration."""
        print(colored("Target model:", on_color="on_yellow"), self.config.target_model.name)
        print(colored("Drafter model:", on_color="on_yellow"), self.config.drafter_model.name)
        print(colored("Loading models...", "light_grey"))

        self.target = AutoModelForCausalLM.from_pretrained(
            self.config.target_model.name,
            quantization_config=self.config.target_model.get_quantization_config(),
            device_map=self.device,
            trust_remote_code=True,
        )
        self.target.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.target_model.name, 
            trust_remote_code=True
        )

        self.drafter = AutoModelForCausalLM.from_pretrained(
            self.config.drafter_model.name,
            quantization_config=self.config.drafter_model.get_quantization_config(),
            device_map=self.device,
            trust_remote_code=True,
        )
        self.drafter.eval()
        
        # End tokens for Llama models
        self.end_tokens = [
            self.tokenizer.eos_token_id, 
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        print(colored("Models loaded successfully!", "green"))

    def _print_config(self):
        """Print current configuration summary."""
        print(colored("\n=== Configuration ===", on_color="on_blue"))
        print(f"  Gamma: {self.config.generation.gamma}")
        print(f"  Max length: {self.config.generation.max_length}")
        print(f"  Sampler: {self.config.sampling.sampler} (T={self.config.sampling.temperature})")
        print(f"  Cache: {self.config.generation.use_cache}")
        print(f"  Chat mode: {self.config.generation.chat_mode}")
        print(colored("  Inference modes:", "cyan"))
        print(f"    Speculative: {self.config.inference_modes.speculative}")
        print(f"    Target AR: {self.config.inference_modes.target_autoregressive}")
        print(f"    Drafter AR: {self.config.inference_modes.drafter_autoregressive}")
        print(colored("=====================\n", on_color="on_blue"))

    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def infer(self, prompt: str) -> dict:
        """
        Run inference on the given prompt.
        
        Args:
            prompt: The input text prompt.
            
        Returns:
            Dictionary containing results from each enabled inference mode.
        """
        if self.config.generation.chat_mode:
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], 
                add_generation_prompt=True, 
                tokenize=False
            )
            
        tokenized = self.tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
        
        results = {}

        # Speculative Decoding
        if self.config.inference_modes.speculative:
            self._set_seed(self.config.debug.seed)
            start_time = time.time()
            output_ids, accept_rate = speculative_generate(
                tokenized,
                self.drafter,
                self.target,
                tokenizer=self.tokenizer,
                logits_processor=self.sampler,
                gamma=self.config.generation.gamma,
                max_gen_len=self.config.generation.max_length,
                eos_tokens_id=self.end_tokens,
                debug=self.config.debug.enabled,
                use_cache=self.config.generation.use_cache,
            )
            elapsed = time.time() - start_time
            output = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            throughput = len(output_ids) / elapsed
            
            results['speculative'] = {
                'output': output,
                'accept_rate': accept_rate,
                'throughput': throughput,
                'time': elapsed,
            }
            
            print(colored("========== Speculative ==========", "green"))
            print(colored("Out:", "green"), output)
            print(colored(f"Acceptance rate: {accept_rate:.3f}", "green"))
            print(colored(f"Throughput: {throughput:.1f} tokens/s", "green"))
            print(colored("========== Speculative ==========", "green"))

        # Target Autoregressive (Baseline)
        if self.config.inference_modes.target_autoregressive:
            self._set_seed(self.config.debug.seed)
            start_time = time.time()
            output_ids = autoregressive_generate(
                tokenized,
                self.target,
                use_cache=self.config.generation.use_cache,
                max_gen_len=self.config.generation.max_length,
                eos_tokens_id=self.end_tokens,
                logits_processor=self.sampler,
                debug=self.config.debug.enabled,
            )
            elapsed = time.time() - start_time
            output = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            throughput = len(output_ids) / elapsed
            
            results['target_ar'] = {
                'output': output,
                'throughput': throughput,
                'time': elapsed,
            }
            
            print(colored("=========== Target AR ===========", "blue"))
            print(colored("Out:", "blue"), output)
            print(colored(f"Throughput: {throughput:.1f} tokens/s", "blue"))
            print(colored("=========== Target AR ===========", "blue"))

        # Drafter Autoregressive
        if self.config.inference_modes.drafter_autoregressive:
            self._set_seed(self.config.debug.seed)
            start_time = time.time()
            output_ids = autoregressive_generate(
                tokenized,
                self.drafter,
                use_cache=self.config.generation.use_cache,
                max_gen_len=self.config.generation.max_length,
                eos_tokens_id=self.end_tokens,
                logits_processor=self.sampler,
                debug=self.config.debug.enabled,
            )
            elapsed = time.time() - start_time
            output = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            throughput = len(output_ids) / elapsed
            
            results['drafter_ar'] = {
                'output': output,
                'throughput': throughput,
                'time': elapsed,
            }
            
            print(colored("========== Drafter AR ==========", "cyan"))
            print(colored("Out:", "cyan"), output)
            print(colored(f"Throughput: {throughput:.1f} tokens/s", "cyan"))
            print(colored("========== Drafter AR ==========", "cyan"))

        # Print speedup comparison
        if 'speculative' in results and 'target_ar' in results:
            speedup = results['speculative']['throughput'] / results['target_ar']['throughput']
            print(colored(f"\nSpeedup (Speculative vs Target AR): {speedup:.2f}x", "magenta"))
        
        return results

    def run_interactive(self):
        """Run interactive prompt loop."""
        print("Enter your prompts (Ctrl+C to exit):\n")
        while True:
            try:
                prompt = input("> ").strip()
                if not prompt:
                    continue
                self.infer(prompt)
                print()
            except KeyboardInterrupt:
                print(colored("\nGoodbye!", on_color="on_red"))
                break
            except Exception as e:
                print(colored(f"Error: {e}", "red"))


def main():
    parser = argparse.ArgumentParser(description="Speculative Decoding Inference")
    parser.add_argument(
        "--config", 
        type=str, 
        default=None,
        help="Path to YAML config file (default: config/base.yaml)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default=None, 
        help="Device override (cuda, cpu, mps)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt to run (non-interactive mode)"
    )
    args = parser.parse_args()

    # Load configuration
    if args.config:
        config_path = args.config
    else:
        config_path = Path(__file__).parent / "config" / "base.yaml"
    
    if Path(config_path).exists():
        print(colored(f"Loading config from: {config_path}", "light_grey"))
        config = Config.from_yaml(str(config_path))
    else:
        print(colored("Config file not found, using defaults", "yellow"))
        config = Config.default()
    
    # Apply CLI overrides
    if args.device:
        config.device = args.device

    # Run inference
    inference = SpeculativeDecodingInference(config=config)
    
    if args.prompt:
        # Single prompt mode
        inference.infer(args.prompt)
    else:
        # Interactive mode
        inference.run_interactive()


if __name__ == "__main__":
    main()
