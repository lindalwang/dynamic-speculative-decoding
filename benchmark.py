"""
Comprehensive Benchmark: Autoregressive vs Static Speculative vs Dynamic Speculative Decoding

Compares performance across various prompts and generation lengths.
Outputs results to console and CSV file.
"""

import argparse
import random
import numpy as np
import torch
import yaml
import csv
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict
from datetime import datetime

from decoding import speculative_generate, dynamic_speculative_generate, DynamicGammaScheduler
from decoding.baseline import autoregressive_generate
from utils.sampling_strategies import (
    Sampler, GreedySampler, MultinomialSampler, 
    TopKSampler, NucleusSampler, TopKNucleusSampler
)
from transformers import AutoTokenizer, AutoModelForCausalLM, QuantoConfig
from termcolor import colored


# Benchmark prompts of varying complexity
BENCHMARK_PROMPTS = {
    "short_qa": [
        "What is 2+2?",
        "What color is the sky?",
        "Name a planet.",
    ],
    "medium_qa": [
        "Explain what machine learning is in simple terms.",
        "What are the benefits of exercise?",
        "Describe the water cycle.",
    ],
    "long_qa": [
        "Write a detailed explanation of how neural networks learn, including the concepts of forward propagation, backpropagation, and gradient descent.",
        "Explain the history and evolution of programming languages from assembly to modern high-level languages.",
        "Describe the process of photosynthesis in plants, including the light-dependent and light-independent reactions.",
    ],
    "creative": [
        "Write a haiku about the ocean.",
        "Start a story: The door creaked open and...",
        "Write a short poem about autumn leaves.",
    ],
    "translation": [
        "Translate to English: Je m'appelle Romain. N'hésitez pas à contribuer à mon projet !",
        "Translate to English: Guten Tag, wie geht es Ihnen?",
        "Translate to French: Hello, how are you today?",
    ],
    "code": [
        "Write a Python function to calculate factorial.",
        "Write a simple hello world program in JavaScript.",
        "Explain what a for loop does and give an example.",
    ],
}

GENERATION_LENGTHS = [25, 50, 100]


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark."""
    target_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    drafter_model: str = "meta-llama/Llama-3.2-1B-Instruct"
    quantization: Optional[str] = "int8"
    device: str = "cuda"
    gamma: int = 4
    min_gamma: int = 1
    max_gamma: int = 10
    num_runs: int = 3  # Number of runs per prompt for averaging
    warmup_runs: int = 1
    seed: int = 42
    # Sampling configuration
    sampler_type: str = "greedy"  # Options: greedy, multinomial, topk, nucleus, topknucleus
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9


class Benchmark:
    """Benchmark runner for comparing decoding methods."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: List[Dict] = []
        
        print(colored("=" * 60, "cyan"))
        print(colored("  Speculative Decoding Benchmark", "cyan", attrs=["bold"]))
        print(colored("=" * 60, "cyan"))
        
        self._load_models()
        self.sampler = self._create_sampler()
    
    def _create_sampler(self) -> Sampler:
        """Create sampler based on config."""
        sampler_type = self.config.sampler_type.lower()
        temp = self.config.temperature
        
        if sampler_type == "greedy":
            sampler = GreedySampler(temperature=temp)
        elif sampler_type == "multinomial":
            sampler = MultinomialSampler(temperature=temp)
        elif sampler_type == "topk":
            sampler = TopKSampler(temperature=temp, top_k=self.config.top_k)
        elif sampler_type == "nucleus":
            sampler = NucleusSampler(temperature=temp, top_p=self.config.top_p)
        elif sampler_type == "topknucleus":
            sampler = TopKNucleusSampler(temperature=temp, top_k=self.config.top_k, top_p=self.config.top_p)
        else:
            print(colored(f"Unknown sampler '{sampler_type}', defaulting to greedy", "yellow"))
            sampler = GreedySampler(temperature=temp)
        
        print(colored(f"  Sampler: {sampler_type} (temp={temp})", "green"))
        return sampler
        
    def _load_models(self):
        """Load target and drafter models."""
        print(colored("\nLoading models...", "yellow"))
        
        quant_config = QuantoConfig(weights=self.config.quantization) if self.config.quantization else None
        
        print(f"  Target: {self.config.target_model}")
        self.target = AutoModelForCausalLM.from_pretrained(
            self.config.target_model,
            quantization_config=quant_config,
            device_map=self.config.device,
            trust_remote_code=True,
        )
        self.target.eval()
        
        print(f"  Drafter: {self.config.drafter_model}")
        self.drafter = AutoModelForCausalLM.from_pretrained(
            self.config.drafter_model,
            quantization_config=quant_config,
            device_map=self.config.device,
            trust_remote_code=True,
        )
        self.drafter.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.target_model,
            trust_remote_code=True
        )
        
        self.end_tokens = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        print(colored("Models loaded successfully!\n", "green"))
    
    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def _prepare_prompt(self, prompt: str) -> List[int]:
        """Prepare prompt with chat template."""
        formatted = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False
        )
        return self.tokenizer(formatted, return_tensors="pt").input_ids[0].tolist()
    
    def _run_autoregressive(self, tokens: List[int], max_len: int) -> Dict:
        """Run autoregressive decoding."""
        self._set_seed(self.config.seed)
        
        start = time.perf_counter()
        output_ids = autoregressive_generate(
            tokens,
            self.target,
            max_gen_len=max_len,
            sampler=self.sampler,
            eos_tokens_id=self.end_tokens,
            use_cache=False,
        )
        elapsed = time.perf_counter() - start
        
        return {
            "method": "autoregressive",
            "tokens_generated": len(output_ids),
            "time": elapsed,
            "throughput": len(output_ids) / elapsed if elapsed > 0 else 0,
            "accept_rate": None,
            "final_gamma": None,
        }
    
    def _run_speculative(self, tokens: List[int], max_len: int) -> Dict:
        """Run static speculative decoding."""
        self._set_seed(self.config.seed)
        
        start = time.perf_counter()
        output_ids, accept_rate = speculative_generate(
            tokens,
            self.drafter,
            self.target,
            gamma=self.config.gamma,
            sampler=self.sampler,
            max_gen_len=max_len,
            eos_tokens_id=self.end_tokens,
            use_cache=False,
        )
        elapsed = time.perf_counter() - start
        
        return {
            "method": "speculative",
            "tokens_generated": len(output_ids),
            "time": elapsed,
            "throughput": len(output_ids) / elapsed if elapsed > 0 else 0,
            "accept_rate": accept_rate,
            "final_gamma": self.config.gamma,
        }
    
    def _run_dynamic_speculative(self, tokens: List[int], max_len: int) -> Dict:
        """Run dynamic speculative decoding."""
        self._set_seed(self.config.seed)
        
        scheduler = DynamicGammaScheduler(
            initial_gamma=self.config.gamma,
            min_gamma=self.config.min_gamma,
            max_gamma=self.config.max_gamma,
            schedule="heuristic",
        )
        
        start = time.perf_counter()
        output_ids, accept_rate, stats = dynamic_speculative_generate(
            tokens,
            self.drafter,
            self.target,
            scheduler=scheduler,
            sampler=self.sampler,
            max_gen_len=max_len,
            eos_tokens_id=self.end_tokens,
            use_cache=False,
        )
        elapsed = time.perf_counter() - start
        
        return {
            "method": "dynamic_speculative",
            "tokens_generated": len(output_ids),
            "time": elapsed,
            "throughput": len(output_ids) / elapsed if elapsed > 0 else 0,
            "accept_rate": accept_rate,
            "final_gamma": stats.get("final_gamma") if stats else None,
        }
    
    def run_single_benchmark(self, prompt: str, category: str, max_len: int) -> List[Dict]:
        """Run all methods on a single prompt."""
        tokens = self._prepare_prompt(prompt)
        prompt_len = len(tokens)
        results = []
        
        # Warmup
        for _ in range(self.config.warmup_runs):
            self._run_autoregressive(tokens, max_len)
        
        for method_name, method_fn in [
            ("autoregressive", self._run_autoregressive),
            ("speculative", self._run_speculative),
            ("dynamic_speculative", self._run_dynamic_speculative),
        ]:
            run_results = []
            for run in range(self.config.num_runs):
                result = method_fn(tokens, max_len)
                run_results.append(result)
            
            # Average results
            avg_result = {
                "category": category,
                "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt,
                "prompt_len": prompt_len,
                "max_gen_len": max_len,
                "method": method_name,
                "tokens_generated": np.mean([r["tokens_generated"] for r in run_results]),
                "time": np.mean([r["time"] for r in run_results]),
                "throughput": np.mean([r["throughput"] for r in run_results]),
                "accept_rate": np.mean([r["accept_rate"] for r in run_results if r["accept_rate"] is not None]) if any(r["accept_rate"] is not None for r in run_results) else None,
                "final_gamma": run_results[-1]["final_gamma"],
            }
            results.append(avg_result)
        
        return results
    
    def run_full_benchmark(self, categories: List[str] = None, lengths: List[int] = None):
        """Run full benchmark suite."""
        if categories is None:
            categories = list(BENCHMARK_PROMPTS.keys())
        if lengths is None:
            lengths = GENERATION_LENGTHS
        
        total_tests = sum(len(BENCHMARK_PROMPTS.get(cat, [])) for cat in categories) * len(lengths)
        current_test = 0
        
        print(colored(f"\nRunning {total_tests} benchmark configurations...\n", "cyan"))
        
        for category in categories:
            prompts = BENCHMARK_PROMPTS.get(category, [])
            if not prompts:
                continue
                
            print(colored(f"\n{'='*60}", "blue"))
            print(colored(f"Category: {category}", "blue", attrs=["bold"]))
            print(colored(f"{'='*60}", "blue"))
            
            for prompt in prompts:
                for max_len in lengths:
                    current_test += 1
                    print(f"\n[{current_test}/{total_tests}] Testing: {prompt[:40]}... (max_len={max_len})")
                    
                    results = self.run_single_benchmark(prompt, category, max_len)
                    self.results.extend(results)
                    
                    # Print immediate results
                    ar_result = next(r for r in results if r["method"] == "autoregressive")
                    spec_result = next(r for r in results if r["method"] == "speculative")
                    dyn_result = next(r for r in results if r["method"] == "dynamic_speculative")
                    
                    print(f"  AR:      {ar_result['throughput']:.1f} tok/s")
                    print(f"  Spec:    {spec_result['throughput']:.1f} tok/s (accept={spec_result['accept_rate']:.2f})")
                    print(f"  Dynamic: {dyn_result['throughput']:.1f} tok/s (accept={dyn_result['accept_rate']:.2f}, final_γ={dyn_result['final_gamma']})")
                    
                    spec_speedup = spec_result['throughput'] / ar_result['throughput'] if ar_result['throughput'] > 0 else 0
                    dyn_speedup = dyn_result['throughput'] / ar_result['throughput'] if ar_result['throughput'] > 0 else 0
                    print(colored(f"  Speedup: Spec={spec_speedup:.2f}x, Dynamic={dyn_speedup:.2f}x", "green"))
    
    def print_summary(self):
        """Print summary statistics."""
        print(colored("\n" + "=" * 60, "cyan"))
        print(colored("  BENCHMARK SUMMARY", "cyan", attrs=["bold"]))
        print(colored("=" * 60, "cyan"))
        
        # Aggregate by method
        methods = ["autoregressive", "speculative", "dynamic_speculative"]
        
        for method in methods:
            method_results = [r for r in self.results if r["method"] == method]
            if not method_results:
                continue
            
            avg_throughput = np.mean([r["throughput"] for r in method_results])
            avg_time = np.mean([r["time"] for r in method_results])
            
            print(f"\n{method.upper()}:")
            print(f"  Avg Throughput: {avg_throughput:.2f} tokens/s")
            print(f"  Avg Time: {avg_time:.3f}s")
            
            if method != "autoregressive":
                accept_rates = [r["accept_rate"] for r in method_results if r["accept_rate"] is not None]
                if accept_rates:
                    print(f"  Avg Accept Rate: {np.mean(accept_rates):.3f}")
        
        # Overall speedups
        ar_throughputs = {(r["category"], r["prompt"], r["max_gen_len"]): r["throughput"] 
                         for r in self.results if r["method"] == "autoregressive"}
        
        spec_speedups = []
        dyn_speedups = []
        
        for r in self.results:
            key = (r["category"], r["prompt"], r["max_gen_len"])
            ar_tp = ar_throughputs.get(key, 0)
            if ar_tp > 0:
                if r["method"] == "speculative":
                    spec_speedups.append(r["throughput"] / ar_tp)
                elif r["method"] == "dynamic_speculative":
                    dyn_speedups.append(r["throughput"] / ar_tp)
        
        print(colored("\nOVERALL SPEEDUPS:", "green", attrs=["bold"]))
        if spec_speedups:
            print(f"  Static Speculative: {np.mean(spec_speedups):.2f}x (std={np.std(spec_speedups):.2f})")
        if dyn_speedups:
            print(f"  Dynamic Speculative: {np.mean(dyn_speedups):.2f}x (std={np.std(dyn_speedups):.2f})")
        
        # Compare static vs dynamic
        if spec_speedups and dyn_speedups:
            dyn_vs_spec = np.mean(dyn_speedups) / np.mean(spec_speedups)
            print(f"\n  Dynamic vs Static: {dyn_vs_spec:.2f}x")
    
    def save_results(self, filename: str = None):
        """Save results to CSV."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.csv"
        
        filepath = Path(filename)
        
        with open(filepath, "w", newline="") as f:
            if self.results:
                writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
                writer.writeheader()
                writer.writerows(self.results)
        
        print(colored(f"\nResults saved to: {filepath}", "green"))
        return filepath


def main():
    parser = argparse.ArgumentParser(description="Benchmark Speculative Decoding Methods")
    parser.add_argument("--target", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--drafter", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--quantization", type=str, default="int8", help="int8, int4, or none")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--num-runs", type=int, default=3, help="Number of runs per test")
    parser.add_argument("--categories", nargs="+", default=None, help="Prompt categories to test")
    parser.add_argument("--lengths", nargs="+", type=int, default=None, help="Generation lengths to test")
    parser.add_argument("--output", type=str, default=None, help="Output CSV filename")
    parser.add_argument("--quick", action="store_true", help="Quick test with fewer prompts")
    # Sampling arguments
    parser.add_argument("--sampler", type=str, default="greedy", 
                        choices=["greedy", "multinomial", "topk", "nucleus", "topknucleus"],
                        help="Sampling strategy")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k for topk/topknucleus sampling")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p for nucleus/topknucleus sampling")
    args = parser.parse_args()
    
    config = BenchmarkConfig(
        target_model=args.target,
        drafter_model=args.drafter,
        quantization=args.quantization if args.quantization != "none" else None,
        device=args.device,
        gamma=args.gamma,
        num_runs=args.num_runs,
        sampler_type=args.sampler,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    
    benchmark = Benchmark(config)
    
    # Quick mode uses subset
    if args.quick:
        categories = ["medium_qa", "translation"]
        lengths = [25, 50]
    else:
        categories = args.categories
        lengths = args.lengths
    
    benchmark.run_full_benchmark(categories=categories, lengths=lengths)
    benchmark.print_summary()
    benchmark.save_results(args.output)


if __name__ == "__main__":
    main()

