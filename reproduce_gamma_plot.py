
import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, QuantoConfig
from decoding import dynamic_speculative_generate, DynamicGammaScheduler
from utils.sampling_strategies import GreedySampler
import time
from pathlib import Path

def main():
    # Configuration
    target_model_name = "meta-llama/Llama-3.2-3B-Instruct"
    drafter_model_name = "meta-llama/Llama-3.2-1B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    quantization = None
    
    print(f"Using device: {device}")
    
    # Load models
    print("Loading models...")
    quant_config = None
    if quantization:
        quant_config = QuantoConfig(weights=quantization)
    
    target = AutoModelForCausalLM.from_pretrained(
        target_model_name,
        quantization_config=quant_config,
        trust_remote_code=True,
    )
    target.to(device)
    target.eval()
    
    drafter = AutoModelForCausalLM.from_pretrained(
        drafter_model_name,
        quantization_config=quant_config,
        trust_remote_code=True,
    )
    drafter.to(device)
    drafter.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(target_model_name)
    
    # Prepare prompt
    prompt = "Explain what machine learning is in simple terms."  # medium_qa
    print(f"Prompt: {prompt}")
    
    messages = [{"role": "user", "content": prompt}]
    prompt_formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(prompt_formatted, return_tensors="pt").input_ids[0].tolist()
    
    # Run Dynamic Speculative Decoding
    print("Running Dynamic Speculative Decoding...")
    scheduler = DynamicGammaScheduler(
        initial_gamma=4,
        min_gamma=1,
        max_gamma=10,
        schedule="heuristic"
    )
    
    start_time = time.time()
    output_ids, accept_rate, stats = dynamic_speculative_generate(
        inputs,
        drafter,
        target,
        scheduler=scheduler,
        sampler=GreedySampler(),
        max_gen_len=100,  # Generate enough tokens to see trends
        eos_tokens_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
        use_cache=True 
    )
    elapsed = time.time() - start_time
    
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    print(f"Output: {output_text[:100]}...") # Print first 100 chars
    print(f"Time: {elapsed:.2f}s")
    print(f"Acceptance Rate: {accept_rate:.2f}")
    
    gamma_history = stats.get("gamma_history", [])
    if not gamma_history:
        print("Error: No gamma history found in stats!")
        return

    print(f"Generated {len(gamma_history)} speculative steps.")

    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot Dynamic Gamma
    plt.plot(range(len(gamma_history)), gamma_history, label='Dynamic Gamma', marker='o', linestyle='-')
    
    # Plot Static Gamma (Reference)
    plt.axhline(y=4, color='r', linestyle='--', label='Static Gamma (4)')
    
    plt.title('Dynamic Gamma vs Static Gamma over Speculative Iterations')
    plt.xlabel('Speculative Iteration')
    plt.ylabel('Gamma Value')
    plt.legend()
    plt.grid(True)
    
    output_file = "gamma_plot.png"
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    
    # Also save raw data for verification
    np.save("gamma_history.npy", gamma_history)
    print("Gamma history saved to gamma_history.npy")

if __name__ == "__main__":
    main()
