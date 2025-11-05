import argparse
import gc
import io
import os
import sys
import time

import torch
from huggingface_hub import snapshot_download
from test_utils import llaisys_device, torch_device
from transformers import AutoModelForCausalLM, AutoTokenizer

import llaisys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def load_hf_model(model_path=None, device_name="cpu"):
    """Load Hugging Face model and tokenizer from local path or hub."""
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    if model_path and os.path.isdir(model_path):
        print(f"Loading model from local path: {model_path}")
    else:
        print(f"Loading model from Hugging Face: {model_id}")
        model_path = snapshot_download(model_id)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map=torch_device(device_name),
        trust_remote_code=True,
    )

    return tokenizer, model, model_path


def hf_infer(
    prompt, tokenizer, model, max_new_tokens=128, top_p=0.8, top_k=50, temperature=0.8
):
    """Run inference with Hugging Face model."""
    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer.encode(input_content, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return outputs[0].tolist(), result


def hf_infer_with_timing(
    prompt, tokenizer, model, max_new_tokens=128, top_p=0.8, top_k=50, temperature=0.8
):
    """Run inference with separate prefill and decode timing measurements."""
    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer.encode(input_content, return_tensors="pt").to(model.device)

    with torch.no_grad():
        # Measure prefill phase
        prefill_start = time.time()
        model(inputs, use_cache=True)
        prefill_time = time.time() - prefill_start
        print(f"🤖 HF Prefill Time: {prefill_time:.4f}s")

        # Measure decode phase
        decode_start = time.time()
        if max_new_tokens > 0:
            generated = model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        else:
            generated = inputs
        decode_time = time.time() - decode_start

        print(f"🌀 HF Decode Time: {decode_time:.4f}s")
        print(f"📊 HF Avg Decode Time per token: {decode_time / max_new_tokens:.4f}s")

    result = tokenizer.decode(generated[0], skip_special_tokens=True)
    return generated[0].tolist(), result, prefill_time, decode_time


def load_llaisys_model(model_path, device_name):
    """Load LLAISYS model from path."""
    model = llaisys.models.Qwen2(model_path, llaisys_device(device_name))
    return model


def llaisys_infer(
    prompt, tokenizer, model, max_new_tokens=128, top_p=0.8, top_k=50, temperature=0.8
):
    """Run inference with LLAISYS model."""
    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer.encode(input_content)
    outputs = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
    )

    return outputs, tokenizer.decode(outputs, skip_special_tokens=True)


def main():
    """Main execution function for model comparison."""
    parser = argparse.ArgumentParser(
        description="Compare HF and LLAISYS model inference"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "nvidia"],
        help="Device to run inference on",
    )
    parser.add_argument("--model", default=None, help="Path to local model directory")
    parser.add_argument(
        "--prompt", default="Who are you?", help="Input prompt for inference"
    )
    parser.add_argument(
        "--max_steps",
        default=128,
        type=int,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--top_p", default=0.8, type=float, help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--top_k", default=50, type=int, help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--temperature", default=1.0, type=float, help="Sampling temperature"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode with deterministic settings",
    )

    args = parser.parse_args()

    # Use deterministic sampling for testing
    top_p = 1.0 if args.test else args.top_p
    top_k = 1 if args.test else args.top_k
    temperature = 1.0 if args.test else args.temperature

    # Load and run HuggingFace model
    print("=" * 60)
    print("Loading HuggingFace Model...")
    start_time = time.time()
    tokenizer, hf_model, model_path = load_hf_model(args.model, args.device)
    print(f"Load Time: {time.time() - start_time:.2f}s\n")

    print("Running HuggingFace Inference...")
    start_time = time.time()
    tokens, output, _, _ = hf_infer_with_timing(
        args.prompt,
        tokenizer,
        hf_model,
        max_new_tokens=args.max_steps,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
    )
    hf_time = time.time() - start_time

    # Clean up HF model
    del hf_model
    gc.collect()

    print("\n=== HuggingFace Results ===")
    print(f"Tokens: {tokens}")
    print(f"Output: {output}")
    print(f"Total Time: {hf_time:.2f}s\n")

    # Load and run LLAISYS model
    print("=" * 60)
    print("Loading LLAISYS Model...")
    start_time = time.time()
    llaisys_model = load_llaisys_model(model_path, args.device)
    print(f"Load Time: {time.time() - start_time:.2f}s\n")

    print("Running LLAISYS Inference...")
    start_time = time.time()
    llaisys_tokens, llaisys_output = llaisys_infer(
        args.prompt,
        tokenizer,
        llaisys_model,
        max_new_tokens=args.max_steps,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
    )
    llaisys_time = time.time() - start_time

    print("\n=== LLAISYS Results ===")
    print(f"Tokens: {llaisys_tokens}")
    print(f"Output: {llaisys_output}")
    print(f"Total Time: {llaisys_time:.2f}s\n")

    # Verify results in test mode
    if args.test:
        assert llaisys_tokens == tokens, "Token mismatch between HF and LLAISYS!"
        print("\033[92m✓ Test passed! Results match.\033[0m\n")

    print("=" * 60)


if __name__ == "__main__":
    main()
