import gc
from test_utils import *

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import snapshot_download
import os
import time
import llaisys
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def load_hf_model(model_path=None, device_name="cpu"):
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
    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer.encode(input_content, return_tensors="pt").to(model.device)

    with torch.no_grad():
        # Prefill阶段计时
        prefill_start = time.time()
        # 使用forward而不是generate，只计算prefill
        outputs = model(inputs, use_cache=True)
        prefill_end = time.time()
        prefill_time = prefill_end - prefill_start

        print(f"🤖 HF Prefill Time: {prefill_time:.4f}s")

        # Decode阶段计时
        decode_start = time.time()
        # 继续生成剩余的token
        if max_new_tokens > 0:
            generated = model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                use_cache=True,  # 确保使用KV缓存
                pad_token_id=tokenizer.eos_token_id,
            )
        else:
            generated = inputs
        decode_end = time.time()
        decode_time = decode_end - decode_start

        print(f"🌀 HF Decode Time: {decode_time:.4f}s")
        print(f"📊 HF Avg Decode Time per token: {decode_time / max_new_tokens:.4f}s")

    result = tokenizer.decode(generated[0], skip_special_tokens=True)
    return generated[0].tolist(), result, prefill_time, decode_time


def load_llaisys_model(model_path, device_name):
    model = llaisys.models.Qwen2(model_path, llaisys_device(device_name))
    return model


def llaisys_infer(
    prompt, tokenizer, model, max_new_tokens=128, top_p=0.8, top_k=50, temperature=0.8
):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--model", default=None, type=str)
    parser.add_argument("--prompt", default="Who are you?", type=str)
    parser.add_argument("--max_steps", default=128, type=int)
    parser.add_argument("--top_p", default=0.8, type=float)
    parser.add_argument("--top_k", default=50, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()

    top_p, top_k, temperature = args.top_p, args.top_k, args.temperature
    if args.test:
        top_p, top_k, temperature = 1.0, 1, 1.0

    start_time = time.time()
    tokenizer, model, model_path = load_hf_model(args.model, args.device)
    end_time = time.time()
    print(f"\n\n\nLoad HF Model => Time elapsed: {(end_time - start_time):.2f}s\n")

    # Example prompt
    # start_time = time.time()
    # tokens, output = hf_infer(
    #     args.prompt,
    #     tokenizer,
    #     model,
    #     max_new_tokens=args.max_steps,
    #     top_p=top_p,
    #     top_k=top_k,
    #     temperature=temperature,
    # )
    # end_time = time.time()

    start_time = time.time()
    tokens, output, _, _ = hf_infer_with_timing(
        args.prompt,
        tokenizer,
        model,
        max_new_tokens=args.max_steps,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
    )
    end_time = time.time()

    del model
    gc.collect()

    print("\n=== Answer ===\n")
    print("Tokens:")
    print(tokens)
    print("\nContents:")
    print(output)
    print("\n")
    print(f"Time elapsed: {(end_time - start_time):.2f}s\n")

    start_time = time.time()
    model = load_llaisys_model(model_path, args.device)
    end_time = time.time()
    print(f"Load LLAISYS Model => Time elapsed: {(end_time - start_time):.2f}s\n")

    start_time = time.time()
    llaisys_tokens, llaisys_output = llaisys_infer(
        args.prompt,
        tokenizer,
        model,
        max_new_tokens=args.max_steps,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
    )

    end_time = time.time()

    print("\n=== Your Result ===\n")
    print("Tokens:")
    print(llaisys_tokens)
    print("\nContents:")
    print(llaisys_output)
    print("\n")
    print(f"Time elapsed: {(end_time - start_time):.2f}s\n")

    if args.test:
        assert llaisys_tokens == tokens
        print("\033[92mTest passed!\033[0m\n")
