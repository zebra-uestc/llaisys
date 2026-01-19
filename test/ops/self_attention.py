import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
import llaisys
import torch
from test_utils import random_tensor, check_equal, benchmark, random_int_tensor


def torch_self_attention(attn_val, query, key, value, scale):
    query = query.transpose(-2, -3)
    key = key.transpose(-2, -3)
    value = value.transpose(-2, -3)
    
    L, S = query.size(-2), key.size(-2)

    compute_dtype = query.dtype if query.dtype.is_floating_point else torch.float32

    q_comp = query.to(compute_dtype)
    k_comp = key.to(compute_dtype)
    v_comp = value.to(compute_dtype)

    attn_bias = torch.zeros(L, S, dtype=compute_dtype, device=query.device)

    temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(
        diagonal=S - L
    )
    attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

    k_comp = k_comp.repeat_interleave(q_comp.size(-3) // k_comp.size(-3), -3)
    v_comp = v_comp.repeat_interleave(q_comp.size(-3) // v_comp.size(-3), -3)

    attn_weight = q_comp @ k_comp.transpose(-2, -1) * scale
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    
    output_float = (attn_weight @ v_comp).transpose(-2, -3)

    if not attn_val.dtype.is_floating_point:
        output_float = output_float.round()
    
    attn_val.copy_(output_float)


def test_op_self_attention(
    qlen,
    kvlen,
    nh,
    nkvh,
    hd,
    dtype_name="f32",
    atol=1e-5,
    rtol=1e-5,
    device_name="cpu",
    profile=False,
):
    print(
        f"   qlen={qlen} kvlen={kvlen} nh={nh} nkvh={nkvh} hd={hd} dtype <{dtype_name}>"
    )
    if dtype_name not in ["i8"]:
        q, q_ = random_tensor((qlen, nh, hd), dtype_name, device_name)
        k, k_ = random_tensor((kvlen, nkvh, hd), dtype_name, device_name)
        v, v_ = random_tensor((kvlen, nkvh, hd), dtype_name, device_name)
        scale = 1.0 / (hd**0.5)
        attn_val, attn_val_ = random_tensor((qlen, nh, hd), dtype_name, device_name)
    else:
        q, q_ = random_int_tensor((qlen, nh, hd), device_name, dtype_name, low=-128, high=127)
        k, k_ = random_int_tensor((kvlen, nkvh, hd), device_name, dtype_name, low=-128, high=127)
        v, v_ = random_int_tensor((kvlen, nkvh, hd), device_name, dtype_name, low=-128, high=127)
        scale = 1.0 / (hd**0.5)
        attn_val, attn_val_ = random_int_tensor((qlen, nh, hd), device_name, dtype_name, low=-128, high=127)
    torch_self_attention(attn_val, q, k, v, scale)
    llaisys.Ops.self_attention(attn_val_, q_, k_, v_, scale)
    assert check_equal(attn_val_, attn_val, atol=atol, rtol=rtol, allow_mismatch_ratio=0.0001)

    if profile:
        benchmark(
            lambda: torch_self_attention(attn_val, q, k, v, scale),
            lambda: llaisys.Ops.self_attention(attn_val_, q_, k_, v_, scale),
            device_name,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    testShapes = [
        # qlen, kvlen, nh, nkvh, hd
        (1, 128, 12, 2, 128),
        (128, 128, 12, 2, 128),
        (1, 512, 32, 8, 128),
        (512, 512, 32, 8, 128),
        (1, 1024, 40, 8, 128),
        (1024, 1024, 40, 8, 128)
    ]
    testDtypePrec = [
        # type, atol, rtol
        ("f32", 1e-5, 1e-5),
        ("f16", 1e-3, 1e-3),
        ("bf16", 1e-2, 1e-2),
        ("i8", 0, 0),
    ]
    print(f"Testing Ops.self_attention on {args.device}")
    for shape in testShapes:
        for dtype_name, atol, rtol in testDtypePrec:
            test_op_self_attention(
                *shape, dtype_name, atol, rtol, args.device, args.profile
            )

    print("\033[92mTest passed!\033[0m\n")
