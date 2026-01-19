import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
import llaisys
import torch
from test_utils import random_tensor, check_equal, benchmark, random_int_tensor


def torch_rms_norm(ans, x, w, eps):
    torch.pow(x, 2, out=ans)
    mean = torch.mean(ans, dim=-1, keepdim=True)
    mean.add_(eps)
    torch.rsqrt(mean, out=mean)
    torch.mul(x, mean, out=ans)
    ans.mul_(w)

def torch_rms_norm_int(ans, x, w, eps):
    x_high = x.to(torch.float32)
    w_high = w.to(torch.float32)
    
    var = torch.pow(x_high, 2)
    
    mean = torch.mean(var, dim=-1, keepdim=True)
    
    mean.add_(eps)
    torch.rsqrt(mean, out=mean)
    
    out_high = x_high * mean * w_high
    
    if ans.dtype in [torch.int8, torch.uint8]:
        out_high = torch.round(out_high)
        out_high = torch.clamp(out_high, -128, 127)
        
    ans.copy_(out_high.to(ans.dtype))

def test_op_rms_norm(
    shape,
    dtype_name="f32",
    atol=1e-5,
    rtol=1e-5,
    device_name="cpu",
    profile=False,
):
    print(f"   shape {shape} dtype <{dtype_name}>")
    if dtype_name not in ["i8"]:
        x, x_ = random_tensor(shape, dtype_name, device_name)
        w, w_ = random_tensor((shape[1],), dtype_name, device_name)
        c, c_ = random_tensor(shape, dtype_name, device_name)
    else:
        x, x_ = random_int_tensor(shape, device_name, dtype_name, low=-128, high=127)
        w, w_ = random_int_tensor((shape[1],), device_name, dtype_name, low=-128, high=127)
        c, c_ = random_int_tensor(shape, device_name, dtype_name, low=-128, high=127)
    eps = 1e-5
    if dtype_name not in ["i8"]:
        torch_rms_norm(c, x, w, eps)
    else:
        torch_rms_norm_int(c, x, w, eps)
    llaisys.Ops.rms_norm(c_, x_, w_, eps)

    assert check_equal(c_, c, atol=atol, rtol=rtol, allow_mismatch_ratio=0.0001)

    if profile:
        if dtype_name not in ["i8"]:
            benchmark(
                lambda: torch_rms_norm(c, x, w, eps),
                lambda: llaisys.Ops.rms_norm(c_, x_, w_, eps),
                device_name,
            )
        else:
            benchmark(
                lambda: torch_rms_norm_int(c, x, w, eps),
                lambda: llaisys.Ops.rms_norm(c_, x_, w_, eps),
                device_name,
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    testShapes = [
        (1, 1536),
        (128, 1536),
        (1, 4096),
        (512, 4096),
        (1, 5120),
        (1024, 5120),
    ]
    testDtypePrec = [
        # type, atol, rtol
        ("f32", 1e-5, 1e-5),
        ("f16", 1e-3, 1e-3),
        ("bf16", 1e-2, 1e-2),
        ("i8", 0, 0),
    ]
    print(f"Testing Ops.rms_norm on {args.device}")
    for shape in testShapes:
        for dtype_name, atol, rtol in testDtypePrec:
            test_op_rms_norm(shape, dtype_name, atol, rtol, args.device, args.profile)

    print("\033[92mTest passed!\033[0m\n")
