import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
import llaisys
import torch
from test_utils import random_tensor, check_equal, benchmark, random_int_tensor


def torch_swiglu(out, gate, up):
    if not torch.is_floating_point(out):        
        gate_f = gate.to(torch.float32)
        up_f = up.to(torch.float32)
        
        sigmoid_val = 1 / (1 + torch.exp(-gate_f))
        res_f = up_f * (gate_f * sigmoid_val)
        
        if out.dtype == torch.int8:
            res_f = torch.clamp(res_f, min=-128, max=127)
        elif out.dtype == torch.uint8:
            res_f = torch.clamp(res_f, min=0, max=255)
            
        res_f = torch.round(res_f)
        
        out.copy_(res_f.to(out.dtype))
        
    else:
        torch.mul(up, gate / (1 + torch.exp(-gate.float()).to(out.dtype)), out=out)


def test_op_swiglu(
    shape,
    dtype_name="f32",
    atol=1e-5,
    rtol=1e-5,
    device_name="cpu",
    profile=False,
):
    print(f"   shape {shape} dtype <{dtype_name}>")
    if dtype_name not in ["i8"]:
        gate, gate_ = random_tensor(shape, dtype_name, device_name)
        up, up_ = random_tensor(shape, dtype_name, device_name)
        out, out_ = random_tensor(shape, dtype_name, device_name)
    else:
        gate, gate_ = random_int_tensor(shape, device_name, dtype_name)
        up, up_ = random_int_tensor(shape, device_name, dtype_name)
        out, out_ = random_int_tensor(shape, device_name, dtype_name)
    torch_swiglu(out, gate, up)
    llaisys.Ops.swiglu(out_, gate_, up_)

    assert check_equal(out_, out, atol=atol, rtol=rtol)

    if profile:
        benchmark(
            lambda: torch_swiglu(out, gate, up),
            lambda: llaisys.Ops.swiglu(out_, gate_, up_),
            device_name,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    testShapes = [
        (1, 8960),
        (128, 8960),
        (1, 12288),
        (512, 12288),
        (1, 27648),
        (1024, 27648),
    ]
    testDtypePrec = [
        # type, atol, rtol
        ("f32", 1e-5, 1e-5),
        ("f16", 1e-3, 1e-3),
        ("bf16", 1e-2, 1e-2),
        ("i8", 0, 0),
    ]
    print(f"Testing Ops.swiglu on {args.device}")
    for shape in testShapes:
        for dtype_name, atol, rtol in testDtypePrec:
            test_op_swiglu(shape, dtype_name, atol, rtol, args.device, args.profile)

    print("\033[92mTest passed!\033[0m\n")
