import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
import llaisys
import torch
from test_utils import random_tensor, check_equal, benchmark, random_int_tensor


def torch_add(ans, a, b):
    torch.add(a, b, out=ans)


def test_op_add(
    shape,
    dtype_name="f32",
    atol=1e-5,
    rtol=1e-5,
    device_name="cpu",
    profile=False,
):
    print(f"   shape {shape} dtype <{dtype_name}>")
    if dtype_name != "i8":
        a, a_ = random_tensor(shape, dtype_name, device_name)
        b, b_ = random_tensor(shape, dtype_name, device_name)

        c, c_ = random_tensor(shape, dtype_name, device_name)
    else:
        a, a_ = random_int_tensor(shape, device_name, "i8", low=-128, high=127)
        b, b_ = random_int_tensor(shape, device_name, "i8", low=-128, high=127)

        c, c_ = random_int_tensor(shape, device_name, "i8", low=-128, high=127)
    torch_add(c, a, b)
    llaisys.Ops.add(c_, a_, b_)

    assert check_equal(c_, c, atol=atol, rtol=rtol)

    if profile:
        benchmark(
            lambda: torch_add(c, a, b),
            lambda: llaisys.Ops.add(c_, a_, b_),
            device_name,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    testShapes = [(1, 1536), (128, 1536), (1, 4096), (512, 4096), (1, 5120), (1024, 5120)]
    testDtypePrec = [
        # type, atol, rtol
        ("f32", 1e-5, 1e-5),
        ("f16", 1e-3, 1e-3),
        ("bf16", 1e-2, 1e-2),
        ("i8", 0, 0),
    ]
    print(f"Testing Ops.add on {args.device}")
    for shape in testShapes:
        for dtype_name, atol, rtol in testDtypePrec:
            test_op_add(shape, dtype_name, atol, rtol, args.device, args.profile)

    print("\033[92mTest passed!\033[0m\n")
