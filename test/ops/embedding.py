import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
import llaisys
from test_utils import random_int_tensor, random_tensor, check_equal, benchmark


def torch_embedding(out, idx, embd):
    out[:] = embd[idx]


def test_op_embedding(
    idx_shape,
    embd_shape,
    dtype_name="f32",
    device_name="cpu",
    profile=False,
):
    print(f"   idx_shape {idx_shape} embd_shape {embd_shape} dtype <{dtype_name}>")
    if dtype_name not in ["i8"]:
        embd, embd_ = random_tensor(embd_shape, dtype_name, device_name)
        out, out_ = random_tensor((idx_shape[0], embd_shape[1]), dtype_name, device_name)
    else:
        embd, embd_ = random_int_tensor(embd_shape, device_name, "i8", low=-128, high=127)
        out, out_ = random_int_tensor((idx_shape[0], embd_shape[1]), device_name, "i8", low=-128, high=127)
    idx, idx_ = random_int_tensor(idx_shape, device_name, high=embd_shape[0])
    #out, out_ = random_tensor((idx_shape[0], embd_shape[1]), dtype_name, device_name)
    torch_embedding(out, idx, embd)
    llaisys.Ops.embedding(out_, idx_, embd_)

    check_equal(out_, out, strict=True)

    if profile:
        benchmark(
            lambda: torch_embedding(out, idx, embd),
            lambda: llaisys.Ops.embedding(out_, idx_, embd_),
            device_name,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    testShapes = [
        ((1,), (151936, 1536)),
        ((128,), (151936, 1536)),
        ((1,), (151936, 4096)),
        ((512,), (151936, 4096)),
        ((1,), (152064, 5120)),
        ((1024,), (152064, 5120)),
    ]
    testDtype = [
        # type
        "f32",
        "f16",
        "bf16",
        "i8",
    ]
    print(f"Testing Ops.embedding on {args.device}")
    for idx_shape, embd_shape in testShapes:
        for dtype_name in testDtype:
            test_op_embedding(
                idx_shape, embd_shape, dtype_name, args.device, args.profile
            )

    print("\033[92mTest passed!\033[0m\n")
