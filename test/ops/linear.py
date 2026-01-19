import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
import llaisys
import torch
from test_utils import random_tensor, check_equal, benchmark, random_int_tensor, zero_tensor


def torch_linear(out, x, w, bias, scales=None):
    if w.dtype == torch.int8:
        # 确定计算精度（跟随输入 x 的精度，通常是 bf16 或 fp32）
        compute_dtype = x.dtype
        
        # 将 int8 权重转换为计算精度
        w_fp = w.to(compute_dtype)
        
        # 处理 Scale 的形状以支持广播 (Per-Channel)
        # 假设 w shape 为 [OC, IC]，scales 原始 shape 通常为 [OC]
        # 我们需要将 scales 变为 [OC, 1] 以便进行 w * s 运算
        if scales.dim() == 1:
            s_reshaped = scales.view(-1, 1)
        else:
            s_reshaped = scales
            
        # 反量化 (Dequantize): W_real = W_int8 * Scale
        w_dequant = w_fp * s_reshaped
        
        # 确保 bias 也是正确的精度
        b_fp = bias.to(compute_dtype) if bias is not None else None
        
        # 执行标准的线性计算 (FP16/BF16/FP32)
        # F.linear(input, weight, bias) -> input @ weight.T + bias
        res = torch.nn.functional.linear(x, w_dequant, b_fp)
        
        # 将结果写入 out
        # 注意：如果 out 是 fp32 而 res 是 bf16，这里会自动转换
        out.copy_(res)
        
    else:
        # 如果 w 不是 int8，回退到普通线性层逻辑
        torch.nn.functional.linear(x, w, bias, out=out)




def test_op_linear(
    out_shape,
    x_shape,
    w_shape,
    use_bias=True,
    dtype_name="f32",
    atol=1e-5,
    rtol=1e-5,
    device_name="cpu",
    profile=False,
):
    print(
        f"   out {out_shape}, x {x_shape}, w {w_shape}, bias {use_bias}, dtype <{dtype_name}>"
    )
    if dtype_name not in ["i8"]:
        x, x_ = random_tensor(x_shape, dtype_name, device_name, scale=0.1)
        w, w_ = random_tensor(w_shape, dtype_name, device_name, scale=0.01)
    else:
        x, x_ = random_tensor(x_shape, "f32" if device_name == "cpu" else "bf16", device_name, scale=0.1)
        w, w_ = random_int_tensor(w_shape, device_name, dtype_name, low=-128, high=127)

    bias, bias_ = None, None
    if use_bias:
        if dtype_name not in ["i8"]:
            bias, bias_ = random_tensor((w_shape[0],), dtype_name, device_name)
        else:
            bias, bias_ = random_tensor((w_shape[0],), "f32" if device_name == "cpu" else "bf16", device_name)
            # bias, bias_ = zero_tensor((w_shape[0],), "bf16", device_name)
            scales, scales_ = random_tensor((w_shape[0],), "f32" if device_name == "cpu" else "bf16", device_name, scale=0.02)

    if dtype_name not in ["i8"]:
        out, out_ = random_tensor(out_shape, dtype_name, device_name)
    else:
        out, out_ = random_tensor(out_shape, "f32" if device_name == "cpu" else "bf16", device_name)

    if dtype_name not in ["i8"]:
        torch_linear(out, x, w, bias)
        llaisys.Ops.linear(out_, x_, w_, bias_)
    else:
        torch_linear(out, x, w, bias, scales)
        llaisys.Ops.linear(out_, x_, w_, bias_, scales_)

    assert check_equal(out_, out, atol=atol, rtol=rtol, allow_mismatch_ratio=0.0001)

    if profile:
        if dtype_name not in ["i8"]:
            benchmark(
                lambda: torch_linear(out, x, w, bias),
                lambda: llaisys.Ops.linear(out_, x_, w_, bias_),
                device_name,
            )
        else:
            benchmark(
                lambda: torch_linear(out, x, w, bias, scales),
                lambda: llaisys.Ops.linear(out_, x_, w_, bias_, scales_),
                device_name,
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    testShapes = [
        ((1, 1536), (1, 1536), (1536, 1536), True),
        ((1, 8960), (1, 1536), (8960, 1536), True),
        ((1, 1536), (1, 8960), (1536, 8960), True),
        ((128, 1536), (128, 1536), (1536, 1536), True),
        ((128, 8960), (128, 1536), (8960, 1536), True),
        ((128, 1536), (128, 8960), (1536, 8960), True),
        ((1, 4096), (1, 4096), (4096, 4096), True),
        ((1, 12288), (1, 4096), (12288, 4096), True),
        ((1, 4096), (1, 12288), (4096, 12288), True),
        ((512, 4096), (512, 4096), (4096, 4096), True),
        ((512, 12288), (512, 4096), (12288, 4096), True),
        ((512, 4096), (512, 12288), (4096, 12288), True),
        ((1, 5120), (1, 5120), (5120, 5120), True),
        ((1, 27648), (1, 5120), (27648, 5120), True),
        ((1, 5120), (1, 27648), (5120, 27648), True),
        ((1024, 5120), (1024, 5120), (5120, 5120), True),
        ((1024, 27648), (1024, 5120), (27648, 5120), True),
        ((1024, 5120), (1024, 27648), (5120, 27648), True),
    ]
    testDtypePrec = [
        # type, atol, rtol
        ("f32", 5e-5, 5e-5),
        ("f16", 1e-3, 1e-3),
        ("bf16", 5e-2, 5e-2),
        ("i8", 6e-2, 6e-2),
    ]
    print(f"Testing Ops.linear on {args.device}")
    for shapes in testShapes:
        for dtype_name, atol, rtol in testDtypePrec:
            test_op_linear(*shapes, dtype_name, atol, rtol, args.device, args.profile)

    print("\033[92mTest passed!\033[0m\n")
