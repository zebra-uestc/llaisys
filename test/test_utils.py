import llaisys
import torch


def random_tensor(
    shape, dtype_name, device_name, device_id=0, scale=None, bias=None
) -> tuple[torch.Tensor, llaisys.Tensor]:
    torch_tensor = torch.rand(
        shape,
        dtype=torch_dtype(dtype_name),
        device=torch_device(device_name, device_id),
    )
    if scale is not None:
        torch_tensor *= scale
    if bias is not None:
        torch_tensor += bias

    llaisys_tensor = llaisys.Tensor(
        shape,
        dtype=llaisys_dtype(dtype_name),
        device=llaisys_device(device_name),
        device_id=device_id,
    )

    api = llaisys.RuntimeAPI(llaisys_device(device_name))
    bytes_ = torch_tensor.numel() * torch_tensor.element_size()
    api.memcpy_sync(
        llaisys_tensor.data_ptr(),
        torch_tensor.data_ptr(),
        bytes_,
        llaisys.MemcpyKind.D2D,
    )

    return torch_tensor, llaisys_tensor


def random_int_tensor(shape, device_name, dtype_name="i64", device_id=0, low=0, high=2):
    torch_tensor = torch.randint(
        low,
        high,
        shape,
        dtype=torch_dtype(dtype_name),
        device=torch_device(device_name, device_id),
    )

    llaisys_tensor = llaisys.Tensor(
        shape,
        dtype=llaisys_dtype(dtype_name),
        device=llaisys_device(device_name),
        device_id=device_id,
    )

    api = llaisys.RuntimeAPI(llaisys_device(device_name))
    bytes_ = torch_tensor.numel() * torch_tensor.element_size()
    api.memcpy_sync(
        llaisys_tensor.data_ptr(),
        torch_tensor.data_ptr(),
        bytes_,
        llaisys.MemcpyKind.D2D,
    )

    return torch_tensor, llaisys_tensor


def zero_tensor(
    shape, dtype_name, device_name, device_id=0
) -> tuple[torch.Tensor, llaisys.Tensor]:
    torch_tensor = torch.zeros(
        shape,
        dtype=torch_dtype(dtype_name),
        device=torch_device(device_name, device_id),
    )

    llaisys_tensor = llaisys.Tensor(
        shape,
        dtype=llaisys_dtype(dtype_name),
        device=llaisys_device(device_name),
        device_id=device_id,
    )

    api = llaisys.RuntimeAPI(llaisys_device(device_name))
    bytes_ = torch_tensor.numel() * torch_tensor.element_size()
    api.memcpy_sync(
        llaisys_tensor.data_ptr(),
        torch_tensor.data_ptr(),
        bytes_,
        llaisys.MemcpyKind.D2D,
    )

    return torch_tensor, llaisys_tensor


def arrange_tensor(
    start, end, device_name, device_id=0
) -> tuple[torch.Tensor, llaisys.Tensor]:
    torch_tensor = torch.arange(start, end, device=torch_device(device_name, device_id))
    llaisys_tensor = llaisys.Tensor(
        (end - start,),
        dtype=llaisys_dtype("i64"),
        device=llaisys_device(device_name),
        device_id=device_id,
    )

    api = llaisys.RuntimeAPI(llaisys_device(device_name))
    bytes_ = torch_tensor.numel() * torch_tensor.element_size()
    api.memcpy_sync(
        llaisys_tensor.data_ptr(),
        torch_tensor.data_ptr(),
        bytes_,
        llaisys.MemcpyKind.D2D,
    )

    return torch_tensor, llaisys_tensor


def check_equal(
    llaisys_result: llaisys.Tensor,
    torch_answer: torch.Tensor,
    atol=1e-5,
    rtol=1e-5,
    strict=False,
    allow_mismatch_ratio=0.0,
):
    shape = llaisys_result.shape()
    strides = llaisys_result.strides()
    
    assert shape == torch_answer.shape, f"Shape mismatch: {shape} vs {torch_answer.shape}"
    assert torch_dtype(dtype_name(llaisys_result.dtype())) == torch_answer.dtype, "Dtype mismatch"

    right = 0
    for i in range(len(shape)):
        if strides[i] > 0:
            right += strides[i] * (shape[i] - 1)
        else:
            raise ValueError("Negative strides are not supported yet")

    tmp = torch.zeros(
        (right + 1,),
        dtype=torch_answer.dtype,
        device=torch_device(
            device_name(llaisys_result.device_type()), llaisys_result.device_id()
        ),
    )
    result = torch.as_strided(tmp, shape, strides)
    
    api = llaisys.RuntimeAPI(llaisys_result.device_type())
    api.memcpy_sync(
        result.data_ptr(),
        llaisys_result.data_ptr(),
        (right + 1) * tmp.element_size(),
        llaisys.MemcpyKind.D2D,
    )

    res_cpu = result.cpu()
    ans_cpu = torch_answer.cpu()
    
    is_integer = not res_cpu.is_floating_point()

    passed = False
    if strict:
        passed = torch.equal(res_cpu, ans_cpu)
    else:
        passed = torch.allclose(res_cpu, ans_cpu, atol=atol, rtol=rtol)

    if passed:
        return True
    
    diff = (res_cpu - ans_cpu).abs()
    
    if strict:
        mismatch_mask = res_cpu != ans_cpu
    else:
        tol = atol + rtol * ans_cpu.abs()
        mismatch_mask = diff > tol

    num_errors = torch.count_nonzero(mismatch_mask).item()
    total_elements = res_cpu.numel()
    error_ratio = num_errors / total_elements

    if num_errors > 0:
        if error_ratio <= allow_mismatch_ratio:
            return True

    print(f"\n{'='*17} Check Equal Failed {'='*17}")
    print(f"Shape: {shape}")
    print(f"Total Mismatches: {num_errors} / {total_elements} ({error_ratio:.2%})")
    print(f"Max Difference: {diff.max().item()}")

    if num_errors > 0:
        error_indices = torch.nonzero(mismatch_mask, as_tuple=False)
        print("\n--- First 10 Mismatches (Index: Llaisys vs Torch | Diff) ---")
        
        for i, idx_tensor in enumerate(error_indices[:10]):
            idx = tuple(idx_tensor.tolist())
            val_dut = res_cpu[idx].item()
            val_ref = ans_cpu[idx].item()
            val_diff = diff[idx].item()
            
            if is_integer:
                 print(f"  {idx}: {int(val_dut):4d} vs {int(val_ref):4d} | Diff: {int(val_diff)}")
            else:
                 print(f"  {idx}: {val_dut:.6f} vs {val_ref:.6f} | Diff: {val_diff:.6f}")
    
    print("="*54 + "\n")
    return False


def benchmark(torch_func, llaisys_func, device_name, warmup=10, repeat=100):
    api = llaisys.RuntimeAPI(llaisys_device(device_name))

    def time_op(func):
        import time

        for _ in range(warmup):
            func()
        api.device_synchronize()
        start = time.time()
        for _ in range(repeat):
            func()
        api.device_synchronize()
        end = time.time()
        return (end - start) / repeat

    torch_time = time_op(torch_func)
    llaisys_time = time_op(llaisys_func)
    print(
        f"        Torch time: {torch_time*1000:.5f} ms \n        LLAISYS time: {llaisys_time*1000:.5f} ms"
    )


def torch_device(device_name: str, device_id=0):
    if device_name == "cpu":
        return torch.device("cpu")
    elif device_name == "nvidia":
        return torch.device(f"cuda:{device_id}")
    else:
        raise ValueError(f"Unsupported device name: {device_name}")


def llaisys_device(device_name: str):
    if device_name == "cpu":
        return llaisys.DeviceType.CPU
    elif device_name == "nvidia":
        return llaisys.DeviceType.NVIDIA
    else:
        raise ValueError(f"Unsupported device name: {device_name}")


def device_name(llaisys_device: llaisys.DeviceType):
    if llaisys_device == llaisys.DeviceType.CPU:
        return "cpu"
    elif llaisys_device == llaisys.DeviceType.NVIDIA:
        return "nvidia"
    else:
        raise ValueError(f"Unsupported llaisys device: {llaisys_device}")


def torch_dtype(dtype_name: str):
    if dtype_name == "f16":
        return torch.float16
    elif dtype_name == "f32":
        return torch.float32
    elif dtype_name == "f64":
        return torch.float64
    elif dtype_name == "bf16":
        return torch.bfloat16
    elif dtype_name == "i32":
        return torch.int32
    elif dtype_name == "i64":
        return torch.int64
    elif dtype_name == "u32":
        return torch.uint32
    elif dtype_name == "u64":
        return torch.uint64
    elif dtype_name == "bool":
        return torch.bool
    elif dtype_name == "i8":
        return torch.int8
    else:
        raise ValueError(f"Unsupported dtype name: {dtype_name}")


def llaisys_dtype(dtype_name: str):
    if dtype_name == "f16":
        return llaisys.DataType.F16
    elif dtype_name == "f32":
        return llaisys.DataType.F32
    elif dtype_name == "f64":
        return llaisys.DataType.F64
    elif dtype_name == "bf16":
        return llaisys.DataType.BF16
    elif dtype_name == "i32":
        return llaisys.DataType.I32
    elif dtype_name == "i64":
        return llaisys.DataType.I64
    elif dtype_name == "u32":
        return llaisys.DataType.U32
    elif dtype_name == "u64":
        return llaisys.DataType.U64
    elif dtype_name == "bool":
        return llaisys.DataType.BOOL
    elif dtype_name == "i8":
        return llaisys.DataType.I8
    else:
        raise ValueError(f"Unsupported dtype name: {dtype_name}")


def dtype_name(llaisys_dtype: llaisys.DataType):
    if llaisys_dtype == llaisys.DataType.F16:
        return "f16"
    elif llaisys_dtype == llaisys.DataType.F32:
        return "f32"
    elif llaisys_dtype == llaisys.DataType.F64:
        return "f64"
    elif llaisys_dtype == llaisys.DataType.BF16:
        return "bf16"
    elif llaisys_dtype == llaisys.DataType.I32:
        return "i32"
    elif llaisys_dtype == llaisys.DataType.I64:
        return "i64"
    elif llaisys_dtype == llaisys.DataType.U32:
        return "u32"
    elif llaisys_dtype == llaisys.DataType.U64:
        return "u64"
    elif llaisys_dtype == llaisys.DataType.BOOL:
        return "bool"
    elif llaisys_dtype == llaisys.DataType.I8:
        return "i8"
    else:
        raise ValueError(f"Unsupported llaisys dtype: {llaisys_dtype}")
