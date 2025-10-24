# LLAISYS（Let's Learn AI SYStem）

本项目源自 [启元人工智能训练营 · 大模型推理框架方向](https://beta.infinitensor.com/camp/summer2025/stage/1/course/llm-inference-and-serving-system) 的课程作业，旨在帮助初学者从零构建大模型推理系统。当前版本已实现基础功能，并作为新推理框架 [neollm]() 的试验平台，主要用于：

- 验证自定义算子实现的**正确性**（与 Hugging Face 推理结果对齐）  
- 评估自定义算子的**性能**

> 📌 **注意**：本项目**不再新增功能**，仅用于算子开发与性能测试。

---

## ✅ 当前实现

- **模型支持**：Qwen2 模型在 CPU 上的完整推理流程（含 KV Cache）
- **算子优化**：
  - 使用 **AVX 指令集** 加速核心计算
  - 通过 **OpenMP** 实现多线程并行
  - 高效数据类型转换：
    - `f16 <=> f32`：利用 **F16C 指令**
    - `bf16 <=> f32`：利用 **AVX 指令**

---

## 🚀 快速开始

### 编译与安装

```bash
# 编译 C++ 后端
xmake

# 安装共享库
xmake install

# 安装 Python 前端包
pip install ./python/
```
### 算子测试

> 以 `add` 算子为例。
```bash
# 正确性测试（CPU）
python test/ops/add.py

# 正确性测试（NVIDIA GPU）
python test/ops/add.py --nvidia

# 性能分析（CPU）
python test/ops/add.py --profile

# 性能分析（NVIDIA GPU）
python test/ops/add.py --nvidia --profile
```

### 推理验证

```bash
python test/test_infer.py --model /path/to/qwen2/model --test
```
> 🔍 启用 `--test` 选项将关闭 Top-p、Top-k 和 Temperature 采样，强制 Hugging Face 使用 ArgMax（贪婪采样），确保与本推理引擎的采样结果严格一致，便于调试与验证。

## 🔜 后续工作（试验方向）

1. 算子优化与扩展
    - 进一步优化现有算子性能
    - 实现新算子（如 LayerNorm、Softmax 等）
2. 量化支持
    - 开发 INT4/INT8/FP8 量化算子
    - 支持端侧高效推理
  
## 📚 相关资源

- 训练营课程：[大模型推理与服务系统](https://beta.infinitensor.com/camp/summer2025/stage/1/course/llm-inference-and-serving-system)
- 项目仓库：[InfiniTensor/llaisys](https://github.com/InfiniTensor/llaisys)
- README: [English](README_EN.md) |  [中文](README_ZN.md)