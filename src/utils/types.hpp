#include "llaisys.h"

#include <cstddef>
#include <immintrin.h>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace llaisys {
struct CustomFloat16 {
    uint16_t _v;
};
typedef struct CustomFloat16 fp16_t;

struct CustomBFloat16 {
    uint16_t _v;
};
typedef struct CustomBFloat16 bf16_t;

// FP8 Exponent 5, Mantissa 2
struct CustomFloat8A {
    uint8_t _v;
};
typedef struct CustomFloat8A f8a_t;

// FP8 Exponent 4, Mantissa 3
struct CustomFloat8B {
    uint8_t _v;
};
typedef struct CustomFloat8B f8b_t;

// FP8_E5M2 Lookup Table (256 entries)
const float FP8_E5M2_LUT[256] = {
    0.00000000e+00f,    1.52587891e-05f,    3.05175781e-05f,    4.57763672e-05f,
    6.10351562e-05f,    7.62939453e-05f,    9.15527344e-05f,    1.06811523e-04f,
    1.22070312e-04f,    1.52587891e-04f,    1.83105469e-04f,    2.13623047e-04f,
    2.44140625e-04f,    3.05175781e-04f,    3.66210938e-04f,    4.27246094e-04f,
    4.88281250e-04f,    6.10351562e-04f,    7.32421875e-04f,    8.54492188e-04f,
    9.76562500e-04f,    1.22070312e-03f,    1.46484375e-03f,    1.70898438e-03f,
    1.95312500e-03f,    2.44140625e-03f,    2.92968750e-03f,    3.41796875e-03f,
    3.90625000e-03f,    4.88281250e-03f,    5.85937500e-03f,    6.83593750e-03f,
    7.81250000e-03f,    9.76562500e-03f,    1.17187500e-02f,    1.36718750e-02f,
    1.56250000e-02f,    1.95312500e-02f,    2.34375000e-02f,    2.73437500e-02f,
    3.12500000e-02f,    3.90625000e-02f,    4.68750000e-02f,    5.46875000e-02f,
    6.25000000e-02f,    7.81250000e-02f,    9.37500000e-02f,    1.09375000e-01f,
    1.25000000e-01f,    1.56250000e-01f,    1.87500000e-01f,    2.18750000e-01f,
    2.50000000e-01f,    3.12500000e-01f,    3.75000000e-01f,    4.37500000e-01f,
    5.00000000e-01f,    6.25000000e-01f,    7.50000000e-01f,    8.75000000e-01f,
    1.00000000e+00f,    1.25000000e+00f,    1.50000000e+00f,    1.75000000e+00f,
    2.00000000e+00f,    2.50000000e+00f,    3.00000000e+00f,    3.50000000e+00f,
    4.00000000e+00f,    5.00000000e+00f,    6.00000000e+00f,    7.00000000e+00f,
    8.00000000e+00f,    1.00000000e+01f,    1.20000000e+01f,    1.40000000e+01f,
    1.60000000e+01f,    2.00000000e+01f,    2.40000000e+01f,    2.80000000e+01f,
    3.20000000e+01f,    4.00000000e+01f,    4.80000000e+01f,    5.60000000e+01f,
    6.40000000e+01f,    8.00000000e+01f,    9.60000000e+01f,    1.12000000e+02f,
    1.28000000e+02f,    1.60000000e+02f,    1.92000000e+02f,    2.24000000e+02f,
    2.56000000e+02f,    3.20000000e+02f,    3.84000000e+02f,    4.48000000e+02f,
    5.12000000e+02f,    6.40000000e+02f,    7.68000000e+02f,    8.96000000e+02f,
    1.02400000e+03f,    1.28000000e+03f,    1.53600000e+03f,    1.79200000e+03f,
    2.04800000e+03f,    2.56000000e+03f,    3.07200000e+03f,    3.58400000e+03f,
    4.09600000e+03f,    5.12000000e+03f,    6.14400000e+03f,    7.16800000e+03f,
    8.19200000e+03f,    1.02400000e+04f,    1.22880000e+04f,    1.43360000e+04f,
    1.63840000e+04f,    2.04800000e+04f,    2.45760000e+04f,    2.86720000e+04f,
    3.27680000e+04f,    4.09600000e+04f,    4.91520000e+04f,    5.73440000e+04f,
    INFINITY,    NAN,    NAN,    NAN,
    -0.00000000e+00f,    -1.52587891e-05f,    -3.05175781e-05f,    -4.57763672e-05f,
    -6.10351562e-05f,    -7.62939453e-05f,    -9.15527344e-05f,    -1.06811523e-04f,
    -1.22070312e-04f,    -1.52587891e-04f,    -1.83105469e-04f,    -2.13623047e-04f,
    -2.44140625e-04f,    -3.05175781e-04f,    -3.66210938e-04f,    -4.27246094e-04f,
    -4.88281250e-04f,    -6.10351562e-04f,    -7.32421875e-04f,    -8.54492188e-04f,
    -9.76562500e-04f,    -1.22070312e-03f,    -1.46484375e-03f,    -1.70898438e-03f,
    -1.95312500e-03f,    -2.44140625e-03f,    -2.92968750e-03f,    -3.41796875e-03f,
    -3.90625000e-03f,    -4.88281250e-03f,    -5.85937500e-03f,    -6.83593750e-03f,
    -7.81250000e-03f,    -9.76562500e-03f,    -1.17187500e-02f,    -1.36718750e-02f,
    -1.56250000e-02f,    -1.95312500e-02f,    -2.34375000e-02f,    -2.73437500e-02f,
    -3.12500000e-02f,    -3.90625000e-02f,    -4.68750000e-02f,    -5.46875000e-02f,
    -6.25000000e-02f,    -7.81250000e-02f,    -9.37500000e-02f,    -1.09375000e-01f,
    -1.25000000e-01f,    -1.56250000e-01f,    -1.87500000e-01f,    -2.18750000e-01f,
    -2.50000000e-01f,    -3.12500000e-01f,    -3.75000000e-01f,    -4.37500000e-01f,
    -5.00000000e-01f,    -6.25000000e-01f,    -7.50000000e-01f,    -8.75000000e-01f,
    -1.00000000e+00f,    -1.25000000e+00f,    -1.50000000e+00f,    -1.75000000e+00f,
    -2.00000000e+00f,    -2.50000000e+00f,    -3.00000000e+00f,    -3.50000000e+00f,
    -4.00000000e+00f,    -5.00000000e+00f,    -6.00000000e+00f,    -7.00000000e+00f,
    -8.00000000e+00f,    -1.00000000e+01f,    -1.20000000e+01f,    -1.40000000e+01f,
    -1.60000000e+01f,    -2.00000000e+01f,    -2.40000000e+01f,    -2.80000000e+01f,
    -3.20000000e+01f,    -4.00000000e+01f,    -4.80000000e+01f,    -5.60000000e+01f,
    -6.40000000e+01f,    -8.00000000e+01f,    -9.60000000e+01f,    -1.12000000e+02f,
    -1.28000000e+02f,    -1.60000000e+02f,    -1.92000000e+02f,    -2.24000000e+02f,
    -2.56000000e+02f,    -3.20000000e+02f,    -3.84000000e+02f,    -4.48000000e+02f,
    -5.12000000e+02f,    -6.40000000e+02f,    -7.68000000e+02f,    -8.96000000e+02f,
    -1.02400000e+03f,    -1.28000000e+03f,    -1.53600000e+03f,    -1.79200000e+03f,
    -2.04800000e+03f,    -2.56000000e+03f,    -3.07200000e+03f,    -3.58400000e+03f,
    -4.09600000e+03f,    -5.12000000e+03f,    -6.14400000e+03f,    -7.16800000e+03f,
    -8.19200000e+03f,    -1.02400000e+04f,    -1.22880000e+04f,    -1.43360000e+04f,
    -1.63840000e+04f,    -2.04800000e+04f,    -2.45760000e+04f,    -2.86720000e+04f,
    -3.27680000e+04f,    -4.09600000e+04f,    -4.91520000e+04f,    -5.73440000e+04f,
    -INFINITY,    NAN,    NAN,    NAN
};

// FP8_E4M3 Lookup Table (256 entries)
const float FP8_E4M3_LUT[256] = {
    0.00000000e+00f,    1.95312500e-03f,    3.90625000e-03f,    5.85937500e-03f,
    7.81250000e-03f,    9.76562500e-03f,    1.17187500e-02f,    1.36718750e-02f,
    1.56250000e-02f,    1.75781250e-02f,    1.95312500e-02f,    2.14843750e-02f,
    2.34375000e-02f,    2.53906250e-02f,    2.73437500e-02f,    2.92968750e-02f,
    3.12500000e-02f,    3.51562500e-02f,    3.90625000e-02f,    4.29687500e-02f,
    4.68750000e-02f,    5.07812500e-02f,    5.46875000e-02f,    5.85937500e-02f,
    6.25000000e-02f,    7.03125000e-02f,    7.81250000e-02f,    8.59375000e-02f,
    9.37500000e-02f,    1.01562500e-01f,    1.09375000e-01f,    1.17187500e-01f,
    1.25000000e-01f,    1.40625000e-01f,    1.56250000e-01f,    1.71875000e-01f,
    1.87500000e-01f,    2.03125000e-01f,    2.18750000e-01f,    2.34375000e-01f,
    2.50000000e-01f,    2.81250000e-01f,    3.12500000e-01f,    3.43750000e-01f,
    3.75000000e-01f,    4.06250000e-01f,    4.37500000e-01f,    4.68750000e-01f,
    5.00000000e-01f,    5.62500000e-01f,    6.25000000e-01f,    6.87500000e-01f,
    7.50000000e-01f,    8.12500000e-01f,    8.75000000e-01f,    9.37500000e-01f,
    1.00000000e+00f,    1.12500000e+00f,    1.25000000e+00f,    1.37500000e+00f,
    1.50000000e+00f,    1.62500000e+00f,    1.75000000e+00f,    1.87500000e+00f,
    2.00000000e+00f,    2.25000000e+00f,    2.50000000e+00f,    2.75000000e+00f,
    3.00000000e+00f,    3.25000000e+00f,    3.50000000e+00f,    3.75000000e+00f,
    4.00000000e+00f,    4.50000000e+00f,    5.00000000e+00f,    5.50000000e+00f,
    6.00000000e+00f,    6.50000000e+00f,    7.00000000e+00f,    7.50000000e+00f,
    8.00000000e+00f,    9.00000000e+00f,    1.00000000e+01f,    1.10000000e+01f,
    1.20000000e+01f,    1.30000000e+01f,    1.40000000e+01f,    1.50000000e+01f,
    1.60000000e+01f,    1.80000000e+01f,    2.00000000e+01f,    2.20000000e+01f,
    2.40000000e+01f,    2.60000000e+01f,    2.80000000e+01f,    3.00000000e+01f,
    3.20000000e+01f,    3.60000000e+01f,    4.00000000e+01f,    4.40000000e+01f,
    4.80000000e+01f,    5.20000000e+01f,    5.60000000e+01f,    6.00000000e+01f,
    6.40000000e+01f,    7.20000000e+01f,    8.00000000e+01f,    8.80000000e+01f,
    9.60000000e+01f,    1.04000000e+02f,    1.12000000e+02f,    1.20000000e+02f,
    1.28000000e+02f,    1.44000000e+02f,    1.60000000e+02f,    1.76000000e+02f,
    1.92000000e+02f,    2.08000000e+02f,    2.24000000e+02f,    2.40000000e+02f,
    2.56000000e+02f,    2.88000000e+02f,    3.20000000e+02f,    3.52000000e+02f,
    3.84000000e+02f,    4.16000000e+02f,    4.48000000e+02f,    NAN,
    -0.00000000e+00f,    -1.95312500e-03f,    -3.90625000e-03f,    -5.85937500e-03f,
    -7.81250000e-03f,    -9.76562500e-03f,    -1.17187500e-02f,    -1.36718750e-02f,
    -1.56250000e-02f,    -1.75781250e-02f,    -1.95312500e-02f,    -2.14843750e-02f,
    -2.34375000e-02f,    -2.53906250e-02f,    -2.73437500e-02f,    -2.92968750e-02f,
    -3.12500000e-02f,    -3.51562500e-02f,    -3.90625000e-02f,    -4.29687500e-02f,
    -4.68750000e-02f,    -5.07812500e-02f,    -5.46875000e-02f,    -5.85937500e-02f,
    -6.25000000e-02f,    -7.03125000e-02f,    -7.81250000e-02f,    -8.59375000e-02f,
    -9.37500000e-02f,    -1.01562500e-01f,    -1.09375000e-01f,    -1.17187500e-01f,
    -1.25000000e-01f,    -1.40625000e-01f,    -1.56250000e-01f,    -1.71875000e-01f,
    -1.87500000e-01f,    -2.03125000e-01f,    -2.18750000e-01f,    -2.34375000e-01f,
    -2.50000000e-01f,    -2.81250000e-01f,    -3.12500000e-01f,    -3.43750000e-01f,
    -3.75000000e-01f,    -4.06250000e-01f,    -4.37500000e-01f,    -4.68750000e-01f,
    -5.00000000e-01f,    -5.62500000e-01f,    -6.25000000e-01f,    -6.87500000e-01f,
    -7.50000000e-01f,    -8.12500000e-01f,    -8.75000000e-01f,    -9.37500000e-01f,
    -1.00000000e+00f,    -1.12500000e+00f,    -1.25000000e+00f,    -1.37500000e+00f,
    -1.50000000e+00f,    -1.62500000e+00f,    -1.75000000e+00f,    -1.87500000e+00f,
    -2.00000000e+00f,    -2.25000000e+00f,    -2.50000000e+00f,    -2.75000000e+00f,
    -3.00000000e+00f,    -3.25000000e+00f,    -3.50000000e+00f,    -3.75000000e+00f,
    -4.00000000e+00f,    -4.50000000e+00f,    -5.00000000e+00f,    -5.50000000e+00f,
    -6.00000000e+00f,    -6.50000000e+00f,    -7.00000000e+00f,    -7.50000000e+00f,
    -8.00000000e+00f,    -9.00000000e+00f,    -1.00000000e+01f,    -1.10000000e+01f,
    -1.20000000e+01f,    -1.30000000e+01f,    -1.40000000e+01f,    -1.50000000e+01f,
    -1.60000000e+01f,    -1.80000000e+01f,    -2.00000000e+01f,    -2.20000000e+01f,
    -2.40000000e+01f,    -2.60000000e+01f,    -2.80000000e+01f,    -3.00000000e+01f,
    -3.20000000e+01f,    -3.60000000e+01f,    -4.00000000e+01f,    -4.40000000e+01f,
    -4.80000000e+01f,    -5.20000000e+01f,    -5.60000000e+01f,    -6.00000000e+01f,
    -6.40000000e+01f,    -7.20000000e+01f,    -8.00000000e+01f,    -8.80000000e+01f,
    -9.60000000e+01f,    -1.04000000e+02f,    -1.12000000e+02f,    -1.20000000e+02f,
    -1.28000000e+02f,    -1.44000000e+02f,    -1.60000000e+02f,    -1.76000000e+02f,
    -1.92000000e+02f,    -2.08000000e+02f,    -2.24000000e+02f,    -2.40000000e+02f,
    -2.56000000e+02f,    -2.88000000e+02f,    -3.20000000e+02f,    -3.52000000e+02f,
    -3.84000000e+02f,    -4.16000000e+02f,    -4.48000000e+02f,    NAN
};

namespace utils {
inline size_t dsize(llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return sizeof(char);
    case LLAISYS_DTYPE_BOOL:
        return sizeof(char);
    case LLAISYS_DTYPE_I8:
        return sizeof(int8_t);
    case LLAISYS_DTYPE_I16:
        return sizeof(int16_t);
    case LLAISYS_DTYPE_I32:
        return sizeof(int32_t);
    case LLAISYS_DTYPE_I64:
        return sizeof(int64_t);
    case LLAISYS_DTYPE_U8:
        return sizeof(uint8_t);
    case LLAISYS_DTYPE_U16:
        return sizeof(uint16_t);
    case LLAISYS_DTYPE_U32:
        return sizeof(uint32_t);
    case LLAISYS_DTYPE_U64:
        return sizeof(uint64_t);
    case LLAISYS_DTYPE_F8:
        return 1; // usually 8-bit float (custom)
    case LLAISYS_DTYPE_F16:
        return 2; // 16-bit float
    case LLAISYS_DTYPE_BF16:
        return 2; // bfloat16
    case LLAISYS_DTYPE_F32:
        return sizeof(float);
    case LLAISYS_DTYPE_F64:
        return sizeof(double);
    case LLAISYS_DTYPE_C16:
        return 2; // 2 bytes complex (not standard)
    case LLAISYS_DTYPE_C32:
        return 4; // 4 bytes complex
    case LLAISYS_DTYPE_C64:
        return 8; // 8 bytes complex
    case LLAISYS_DTYPE_C128:
        return 16; // 16 bytes complex
    case LLAISYS_DTYPE_INVALID:
    default:
        throw std::invalid_argument("Unsupported or invalid data type.");
    }
}

inline const char *dtype_to_str(llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return "byte";
    case LLAISYS_DTYPE_BOOL:
        return "bool";
    case LLAISYS_DTYPE_I8:
        return "int8";
    case LLAISYS_DTYPE_I16:
        return "int16";
    case LLAISYS_DTYPE_I32:
        return "int32";
    case LLAISYS_DTYPE_I64:
        return "int64";
    case LLAISYS_DTYPE_U8:
        return "uint8";
    case LLAISYS_DTYPE_U16:
        return "uint16";
    case LLAISYS_DTYPE_U32:
        return "uint32";
    case LLAISYS_DTYPE_U64:
        return "uint64";
    case LLAISYS_DTYPE_F8:
        return "float8";
    case LLAISYS_DTYPE_F16:
        return "float16";
    case LLAISYS_DTYPE_BF16:
        return "bfloat16";
    case LLAISYS_DTYPE_F32:
        return "float32";
    case LLAISYS_DTYPE_F64:
        return "float64";
    case LLAISYS_DTYPE_C16:
        return "complex16";
    case LLAISYS_DTYPE_C32:
        return "complex32";
    case LLAISYS_DTYPE_C64:
        return "complex64";
    case LLAISYS_DTYPE_C128:
        return "complex128";
    case LLAISYS_DTYPE_INVALID:
    default:
        throw std::invalid_argument("Unsupported or invalid data type.");
    }
}

float _f16_to_f32(fp16_t val);
fp16_t _f32_to_f16(float val);

float _bf16_to_f32(bf16_t val);
bf16_t _f32_to_bf16(float val);

float fp16_to_fp32_f16c(fp16_t x);
fp16_t fp32_to_fp16_f16c(float x);

float _fp8_e5m2_to_fp32(f8a_t x);
f8a_t _fp32_to_fp8_e5m2(float x);

float _fp8_e4m3_to_fp32(f8b_t x);
f8b_t _fp32_to_fp8_e4m3(float x);

void fp16_to_fp32_batch_f16c(float *dst, const fp16_t *src, size_t count);
void fp32_to_fp16_batch_f16c(fp16_t *dst, const float *src, size_t count);

void bf16_to_fp32_batch(float *dst, const bf16_t *src, size_t count);
void fp32_to_bf16_batch(bf16_t *dst, const float *src, size_t count);

void int8_to_fp32_batch(float* dst, const int8_t* src, size_t size);
void fp32_to_int8_batch(int8_t* dst, const float* src, size_t size);

void f8a_to_fp32_batch(float *dst, const f8a_t *src, size_t count);
void fp32_to_f8a_batch(f8a_t *dst, const float *src, size_t count);

void f8b_to_fp32_batch(float *dst, const f8b_t *src, size_t count);
void fp32_to_f8b_batch(f8b_t *dst, const float *src, size_t count);

template <typename TypeTo, typename TypeFrom>
TypeTo cast(TypeFrom val) {
    if constexpr (std::is_same<TypeTo, TypeFrom>::value) {
        return val;
    } else if constexpr (std::is_same<TypeTo, fp16_t>::value && std::is_same<TypeFrom, float>::value) {
        return fp32_to_fp16_f16c(val);
    } else if constexpr (std::is_same<TypeTo, fp16_t>::value && !std::is_same<TypeFrom, float>::value) {
        return fp32_to_fp16_f16c(static_cast<float>(val));
    } else if constexpr (std::is_same<TypeFrom, fp16_t>::value && std::is_same<TypeTo, float>::value) {
        return fp16_to_fp32_f16c(val);
    } else if constexpr (std::is_same<TypeFrom, fp16_t>::value && !std::is_same<TypeTo, float>::value) {
        return static_cast<TypeTo>(fp16_to_fp32_f16c(val));
    } else if constexpr (std::is_same<TypeTo, bf16_t>::value && std::is_same<TypeFrom, float>::value) {
        return _f32_to_bf16(val);
    } else if constexpr (std::is_same<TypeTo, bf16_t>::value && !std::is_same<TypeFrom, float>::value) {
        return _f32_to_bf16(static_cast<float>(val));
    } else if constexpr (std::is_same<TypeFrom, bf16_t>::value && std::is_same<TypeTo, float>::value) {
        return _bf16_to_f32(val);
    } else if constexpr (std::is_same<TypeFrom, bf16_t>::value && !std::is_same<TypeTo, float>::value) {
        return static_cast<TypeTo>(_bf16_to_f32(val));
    } else if constexpr (std::is_same<TypeTo, f8a_t>::value && std::is_same<TypeFrom, float>::value) {
        return _fp32_to_fp8_e5m2(val);
    } else if constexpr (std::is_same<TypeTo, f8a_t>::value && !std::is_same<TypeFrom, float>::value) {
        return _fp32_to_fp8_e5m2(static_cast<float>(val));
    } else if constexpr (std::is_same<TypeFrom, f8a_t>::value && std::is_same<TypeTo, float>::value) {
        return _fp8_e5m2_to_fp32(val);
    } else if constexpr (std::is_same<TypeFrom, f8a_t>::value && !std::is_same<TypeTo, float>::value) {
        return static_cast<TypeTo>(_fp8_e5m2_to_fp32(val));
    } else if constexpr (std::is_same<TypeTo, f8b_t>::value && std::is_same<TypeFrom, float>::value) {
        return _fp32_to_fp8_e4m3(val);
    } else if constexpr (std::is_same<TypeTo, f8b_t>::value && !std::is_same<TypeFrom, float>::value) {
        return _fp32_to_fp8_e4m3(static_cast<float>(val));
    } else if constexpr (std::is_same<TypeFrom, f8b_t>::value && std::is_same<TypeTo, float>::value) {
        return _fp8_e4m3_to_fp32(val);
    } else if constexpr (std::is_same<TypeFrom, f8b_t>::value && !std::is_same<TypeTo, float>::value) {
        return static_cast<TypeTo>(_fp8_e4m3_to_fp32(val));
    } else if constexpr (std::is_same<TypeTo, int8_t>::value) {
        float temp = std::nearbyint(val);
        temp = std::clamp(temp, -128.0f, 127.0f);
        return static_cast<TypeTo>(temp);
    } else {
        return static_cast<TypeTo>(val);
    }
}

} // namespace utils
} // namespace llaisys
