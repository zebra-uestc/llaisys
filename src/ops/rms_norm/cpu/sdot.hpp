#pragma once
#include <cstddef>

namespace llaisys::ops::cpu {
float sdot(const float *x, const float *y, const size_t N);
int32_t sdot_int8(const int8_t *x, const int8_t *y, const size_t N);
}