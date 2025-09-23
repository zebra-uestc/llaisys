#pragma once
#include <cstddef>

namespace llaisys::ops::cpu {
float sdot(const float *x, const float *y, const size_t N);
}