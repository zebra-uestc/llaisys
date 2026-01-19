#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up, float gate_scale = 1.0f, float up_scale = 1.0f, float output_scale = 1.0f);
}
