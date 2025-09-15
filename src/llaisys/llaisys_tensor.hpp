#pragma once
#include "llaisys/tensor.h"

#include "../tensor/tensor.hpp"

__LLAISYS__C {
    typedef struct LlaisysTensor {
        llaisys::tensor_t tensor;
    } LlaisysTensor;
}
