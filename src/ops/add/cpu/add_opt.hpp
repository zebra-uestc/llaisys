#include "../../../utils.hpp"

void add_bf16(llaisys::bf16_t *c, const llaisys::bf16_t *a, const llaisys::bf16_t *b, size_t numel);
void add_f16(llaisys::fp16_t *c, const llaisys::fp16_t *a, const llaisys::fp16_t *b, size_t numel);
void add_f32(float *c, const float *a, const float *b, size_t numel);
void add_i8(int8_t *c, const int8_t *a, const int8_t *b, size_t numel);