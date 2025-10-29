#pragma once
// c = a * B^T, all row-major
// c: [1, N]
// a: [1, K]
// B: [N, K]
void vecmul(const float *a, const float *B, float *c, int N, int K);