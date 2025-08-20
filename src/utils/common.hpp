#include <vector>

namespace llaisys {

namespace ops {
size_t indexToOffset(
    size_t flat_index,
    size_t ndim,
    const std::vector<size_t> &shape,
    const std::vector<ptrdiff_t> &strides);

} // namespace ops

} // namespace llaisys