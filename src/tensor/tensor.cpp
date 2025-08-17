#include "tensor.hpp"

#include "../utils.hpp"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

size_t Tensor::dim(size_t i) const {
    return _meta.shape[i];
}

ptrdiff_t Tensor::stride(size_t i) const {
    return _meta.strides[i];
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

/**
 * @brief Checks if the tensor's storage is contiguous in row-major (C-style) order.
 *
 * A tensor is considered contiguous if its elements are stored in memory such that
 * the logical index ordering (row-major) matches the physical memory layout —
 * i.e., each dimension's stride equals the product of the sizes of all lower dimensions.
 *
 * @return true if the tensor is contiguous; false otherwise.
 * @note This function does not modify the tensor. It only inspects metadata.
 */
bool Tensor::isContiguous() const {
    const auto &shape = _meta.shape;
    const auto &strides = _meta.strides;

    size_t ndim = shape.size();
    ptrdiff_t expected_stride = 1;

    // Traverse dimensions from last to first (row-major: innermost dimension first)
    for (int i = static_cast<int>(ndim) - 1; i >= 0; --i) {
        // If current stride doesn't match expected dense layout, not contiguous
        if (strides[i] != expected_stride) {
            return false;
        }
        // Update expected stride for next (outer) dimension
        expected_stride *= shape[i];
    }

    return true;
}

/**
 * @brief Returns a new tensor view with dimensions permuted according to the given order.
 *
 * This function creates a logical reordering of the tensor's dimensions without modifying
 * the underlying memory layout. The returned tensor shares the same storage and offset as
 * the original — it is a zero-copy view. Strides are rearranged to reflect the new dimension
 * ordering, allowing correct indexing even if the tensor was originally non-contiguous.
 *
 * Example:
 *   Original: shape = [2, 3, 4], stride = [12, 4, 1] (contiguous)
 *   permute({2, 0, 1}) → new shape = [4, 2, 3], new stride = [1, 12, 4]
 *
 *   Even if original was non-contiguous (e.g., from transpose), permute still works correctly:
 *   Original: shape = [2, 3, 4], stride = [4, 8, 1] (non-contiguous)
 *   permute({2, 0, 1}) → new shape = [4, 2, 3], new stride = [1, 4, 8]
 *
 * @param order A permutation vector of size `ndim()`, where `order[i]` indicates that
 *              the i-th dimension in the output corresponds to the `order[i]`-th dimension
 *              in the input. Must be a valid permutation of {0, 1, ..., ndim()-1}.
 * @return A new Tensor view with permuted dimensions, sharing this tensor's storage.
 * @throws std::invalid_argument if `order.size() != ndim()`
 * @throws std::invalid_argument if `order` does not contain each index from 0 to ndim()-1 exactly once.
 * @note This is a zero-copy operation. Changes to the returned tensor affect the original.
 *       No contiguity check is performed — permute works on any tensor, contiguous or not.
 */
tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    auto ndim_ = ndim();

    // Validate: order must have same number of dimensions
    CHECK_ARGUMENT(order.size() == ndim_, "order.size() must equal tensor's number of dimensions");

    std::vector<size_t> new_shape(ndim_);
    std::vector<ptrdiff_t> new_strides(ndim_);

    // For each output dimension i, map it to input dimension order[i]
    // This works regardless of whether the original tensor is contiguous
    for (size_t i = 0; i < ndim_; ++i) {
        // Ensure every input dimension index appears exactly once in order
        auto it = std::find(order.begin(), order.end(), i);
        CHECK_ARGUMENT(it != order.end(),
                       "order must contain each dimension index (0 to ndim-1) exactly once");

        // Assign new shape and stride by reindexing the original
        new_shape[i] = dim(order[i]);      // Shape of output dim i = input dim order[i]'s shape
        new_strides[i] = stride(order[i]); // Stride of output dim i = input dim order[i]'s stride
    }

    // Create new tensor metadata sharing the same storage and offset
    TensorMeta new_meta{_meta.dtype, new_shape, new_strides};
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage));
}

/**
 * @brief Creates a new tensor view with the specified shape, sharing the same storage.
 *
 * This function returns a tensor that logically reinterprets the underlying data
 * as having the given shape, without copying or modifying any memory. The original
 * tensor must be contiguous — otherwise, the mapping between logical indices and
 * physical memory would be ambiguous.
 *
 * @param shape The desired new shape. Must contain the same number of elements
 *              as the current tensor (i.e., prod(shape) == numel()).
 * @return A new Tensor object sharing this tensor's storage, with updated metadata.
 * @throws std::runtime_error if the tensor is not contiguous.
 * @throws std::runtime_error if the total number of elements in `shape` does not match.
 * @note This is a zero-copy operation. Changes to the returned tensor affect
 *       the original tensor's data. Use reshape() if you need automatic contiguity handling.
 */
tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    // Ensure tensor is contiguous — view requires deterministic memory layout
    ASSERT(this->isContiguous(), "requires contiguous tensor");

    // Validate element count compatibility
    size_t numel = std::accumulate(shape.begin(), shape.end(), (size_t)1, std::multiplies<size_t>());
    CHECK_ARGUMENT(numel == this->numel(), "shape is incompatible with number of elements");

    // Construct new strides assuming row-major contiguous layout
    std::vector<ptrdiff_t> new_strides(shape.size());
    ptrdiff_t stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        new_strides[i] = stride;
        stride *= static_cast<ptrdiff_t>(shape[i]);
    }

    // Create new tensor metadata sharing the same storage and offset
    TensorMeta new_meta{_meta.dtype, shape, new_strides};
    return std::shared_ptr<Tensor>(new Tensor(new_meta, this->_storage, this->_offset));
}

/**
 * @brief Returns a new tensor view that slices a range along a specified dimension.
 *
 * Creates a view of the tensor by extracting elements from index `start` (inclusive)
 * to `end` (exclusive) along dimension `dim`. The returned tensor shares the same
 * underlying storage as the original, with updated shape and offset — no data is copied.
 *
 * Example:
 *   Original: shape = [3, 4], stride = [4, 1], dtype = float32
 *   slice(0, 1, 3) → new shape = [2, 4], offset = 4 * sizeof(float) = 16 bytes
 *   Result: views rows 1 and 2 (i.e., skips first row).
 *
 * @param dim The dimension along which to slice. Must be in [0, ndim()).
 * @param start Starting index (inclusive) for slicing. Must satisfy 0 <= start < end.
 * @param end Ending index (exclusive) for slicing. Must satisfy start < end <= shape[dim].
 * @return A new Tensor view with reduced size along `dim`, sharing this tensor's storage.
 * @throws std::invalid_argument if `dim` is out of bounds, or if `start`/`end` are invalid.
 * @note This is a zero-copy operation. Modifications to the returned tensor affect the original.
 *       The slice may be non-contiguous if the original tensor was not contiguous along `dim`.
 */
tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    // Validate input arguments
    CHECK_ARGUMENT(dim < ndim() && start < end && end <= _meta.shape[dim],
                   "dim out of bounds, or start > end, or end exceeds dimension size");

    const size_t slice_size = end - start;

    // Construct new shape: only the sliced dimension changes
    std::vector<size_t> new_shape = _meta.shape;
    new_shape[dim] = slice_size;

    // Calculate byte offset into the original storage:
    // Offset = (number of elements skipped along dim) × (stride along dim) × (element size)
    // Note: stride[dim] gives number of elements to skip per step in this dimension.
    // Multiply by dsize to convert to byte offset.
    const size_t offset_bytes = static_cast<size_t>(_meta.strides[dim]) * start * utils::dsize(_meta.dtype);

    // Create new tensor metadata sharing the same storage and offset
    TensorMeta new_meta{_meta.dtype, new_shape, _meta.strides};
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, offset_bytes));
}

/**
 * @brief Load data from host memory into the tensor's storage.
 *
 * Synchronously copies `_storage->size()` bytes from the host buffer `src_`
 * to this tensor's device memory. The tensor's device context is automatically
 * set to match its own device type and ID before the copy to ensure correctness
 * in multi-device environments.
 *
 * @param src_ Pointer to source data in host memory. Must be non-null and point
 *             to a buffer of exactly `_storage->size()` bytes, with data layout
 *             compatible with the tensor's dtype (e.g., float32, int64).
 * @throws std::invalid_argument if `src_` is nullptr or if storage size is zero.
 * @note This operation is blocking. For high-performance scenarios, consider
 *       using asynchronous memcpy with streams.
 */
void Tensor::load(const void *src_) {
    // Validate input: src_ must not be null
    CHECK_ARGUMENT(src_ != nullptr, "source buffer is nullptr");

    // Validate storage: must have non-zero size
    ASSERT(_storage->size() != 0, "storage size must be no-zero");

    // Ensure the runtime device matches this tensor's affinity
    llaisysDeviceType_t device_type = deviceType();
    int device_id = deviceId();
    if (device_type != core::context().runtime().deviceType() || device_id != core::context().runtime().deviceId()) {
        core::context().setDevice(device_type, device_id);
    }

    // Perform synchronous host-to-device memory copy
    // Assumes src_ contains exactly _storage->size() bytes of valid data
    core::context().runtime().api()->memcpy_sync(
        _storage->memory(), // Destination: device memory
        src_,               // Source: host memory
        _storage->size(),   // Size in bytes
        LLAISYS_MEMCPY_H2D  // Direction: Host -> Device
    );
}

tensor_t Tensor::contiguous() const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

} // namespace llaisys
