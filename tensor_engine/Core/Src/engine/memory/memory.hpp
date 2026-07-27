#ifndef ENGINE_MEMORY_HPP_
#define ENGINE_MEMORY_HPP_

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>

namespace engine {

// Enumeration of supported weight data types
enum class DataType {
    INT8,
    INT16,
    INT32,
    FLOAT16,
    FLOAT32
};

namespace memory {

// IEEE 754 Single (32-bit) <-> Half (16-bit) conversion utilities
inline uint16_t float32_to_float16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = bits & 0x007FFFFF;

    if (exp <= 0) {
        return static_cast<uint16_t>(sign);
    } else if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00);
    }
    return static_cast<uint16_t>(sign | (exp << 10) | (mant >> 13));
}

inline float float16_to_float32(uint16_t h) {
    uint32_t sign = static_cast<uint32_t>(h & 0x8000) << 16;
    int32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x03FF;
    uint32_t f_bits;

    if (exp == 0) {
        f_bits = sign;
    } else if (exp == 31) {
        f_bits = sign | 0x7F800000 | (mant << 13);
    } else {
        f_bits = sign | (static_cast<uint32_t>(exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &f_bits, sizeof(float));
    return f;
}

// Encapsulates multi-datatype weight buffer storage (INT8, INT16, INT32, FLOAT16, FLOAT32)
struct WeightBuffer {
    DataType dtype;
    size_t count;
    float scale;
    int32_t zero_point;
    void* raw_data;

    static size_t element_size(DataType type) {
        switch (type) {
            case DataType::INT8:    return sizeof(int8_t);
            case DataType::INT16:   return sizeof(int16_t);
            case DataType::INT32:   return sizeof(int32_t);
            case DataType::FLOAT16: return sizeof(uint16_t);
            case DataType::FLOAT32: return sizeof(float);
            default:                return sizeof(float);
        }
    }

    WeightBuffer(DataType type, size_t n_elements, float sc = 1.0f, int32_t zp = 0)
        : dtype(type), count(n_elements), scale(sc), zero_point(zp), raw_data(nullptr) {
        if (n_elements > 0) {
            size_t bytes = n_elements * element_size(type);
            raw_data = std::malloc(bytes);
            if (raw_data) std::memset(raw_data, 0, bytes);
        }
    }

    ~WeightBuffer() {
        if (raw_data) {
            std::free(raw_data);
            raw_data = nullptr;
        }
    }

    WeightBuffer(const WeightBuffer&) = delete;
    WeightBuffer& operator=(const WeightBuffer&) = delete;

    void pack_from_float(const float* src, size_t n) {
        if (!raw_data || !src) return;
        size_t limit = (n < count) ? n : count;

        switch (dtype) {
            case DataType::INT8: {
                int8_t* dst = static_cast<int8_t*>(raw_data);
                for (size_t i = 0; i < limit; ++i) {
                    long val = std::lround(src[i] / scale) + zero_point;
                    if (val < -128) val = -128;
                    if (val > 127)  val = 127;
                    dst[i] = static_cast<int8_t>(val);
                }
                break;
            }
            case DataType::INT16: {
                int16_t* dst = static_cast<int16_t*>(raw_data);
                for (size_t i = 0; i < limit; ++i) {
                    long val = std::lround(src[i] / scale) + zero_point;
                    if (val < -32768) val = -32768;
                    if (val > 32767)  val = 32767;
                    dst[i] = static_cast<int16_t>(val);
                }
                break;
            }
            case DataType::INT32: {
                int32_t* dst = static_cast<int32_t*>(raw_data);
                for (size_t i = 0; i < limit; ++i) {
                    long val = std::lround(src[i] / scale) + zero_point;
                    dst[i] = static_cast<int32_t>(val);
                }
                break;
            }
            case DataType::FLOAT16: {
                uint16_t* dst = static_cast<uint16_t*>(raw_data);
                for (size_t i = 0; i < limit; ++i) {
                    dst[i] = float32_to_float16(src[i]);
                }
                break;
            }
            case DataType::FLOAT32: {
                float* dst = static_cast<float*>(raw_data);
                std::memcpy(dst, src, limit * sizeof(float));
                break;
            }
        }
    }

    void unpack_to_float(float* dst, size_t n) const {
        if (!raw_data || !dst) return;
        size_t limit = (n < count) ? n : count;

        switch (dtype) {
            case DataType::INT8: {
                const int8_t* src = static_cast<const int8_t*>(raw_data);
                for (size_t i = 0; i < limit; ++i) {
                    dst[i] = (static_cast<float>(src[i]) - zero_point) * scale;
                }
                break;
            }
            case DataType::INT16: {
                const int16_t* src = static_cast<const int16_t*>(raw_data);
                for (size_t i = 0; i < limit; ++i) {
                    dst[i] = (static_cast<float>(src[i]) - zero_point) * scale;
                }
                break;
            }
            case DataType::INT32: {
                const int32_t* src = static_cast<const int32_t*>(raw_data);
                for (size_t i = 0; i < limit; ++i) {
                    dst[i] = (static_cast<float>(src[i]) - zero_point) * scale;
                }
                break;
            }
            case DataType::FLOAT16: {
                const uint16_t* src = static_cast<const uint16_t*>(raw_data);
                for (size_t i = 0; i < limit; ++i) {
                    dst[i] = float16_to_float32(src[i]);
                }
                break;
            }
            case DataType::FLOAT32: {
                const float* src = static_cast<const float*>(raw_data);
                std::memcpy(dst, src, limit * sizeof(float));
                break;
            }
        }
    }
};

// Encapsulates raw dynamic memory buffer allocation and deallocation for tensor engine
struct MemoryBuffer {
    float* data;
    size_t size;

    MemoryBuffer(size_t n_elements) : size(n_elements) {
        if (n_elements > 0) {
            data = new float[n_elements]();
        } else {
            data = nullptr;
        }
    }

    ~MemoryBuffer() {
        if (data) {
            delete[] data;
            data = nullptr;
        }
    }

    // Disable copy semantics to enforce strict memory ownership
    MemoryBuffer(const MemoryBuffer&) = delete;
    MemoryBuffer& operator=(const MemoryBuffer&) = delete;

    void fill_zero() {
        if (data && size > 0) {
            std::memset(data, 0, size * sizeof(float));
        }
    }

    void copy_from(const float* src, size_t count) {
        if (data && src && count <= size) {
            std::memcpy(data, src, count * sizeof(float));
        }
    }
};

inline float* allocate_raw(size_t n_elements) {
    return new float[n_elements]();
}

inline void deallocate_raw(float* ptr) {
    delete[] ptr;
}

} // namespace memory
} // namespace engine

#endif // ENGINE_MEMORY_HPP_
