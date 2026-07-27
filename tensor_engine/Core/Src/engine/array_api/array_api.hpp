#ifndef ENGINE_ARRAY_API_HPP_
#define ENGINE_ARRAY_API_HPP_

#include "../memory/memory.hpp"

namespace engine {

// Lightweight 2D array container (Array API) backed by MemoryBuffer
struct Matrix {
    int rows;
    int cols;
    memory::MemoryBuffer buffer;
    float* data; // Convenience pointer to memory buffer data

    Matrix(int r, int c) : rows(r), cols(c), buffer(r * c) {
        data = buffer.data;
    }

    ~Matrix() {
        // MemoryBuffer manages deallocation automatically
    }

    Matrix(const Matrix&) = delete;
    Matrix& operator=(const Matrix&) = delete;

    inline float get(int r, int c) const { return data[r * cols + c]; }
    inline void set(int r, int c, float val) { data[r * cols + c] = val; }

    inline int size() const { return rows * cols; }

    void zero() {
        buffer.fill_zero();
    }

    void copy_from(const float* src, size_t count) {
        buffer.copy_from(src, count);
    }
};

// Typedefs for framework compatibility
typedef Matrix NDArray;
typedef Matrix Tensor;

} // namespace engine

#endif // ENGINE_ARRAY_API_HPP_
