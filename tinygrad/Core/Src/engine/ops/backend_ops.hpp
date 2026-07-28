#ifndef ENGINE_OPS_BACKEND_OPS_HPP_
#define ENGINE_OPS_BACKEND_OPS_HPP_

#include <cmath>
#include <cstdlib>
#include "../array_api/array_api.hpp"

// Enable CMSIS-NN ARM Cortex-M hardware acceleration backend
#ifndef USE_CMSIS_NN
#define USE_CMSIS_NN 1
#endif

#ifdef USE_CMSIS_NN
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"
#endif

namespace engine {
namespace backend {

// Dynamic toggle for CMSIS-NN hardware acceleration
inline bool& get_cmsis_nn_enabled_ref() {
    static bool enabled = true;
    return enabled;
}

inline void set_cmsis_nn_enabled(bool enabled) {
    get_cmsis_nn_enabled_ref() = enabled;
}

inline bool is_cmsis_nn_enabled() {
    return get_cmsis_nn_enabled_ref();
}

// ----------------------------------------------------------------------------
// Linear Algebra Kernels (MatMul, Batch MatMul, Bias Add)
// ----------------------------------------------------------------------------

inline void matmul(const Matrix* A, const Matrix* B, Matrix* C) {
#ifdef USE_CMSIS_NN
    if (is_cmsis_nn_enabled()) {
        int M = A->rows;
        int K = A->cols;
        int N = B->cols;
        float* BT = static_cast<float*>(std::malloc(N * K * sizeof(float)));
        if (BT) {
            for (int r = 0; r < K; ++r) {
                for (int c = 0; c < N; ++c) {
                    BT[c * K + r] = B->get(r, c);
                }
            }
            arm_nn_mat_mult_nt_t_f32(A->data, BT, nullptr, C->data, M, N, K, N, -1e30f, 1e30f);
            std::free(BT);
            return;
        }
    }
#endif
    for (int i = 0; i < A->rows; ++i) {
        for (int j = 0; j < B->cols; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < A->cols; ++k) {
                sum += A->get(i, k) * B->get(k, j);
            }
            C->set(i, j, sum);
        }
    }
}

inline void matmul_AT_B(const Matrix* A, const Matrix* B, Matrix* C) {
#ifdef USE_CMSIS_NN
    if (is_cmsis_nn_enabled()) {
        int K = A->rows;
        int M = A->cols;
        int N = B->cols;
        float* AT = static_cast<float*>(std::malloc(M * K * sizeof(float)));
        float* BT = static_cast<float*>(std::malloc(N * K * sizeof(float)));
        if (AT && BT) {
            for (int r = 0; r < K; ++r) {
                for (int c = 0; c < M; ++c) {
                    AT[c * K + r] = A->get(r, c);
                }
            }
            for (int r = 0; r < K; ++r) {
                for (int c = 0; c < N; ++c) {
                    BT[c * K + r] = B->get(r, c);
                }
            }
            arm_nn_mat_mult_nt_t_f32(AT, BT, nullptr, C->data, M, N, K, N, -1e30f, 1e30f);
            std::free(AT);
            std::free(BT);
            return;
        }
        if (AT) std::free(AT);
        if (BT) std::free(BT);
    }
#endif
    for (int i = 0; i < A->cols; ++i) {
        for (int j = 0; j < B->cols; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < A->rows; ++k) {
                sum += A->get(k, i) * B->get(k, j);
            }
            C->set(i, j, sum);
        }
    }
}

inline void matmul_A_BT(const Matrix* A, const Matrix* B, Matrix* C) {
#ifdef USE_CMSIS_NN
    if (is_cmsis_nn_enabled()) {
        int M = A->rows;
        int K = A->cols;
        int N = B->rows;
        arm_nn_mat_mult_nt_t_f32(A->data, B->data, nullptr, C->data, M, N, K, N, -1e30f, 1e30f);
        return;
    }
#endif
    for (int i = 0; i < A->rows; ++i) {
        for (int j = 0; j < B->rows; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < A->cols; ++k) {
                sum += A->get(i, k) * B->get(j, k);
            }
            C->set(i, j, sum);
        }
    }
}

inline void add_bias(Matrix* A, const Matrix* b) {
#ifdef USE_CMSIS_NN
    if (is_cmsis_nn_enabled()) {
        for (int i = 0; i < A->rows; ++i) {
            arm_elementwise_add_f32(A->data + i * A->cols, b->data, A->data + i * A->cols, -1e30f, 1e30f, A->cols);
        }
        return;
    }
#endif
    for (int i = 0; i < A->rows; ++i) {
        for (int j = 0; j < A->cols; ++j) {
            A->data[i * A->cols + j] += b->data[j];
        }
    }
}

// ----------------------------------------------------------------------------
// Activation & Regularization Kernels (ReLU, LeakyReLU, Dropout)
// ----------------------------------------------------------------------------

inline void apply_leaky_relu(const Matrix* input, Matrix* output, Matrix* mask, float alpha) {
    int total = input->rows * input->cols;
    for (int i = 0; i < total; ++i) {
        float val = input->data[i];
        if (val > 0.0f) {
            output->data[i] = val;
            if (mask) mask->data[i] = 1.0f;
        } else {
            output->data[i] = alpha * val;
            if (mask) mask->data[i] = alpha;
        }
    }
}

inline void leaky_relu_backward(const Matrix* dZ, const Matrix* mask, Matrix* dX) {
    int total = dZ->rows * dZ->cols;
    for (int i = 0; i < total; ++i) {
        dX->data[i] = dZ->data[i] * mask->data[i];
    }
}

inline void apply_dropout(const Matrix* input, Matrix* output, Matrix* mask, float p) {
    int total = input->rows * input->cols;
    float scale = (p < 1.0f) ? (1.0f / (1.0f - p)) : 1.0f;
    for (int i = 0; i < total; ++i) {
        float r = (float)rand() / (float)RAND_MAX;
        if (r >= p) {
            if (mask) mask->data[i] = scale;
            output->data[i] = input->data[i] * scale;
        } else {
            if (mask) mask->data[i] = 0.0f;
            output->data[i] = 0.0f;
        }
    }
}

inline void dropout_backward(const Matrix* dY, const Matrix* mask, Matrix* dX) {
    int total = dY->rows * dY->cols;
    for (int i = 0; i < total; ++i) {
        dX->data[i] = dY->data[i] * mask->data[i];
    }
}

// ----------------------------------------------------------------------------
// Tanh, Exp, and Softmax Kernels
// ----------------------------------------------------------------------------

inline void apply_tanh(const Matrix* input, Matrix* output) {
    int total = input->rows * input->cols;
    for (int i = 0; i < total; ++i) {
        output->data[i] = std::tanh(input->data[i]);
    }
}

inline void tanh_backward(const Matrix* dY, const Matrix* cached_output, Matrix* dX) {
    int total = dY->rows * dY->cols;
    for (int i = 0; i < total; ++i) {
        float y = cached_output->data[i];
        dX->data[i] = dY->data[i] * (1.0f - y * y);
    }
}

inline void apply_exp(const Matrix* input, Matrix* output) {
    int total = input->rows * input->cols;
    for (int i = 0; i < total; ++i) {
        output->data[i] = std::exp(input->data[i]);
    }
}

inline void exp_backward(const Matrix* dY, const Matrix* cached_output, Matrix* dX) {
    int total = dY->rows * dY->cols;
    for (int i = 0; i < total; ++i) {
        dX->data[i] = dY->data[i] * cached_output->data[i];
    }
}

inline void apply_softmax(const Matrix* input, Matrix* output) {
    int rows = input->rows;
    int cols = input->cols;
    for (int r = 0; r < rows; ++r) {
        float max_val = input->get(r, 0);
        for (int c = 1; c < cols; ++c) {
            float val = input->get(r, c);
            if (val > max_val) max_val = val;
        }
        float sum = 0.0f;
        for (int c = 0; c < cols; ++c) {
            float exp_val = std::exp(input->get(r, c) - max_val);
            output->set(r, c, exp_val);
            sum += exp_val;
        }
        float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 1.0f;
        for (int c = 0; c < cols; ++c) {
            output->set(r, c, output->get(r, c) * inv_sum);
        }
    }
}

inline void softmax_backward(const Matrix* dY, const Matrix* cached_output, Matrix* dX) {
    int rows = dY->rows;
    int cols = dY->cols;
    for (int r = 0; r < rows; ++r) {
        float dot = 0.0f;
        for (int c = 0; c < cols; ++c) {
            dot += dY->get(r, c) * cached_output->get(r, c);
        }
        for (int c = 0; c < cols; ++c) {
            float y = cached_output->get(r, c);
            float grad = dY->get(r, c);
            dX->set(r, c, y * (grad - dot));
        }
    }
}

} // namespace backend
} // namespace engine

#endif // ENGINE_OPS_BACKEND_OPS_HPP_
