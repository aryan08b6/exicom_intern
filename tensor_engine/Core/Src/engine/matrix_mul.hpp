/*
 * matrix_mul.hpp
 *
 *  Created on: Jul 6, 2026
 *      Author: aryan
 */

#ifndef SRC_ENGINE_MATRIX_MUL_HPP_
#define SRC_ENGINE_MATRIX_MUL_HPP_

// engine.hpp
#pragma once
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <random>

// Lightweight 2D Matrix structure utilizing a flattened 1D array
struct Matrix {
    int rows;
    int cols;
    float* data;

    Matrix(int r, int c) : rows(r), cols(c) {
        data = new float[r * c]();
    }

    ~Matrix() {
        delete[] data;
    }

    // Explicitly delete copy constructor to prevent double-free hard faults
    Matrix(const Matrix&) = delete;
    Matrix& operator=(const Matrix&) = delete;

    inline float get(int r, int c) const { return data[r * cols + c]; }
    inline void set(int r, int c, float val) { data[r * cols + c] = val; }
};

// Standard Matrix Multiplication: C = A * B (Forward Pass)
void matmul(const Matrix* A, const Matrix* B, Matrix* C) {
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

// Transpose A Matrix Multiplication: C = A^T * B (Used for dW computation)
void matmul_AT_B(const Matrix* A, const Matrix* B, Matrix* C) {
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

// Transpose B Matrix Multiplication: C = A * B^T (Used for dX computation)
void matmul_A_BT(const Matrix* A, const Matrix* B, Matrix* C) {
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


#endif /* SRC_ENGINE_MATRIX_MUL_HPP_ */
