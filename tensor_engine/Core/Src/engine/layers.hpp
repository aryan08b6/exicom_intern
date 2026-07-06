/*
 * layers.hpp
 *
 *  Created on: Jul 6, 2026
 *      Author: aryan
 */

#include "matrix_mul.hpp"

#ifndef SRC_ENGINE_LAYERS_HPP_
#define SRC_ENGINE_LAYERS_HPP_

class Layer {
public:
    virtual ~Layer() {}
    // The training flag indicates whether to cache values for backprop or use inference mode
    virtual Matrix* forward(Matrix* input, bool training) = 0;
    virtual Matrix* backward(Matrix* dY) = 0;
    virtual void update(float lr) = 0;
};

class LinearLayer : public Layer {
private:
    int in_features, out_features;
    Matrix *W, *b, *dW, *db;
    Matrix *cached_input;

public:
    LinearLayer(int in_f, int out_f) : in_features(in_f), out_features(out_f) {
        W = new Matrix(in_features, out_features);
        b = new Matrix(1, out_features);
        dW = new Matrix(in_features, out_features);
        db = new Matrix(1, out_features);
        cached_input = nullptr;

        // Initialization heuristic optimized for LeakyReLU
        float limit = sqrt(2.0f / in_features);
        for(int i = 0; i < in_features * out_features; ++i) {
            W->data[i] = ((float)rand() / RAND_MAX) * 2 * limit - limit;
        }
        for(int i = 0; i < out_features; ++i) b->data[i] = 0.0f;
    }

    ~LinearLayer() {
        delete W; delete b; delete dW; delete db;
        if(cached_input) delete cached_input;
    }

    Matrix* forward(Matrix* input, bool training) override {
        if(cached_input) delete cached_input;

        // Deep copy the input matrix required for dW computation later
        cached_input = new Matrix(input->rows, input->cols);
        memcpy(cached_input->data, input->data, input->rows * input->cols * sizeof(float));

        Matrix* output = new Matrix(input->rows, out_features);
        matmul(input, W, output);

        // Affine bias addition
        for(int i = 0; i < output->rows; ++i) {
            for(int j = 0; j < output->cols; ++j) {
                output->set(i, j, output->get(i, j) + b->get(0, j));
            }
        }
        return output;
    }

    Matrix* backward(Matrix* dZ) override {
        // Gradient wrt Weights: dW = X^T * dZ
        matmul_AT_B(cached_input, dZ, dW);

        // Gradient wrt Biases: db = sum(dZ, axis=0)
        for(int j = 0; j < out_features; ++j) {
            float sum = 0;
            for(int i = 0; i < dZ->rows; ++i) sum += dZ->get(i, j);
            db->set(0, j, sum);
        }

        // Gradient wrt Input: dX = dZ * W^T
        Matrix* dX = new Matrix(cached_input->rows, in_features);
        matmul_A_BT(dZ, W, dX);
        return dX;
    }

    void update(float lr) override {
        // Standard Stochastic Gradient Descent (SGD) parameter update
        for(int i = 0; i < in_features * out_features; ++i) {
            W->data[i] -= lr * dW->data[i];
        }
        for(int i = 0; i < out_features; ++i) {
            b->data[i] -= lr * db->data[i];
        }
    }
};

class BatchNorm1dLayer : public Layer {
private:
    int num_features;
    float eps, momentum;
    Matrix *gamma, *beta, *dgamma, *dbeta;
    Matrix *running_mean, *running_var;

    // Gradient computation caches
    Matrix *cached_x_hat, *cached_var;
    int last_batch_size;

public:
    BatchNorm1dLayer(int features, float epsilon = 1e-5, float mom = 0.1f)
        : num_features(features), eps(epsilon), momentum(mom) {
        gamma = new Matrix(1, features);
        beta = new Matrix(1, features);
        dgamma = new Matrix(1, features);
        dbeta = new Matrix(1, features);
        running_mean = new Matrix(1, features);
        running_var = new Matrix(1, features);

        cached_x_hat = nullptr;
        cached_var = nullptr;

        for(int i = 0; i < features; ++i) {
            gamma->data[i] = 1.0f;
            beta->data[i] = 0.0f;
            running_mean->data[i] = 0.0f;
            running_var->data[i] = 1.0f;
        }
    }

    ~BatchNorm1dLayer() {
        delete gamma; delete beta; delete dgamma; delete dbeta;
        delete running_mean; delete running_var;
        if(cached_x_hat) delete cached_x_hat;
        if(cached_var) delete cached_var;
    }

    Matrix* forward(Matrix* input, bool training) override {
        int N = input->rows;
        last_batch_size = N;
        Matrix* output = new Matrix(N, num_features);

        if (training) {
            if(cached_x_hat) delete cached_x_hat;
            if(cached_var) delete cached_var;
            cached_x_hat = new Matrix(N, num_features);
            cached_var = new Matrix(1, num_features);

            for (int j = 0; j < num_features; ++j) {
                float mean = 0.0f;
                for (int i = 0; i < N; ++i) mean += input->get(i, j);
                mean /= N;

                float var = 0.0f;
                for (int i = 0; i < N; ++i) {
                    float diff = input->get(i, j) - mean;
                    var += diff * diff;
                }
                var /= N;
                cached_var->set(0, j, var);

                // Exponential moving average for inference
                running_mean->set(0, j, (1 - momentum) * running_mean->get(0, j) + momentum * mean);
                running_var->set(0, j, (1 - momentum) * running_var->get(0, j) + momentum * var);

                float inv_std = 1.0f / sqrt(var + eps);
                for (int i = 0; i < N; ++i) {
                    float x_hat = (input->get(i, j) - mean) * inv_std;
                    cached_x_hat->set(i, j, x_hat);
                    output->set(i, j, gamma->get(0, j) * x_hat + beta->get(0, j));
                }
            }
        } else {
            // Edge Deployment Inference Mode utilizing running statistics
            for (int j = 0; j < num_features; ++j) {
                float mean = running_mean->get(0, j);
                float var = running_var->get(0, j);
                float inv_std = 1.0f / sqrt(var + eps);
                for (int i = 0; i < N; ++i) {
                    float x_hat = (input->get(i, j) - mean) * inv_std;
                    output->set(i, j, gamma->get(0, j) * x_hat + beta->get(0, j));
                }
            }
        }
        return output;
    }

    Matrix* backward(Matrix* dY) override {
        int N = last_batch_size;
        Matrix* dX = new Matrix(N, num_features);

        for (int j = 0; j < num_features; ++j) {
            float dgamma_val = 0.0f;
            float dbeta_val = 0.0f;
            float sum_dXhat = 0.0f;
            float sum_dXhat_Xhat = 0.0f;

            for (int i = 0; i < N; ++i) {
                float dy_ij = dY->get(i, j);
                float x_hat_ij = cached_x_hat->get(i, j);

                dgamma_val += dy_ij * x_hat_ij;
                dbeta_val += dy_ij;

                float dXhat_ij = dy_ij * gamma->get(0, j);
                sum_dXhat += dXhat_ij;
                sum_dXhat_Xhat += dXhat_ij * x_hat_ij;
            }

            dgamma->set(0, j, dgamma_val);
            dbeta->set(0, j, dbeta_val);

            // Applying the optimized, mathematically stabilized dX algebraic derivation
            float inv_std = 1.0f / sqrt(cached_var->get(0, j) + eps);
            for (int i = 0; i < N; ++i) {
                float dXhat_ij = dY->get(i, j) * gamma->get(0, j);
                float x_hat_ij = cached_x_hat->get(i, j);

                float dx_ij = (inv_std / N) * (N * dXhat_ij - sum_dXhat - x_hat_ij * sum_dXhat_Xhat);
                dX->set(i, j, dx_ij);
            }
        }
        return dX;
    }

    void update(float lr) override {
        for(int j = 0; j < num_features; ++j) {
            gamma->data[j] -= lr * dgamma->data[j];
            beta->data[j] -= lr * dbeta->data[j];
        }
    }
};

class LeakyReLULayer : public Layer {
private:
    float alpha;
    Matrix* cached_mask;

public:
    LeakyReLULayer(float a = 0.1f) : alpha(a), cached_mask(nullptr) {}
    ~LeakyReLULayer() { if(cached_mask) delete cached_mask; }

    Matrix* forward(Matrix* input, bool training) override {
        Matrix* output = new Matrix(input->rows, input->cols);

        if (training) {
            if(cached_mask) delete cached_mask;
            cached_mask = new Matrix(input->rows, input->cols);
        }

        for(int i = 0; i < input->rows * input->cols; ++i) {
            float val = input->data[i];
            if (val > 0) {
                output->data[i] = val;
                if (training) cached_mask->data[i] = 1.0f;
            } else {
                output->data[i] = alpha * val;
                if (training) cached_mask->data[i] = alpha;
            }
        }
        return output;
    }

    Matrix* backward(Matrix* dZ) override {
        Matrix* dX = new Matrix(dZ->rows, dZ->cols);
        for(int i = 0; i < dZ->rows * dZ->cols; ++i) {
            dX->data[i] = dZ->data[i] * cached_mask->data[i]; // Element-wise masking
        }
        return dX;
    }

    void update(float lr) override { /* No learnable parameters present */ }
};


class DropoutLayer : public Layer {
private:
    float p;
    Matrix* cached_mask;

public:
    DropoutLayer(float drop_prob = 0.2f) : p(drop_prob), cached_mask(nullptr) {}
    ~DropoutLayer() { if(cached_mask) delete cached_mask; }

    Matrix* forward(Matrix* input, bool training) override {
        Matrix* output = new Matrix(input->rows, input->cols);

        if (training) {
            if(cached_mask) delete cached_mask;
            cached_mask = new Matrix(input->rows, input->cols);

            float scale = 1.0f / (1.0f - p);
            for(int i = 0; i < input->rows * input->cols; ++i) {
                float rand_val = (float)rand() / RAND_MAX;
                if (rand_val >= p) {
                    cached_mask->data[i] = scale;
                    output->data[i] = input->data[i] * scale;
                } else {
                    cached_mask->data[i] = 0.0f;
                    output->data[i] = 0.0f;
                }
            }
        } else {
            // Identity function mechanism deployed during MCU inference
            memcpy(output->data, input->data, input->rows * input->cols * sizeof(float));
        }
        return output;
    }

    Matrix* backward(Matrix* dY) override {
        Matrix* dX = new Matrix(dY->rows, dY->cols);
        for(int i = 0; i < dY->rows * dY->cols; ++i) {
            dX->data[i] = dY->data[i] * cached_mask->data[i];
        }
        return dX;
    }

    void update(float lr) override { /* No learnable parameters present */ }
};


class AdvancedPowerPredictor {
private:
    Layer* layers[9];
    int num_layers;

public:
    AdvancedPowerPredictor() {
        num_layers = 9;
        layers[0] = new LinearLayer(5, 32);
        layers[1] = new BatchNorm1dLayer(32);
        layers[2] = new LeakyReLULayer(0.1f);
        layers[3] = new DropoutLayer(0.2f);

        layers[4] = new LinearLayer(32, 16);
        layers[5] = new BatchNorm1dLayer(16);
        layers[6] = new LeakyReLULayer(0.1f);
        layers[7] = new DropoutLayer(0.2f);

        layers[8] = new LinearLayer(16, 1);
    }

    ~AdvancedPowerPredictor() {
        for(int i = 0; i < num_layers; ++i) delete layers[i];
    }

    Matrix* forward(Matrix* X, bool training) {
        Matrix* current_out = X;
        for(int i = 0; i < num_layers; ++i) {
            Matrix* next_out = layers[i]->forward(current_out, training);
            // Systematically free intermediate matrices to prevent hard-faults
            if (i > 0) delete current_out;
            current_out = next_out;
        }
        return current_out;
    }

    void backward(Matrix* dY) {
        Matrix* current_grad = dY;
        for(int i = num_layers - 1; i >= 0; --i) {
            Matrix* next_grad = layers[i]->backward(current_grad);
            if (i < num_layers - 1) delete current_grad;
            current_grad = next_grad;
        }
        delete current_grad;
    }

    void update(float lr) {
        for(int i = 0; i < num_layers; ++i) layers[i]->update(lr);
    }
};

#endif /* SRC_ENGINE_LAYERS_HPP_ */
