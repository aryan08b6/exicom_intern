#ifndef ENGINE_LAYERS_HPP_
#define ENGINE_LAYERS_HPP_

#include <cstring>
#include <cmath>
#include <cstdlib>
#include <vector>
#include "../array_api/array_api.hpp"
#include "../ops/ops.hpp"

namespace engine {

class Layer {
public:
    virtual ~Layer() {}
    virtual Matrix* forward(Matrix* input, bool training) = 0;
    virtual Matrix* backward(Matrix* dY) = 0;
    virtual void update(float lr) = 0;
};

class LinearLayer : public Layer {
private:
    int in_features, out_features;
    DataType dtype;
    Matrix *W, *b, *dW, *db;
    memory::WeightBuffer* weight_storage;
    Matrix *cached_input;

public:
    LinearLayer(int in_f, int out_f, DataType type = DataType::FLOAT32)
        : in_features(in_f), out_features(out_f), dtype(type) {
        W = new Matrix(in_features, out_features);
        b = new Matrix(1, out_features);
        dW = new Matrix(in_features, out_features);
        db = new Matrix(1, out_features);
        cached_input = nullptr;

        float limit = sqrt(2.0f / in_features);
        for(int i = 0; i < in_features * out_features; ++i) {
            W->data[i] = ((float)rand() / RAND_MAX) * 2 * limit - limit;
        }
        for(int i = 0; i < out_features; ++i) b->data[i] = 0.0f;

        weight_storage = new memory::WeightBuffer(dtype, in_features * out_features, 0.01f, 0);
        weight_storage->pack_from_float(W->data, in_features * out_features);
    }

    ~LinearLayer() override {
        delete W; delete b; delete dW; delete db;
        delete weight_storage;
        if(cached_input) delete cached_input;
    }

    DataType get_dtype() const { return dtype; }

    Matrix* forward(Matrix* input, bool training) override {
        if(cached_input) delete cached_input;
        cached_input = new Matrix(input->rows, input->cols);
        memcpy(cached_input->data, input->data, input->rows * input->cols * sizeof(float));

        if (dtype != DataType::FLOAT32) {
            weight_storage->unpack_to_float(W->data, in_features * out_features);
        }

        Matrix* output = new Matrix(input->rows, out_features);
        matmul(input, W, output);
        add_bias(output, b);
        return output;
    }

    Matrix* backward(Matrix* dZ) override {
        matmul_AT_B(cached_input, dZ, dW);

        for(int j = 0; j < out_features; ++j) {
            float sum = 0.0f;
            for(int i = 0; i < dZ->rows; ++i) sum += dZ->get(i, j);
            db->set(0, j, sum);
        }

        Matrix* dX = new Matrix(cached_input->rows, in_features);
        matmul_A_BT(dZ, W, dX);
        return dX;
    }

    void update(float lr) override {
        for(int i = 0; i < in_features * out_features; ++i) {
            W->data[i] -= lr * dW->data[i];
        }
        for(int i = 0; i < out_features; ++i) {
            b->data[i] -= lr * db->data[i];
        }
        weight_storage->pack_from_float(W->data, in_features * out_features);
    }
};

class BatchNorm1dLayer : public Layer {
private:
    int num_features;
    float eps, momentum;
    Matrix *gamma, *beta, *dgamma, *dbeta;
    Matrix *running_mean, *running_var;
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

    ~BatchNorm1dLayer() override {
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
    ~LeakyReLULayer() override { if(cached_mask) delete cached_mask; }

    Matrix* forward(Matrix* input, bool training) override {
        Matrix* output = new Matrix(input->rows, input->cols);
        if (training) {
            if(cached_mask) delete cached_mask;
            cached_mask = new Matrix(input->rows, input->cols);
        }
        apply_leaky_relu(input, output, training ? cached_mask : nullptr, alpha);
        return output;
    }

    Matrix* backward(Matrix* dZ) override {
        Matrix* dX = new Matrix(dZ->rows, dZ->cols);
        leaky_relu_backward(dZ, cached_mask, dX);
        return dX;
    }

    void update(float lr) override {}
};

class DropoutLayer : public Layer {
private:
    float p;
    Matrix* cached_mask;

public:
    DropoutLayer(float drop_prob = 0.2f) : p(drop_prob), cached_mask(nullptr) {}
    ~DropoutLayer() override { if(cached_mask) delete cached_mask; }

    Matrix* forward(Matrix* input, bool training) override {
        Matrix* output = new Matrix(input->rows, input->cols);
        if (training) {
            if(cached_mask) delete cached_mask;
            cached_mask = new Matrix(input->rows, input->cols);
            apply_dropout(input, output, cached_mask, p);
        } else {
            memcpy(output->data, input->data, input->rows * input->cols * sizeof(float));
        }
        return output;
    }

    Matrix* backward(Matrix* dY) override {
        Matrix* dX = new Matrix(dY->rows, dY->cols);
        dropout_backward(dY, cached_mask, dX);
        return dX;
    }

    void update(float lr) override {}
};

class TanhLayer : public Layer {
private:
    Matrix* cached_output;

public:
    TanhLayer() : cached_output(nullptr) {}
    ~TanhLayer() override { if (cached_output) delete cached_output; }

    Matrix* forward(Matrix* input, bool training) override {
        Matrix* output = new Matrix(input->rows, input->cols);
        apply_tanh(input, output);
        if (training) {
            if (cached_output) delete cached_output;
            cached_output = new Matrix(input->rows, input->cols);
            memcpy(cached_output->data, output->data, input->rows * input->cols * sizeof(float));
        }
        return output;
    }

    Matrix* backward(Matrix* dY) override {
        Matrix* dX = new Matrix(dY->rows, dY->cols);
        tanh_backward(dY, cached_output, dX);
        return dX;
    }

    void update(float lr) override {}
};

class ExpLayer : public Layer {
private:
    Matrix* cached_output;

public:
    ExpLayer() : cached_output(nullptr) {}
    ~ExpLayer() override { if (cached_output) delete cached_output; }

    Matrix* forward(Matrix* input, bool training) override {
        Matrix* output = new Matrix(input->rows, input->cols);
        apply_exp(input, output);
        if (training) {
            if (cached_output) delete cached_output;
            cached_output = new Matrix(input->rows, input->cols);
            memcpy(cached_output->data, output->data, input->rows * input->cols * sizeof(float));
        }
        return output;
    }

    Matrix* backward(Matrix* dY) override {
        Matrix* dX = new Matrix(dY->rows, dY->cols);
        exp_backward(dY, cached_output, dX);
        return dX;
    }

    void update(float lr) override {}
};

class SoftmaxLayer : public Layer {
private:
    Matrix* cached_output;

public:
    SoftmaxLayer() : cached_output(nullptr) {}
    ~SoftmaxLayer() override { if (cached_output) delete cached_output; }

    Matrix* forward(Matrix* input, bool training) override {
        Matrix* output = new Matrix(input->rows, input->cols);
        apply_softmax(input, output);
        if (training) {
            if (cached_output) delete cached_output;
            cached_output = new Matrix(input->rows, input->cols);
            memcpy(cached_output->data, output->data, input->rows * input->cols * sizeof(float));
        }
        return output;
    }

    Matrix* backward(Matrix* dY) override {
        Matrix* dX = new Matrix(dY->rows, dY->cols);
        softmax_backward(dY, cached_output, dX);
        return dX;
    }

    void update(float lr) override {}
};

// Sequential container layer (Needle framework style)
class SequentialLayer : public Layer {
private:
    std::vector<Layer*> layers;

public:
    SequentialLayer() {}
    ~SequentialLayer() override {
        for (size_t i = 0; i < layers.size(); ++i) {
            delete layers[i];
        }
    }

    void add(Layer* layer) {
        layers.push_back(layer);
    }

    size_t size() const { return layers.size(); }

    Matrix* forward(Matrix* X, bool training) override {
        Matrix* current_out = X;
        for (size_t i = 0; i < layers.size(); ++i) {
            Matrix* next_out = layers[i]->forward(current_out, training);
            if (i > 0) delete current_out;
            current_out = next_out;
        }
        return current_out;
    }

    Matrix* backward(Matrix* dY) override {
        Matrix* current_grad = dY;
        for (int i = static_cast<int>(layers.size()) - 1; i >= 0; --i) {
            Matrix* next_grad = layers[i]->backward(current_grad);
            if (i < static_cast<int>(layers.size()) - 1) delete current_grad;
            current_grad = next_grad;
        }
        delete current_grad;
        return nullptr;
    }

    void update(float lr) override {
        for (size_t i = 0; i < layers.size(); ++i) {
            layers[i]->update(lr);
        }
    }
};

} // namespace engine

#endif // ENGINE_LAYERS_HPP_
