#include "c_bridge.h"
#include "../layers/layers.hpp"

using namespace engine;

extern "C" {

PredictorHandle create_predictor(void) {
    return new SequentialLayer();
}

void destroy_predictor(PredictorHandle handle) {
    if (handle) {
        delete static_cast<SequentialLayer*>(handle);
    }
}

void predictor_add_layer(PredictorHandle handle, LayerHandle layer) {
    if (handle && layer) {
        static_cast<SequentialLayer*>(handle)->add(static_cast<Layer*>(layer));
    }
}

LayerHandle create_linear_layer(int in_features, int out_features) {
    return new LinearLayer(in_features, out_features, DataType::FLOAT32);
}

LayerHandle create_linear_layer_with_dtype(int in_features, int out_features, int dtype_code) {
    DataType type = DataType::FLOAT32;
    switch (dtype_code) {
        case 0: type = DataType::INT8; break;
        case 1: type = DataType::INT16; break;
        case 2: type = DataType::INT32; break;
        case 3: type = DataType::FLOAT16; break;
        case 4: type = DataType::FLOAT32; break;
        default: type = DataType::FLOAT32; break;
    }
    return new LinearLayer(in_features, out_features, type);
}

LayerHandle create_batchnorm1d_layer(int num_features, float eps, float momentum) {
    return new BatchNorm1dLayer(num_features, eps, momentum);
}

LayerHandle create_leaky_relu_layer(float alpha) {
    return new LeakyReLULayer(alpha);
}

LayerHandle create_dropout_layer(float p) {
    return new DropoutLayer(p);
}

LayerHandle create_tanh_layer(void) {
    return new TanhLayer();
}

LayerHandle create_exp_layer(void) {
    return new ExpLayer();
}

LayerHandle create_softmax_layer(void) {
    return new SoftmaxLayer();
}

void destroy_layer(LayerHandle layer) {
    if (layer) {
        delete static_cast<Layer*>(layer);
    }
}

float train_step(PredictorHandle handle, float* inputs, float* targets, int batch_size, float lr) {
    SequentialLayer* model = static_cast<SequentialLayer*>(handle);

    Matrix X(batch_size, 5);
    X.copy_from(inputs, batch_size * 5);

    Matrix* Y_pred = model->forward(&X, true);

    float loss = 0.0f;
    Matrix dY(batch_size, 1);

    for (int i = 0; i < batch_size; ++i) {
        float diff = Y_pred->get(i, 0) - targets[i];
        loss += diff * diff;
        dY.set(i, 0, (2.0f / batch_size) * diff);
    }
    loss /= batch_size;

    model->backward(&dY);
    model->update(lr);

    delete Y_pred;
    return loss;
}

float predict(PredictorHandle handle, float* input) {
    SequentialLayer* model = static_cast<SequentialLayer*>(handle);

    Matrix X(1, 5);
    X.copy_from(input, 5);

    Matrix* Y_pred = model->forward(&X, false);
    float prediction = Y_pred->get(0, 0);

    delete Y_pred;
    return prediction;
}

void set_cmsis_nn_acceleration_enabled(int enabled) {
    backend::set_cmsis_nn_enabled(enabled != 0);
}

} // extern "C"
