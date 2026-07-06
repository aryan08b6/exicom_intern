/*
 * c_bridge.cpp
 *
 *  Created on: Jul 6, 2026
 *      Author: aryan
 */




#include "c_bridge.h"
#include "layers.hpp"
// Assume the comprehensive C++ classes defined previously are included here

extern "C" {

PredictorHandle create_predictor() {
    return new AdvancedPowerPredictor();
}

void destroy_predictor(PredictorHandle handle) {
    if (handle) {
        delete static_cast<AdvancedPowerPredictor*>(handle);
    }
}

float train_step(PredictorHandle handle, float* inputs, float* targets, int batch_size, float lr) {
    AdvancedPowerPredictor* model = static_cast<AdvancedPowerPredictor*>(handle);

    // Wrap standard C arrays in the engine's Matrix struct format
    Matrix X(batch_size, 5);
    memcpy(X.data, inputs, batch_size * 5 * sizeof(float));

    // Execute Forward Pass
    Matrix* Y_pred = model->forward(&X, true);

    // Compute MSE Loss and construct the terminal gradient vector
    float loss = 0.0f;
    Matrix dY(batch_size, 1);

    for (int i = 0; i < batch_size; ++i) {
        float diff = Y_pred->get(i, 0) - targets[i];
        loss += diff * diff;
        dY.set(i, 0, (2.0f / batch_size) * diff);
    }
    loss /= batch_size;

    // Execute Backward Pass and Optimizer Update
    model->backward(&dY);
    model->update(lr);

    delete Y_pred;
    return loss;
}

float predict(PredictorHandle handle, float* input) {
    AdvancedPowerPredictor* model = static_cast<AdvancedPowerPredictor*>(handle);

    Matrix X(1, 5);
    memcpy(X.data, input, 5 * sizeof(float));

    // Execute Forward Pass (Inference mode disabled to utilize running stats)
    Matrix* Y_pred = model->forward(&X, false);
    float prediction = Y_pred->get(0, 0);

    delete Y_pred;
    return prediction;
}

} // extern "C"
