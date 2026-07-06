/*
 * c_bridge.h
 *
 *  Created on: Jul 6, 2026
 *      Author: aryan
 */


#ifndef SRC_ENGINE_C_BRIDGE_H_
#define SRC_ENGINE_C_BRIDGE_H_

#ifdef __cplusplus
extern "C" {
#endif

// Opaque pointer designed to obscure C++ classes from the C compiler
typedef void* PredictorHandle;

PredictorHandle create_predictor();
void destroy_predictor(PredictorHandle handle);

// Executes one comprehensive training pass and returns the Loss
float train_step(PredictorHandle handle, float* inputs, float* targets, int batch_size, float lr);

// Executes deployment inference on a single 5-dimensional sample
float predict(PredictorHandle handle, float* input);

#ifdef __cplusplus
}
#endif


#endif /* SRC_ENGINE_C_BRIDGE_H_ */
