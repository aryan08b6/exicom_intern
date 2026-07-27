#ifndef ENGINE_BRIDGE_C_BRIDGE_H_
#define ENGINE_BRIDGE_C_BRIDGE_H_

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handles designed to obscure C++ classes from the C compiler
typedef void* LayerHandle;
typedef void* PredictorHandle;

// Predictor / Sequential Model container management
PredictorHandle create_predictor(void);
void destroy_predictor(PredictorHandle handle);
void predictor_add_layer(PredictorHandle handle, LayerHandle layer);

// Layer creation functions
LayerHandle create_linear_layer(int in_features, int out_features);
LayerHandle create_linear_layer_with_dtype(int in_features, int out_features, int dtype_code);
LayerHandle create_batchnorm1d_layer(int num_features, float eps, float momentum);
LayerHandle create_leaky_relu_layer(float alpha);
LayerHandle create_dropout_layer(float p);
void destroy_layer(LayerHandle layer);

// Executes one comprehensive training pass and returns Loss
float train_step(PredictorHandle handle, float* inputs, float* targets, int batch_size, float lr);

// Executes deployment inference on a single 5-dimensional sample
float predict(PredictorHandle handle, float* input);

// Controls runtime toggle of CMSIS-NN hardware acceleration (1 = enabled, 0 = disabled CPU fallback)
void set_cmsis_nn_acceleration_enabled(int enabled);

#ifdef __cplusplus
}
#endif

#endif /* ENGINE_BRIDGE_C_BRIDGE_H_ */
