# STM32 Edge Tensor Engine

A lightweight, PyTorch / Needle-style Deep Learning Tensor Engine written in C++ and optimized for ARM Cortex-M microcontrollers (STM32F429ZI). Supports **on-device training**, **inference**, **multi-datatype weight storage (`INT8`, `INT16`, `INT32`, `FLOAT16`, `FLOAT32`)**, and **ARM CMSIS-NN SIMD/DSP hardware acceleration**.

---

## 🌟 Key Features

- **On-Device Training & Inference**: Complete forward pass, backpropagation, parameter updates, and evaluation directly on MCU hardware.
- **ARM CMSIS-NN Hardware Acceleration**: Dispatches matrix multiplication and vector operations to ARM Cortex-M DSP assembly routines (`arm_nn_mat_mult_nt_t_f32`, `arm_elementwise_add_f32`).
- **Multi-Datatype Weight Storage**:
  - `INT8`, `INT16`, `INT32`: Quantized storage with scale and zero-point packing.
  - `FLOAT16`: IEEE 754 half-precision bit conversions (`float32_to_float16` and `float16_to_float32`).
  - `FLOAT32`: Full single-precision floating point.
- **Layer-Based Modular Autograd**: Pre-allocates cached tensors (`cached_input`, `cached_mask`, `cached_var`) to guarantee zero dynamic heap allocation during training epochs.
- **C-Bridge Architecture**: Clean C-API wrapper (`c_bridge.h`) allowing C++ neural network modules to run seamlessly in standard STM32CubeIDE C applications (`main.c`).

---

## 🏗️ System Architecture & Design Choices

```
                                 +-----------------------------------+
                                 |         main.c (C Application)     |
                                 +-----------------------------------+
                                                   |
                                     (Opaque Handles & C-API)
                                                   v
                                 +-----------------------------------+
                                 |       c_bridge.h / c_bridge.cpp   |
                                 +-----------------------------------+
                                                   |
                                                   v
                                 +-----------------------------------+
                                 |    layers.hpp (Sequential,      |
                                 |    Linear, BatchNorm, ReLU)       |
                                 +-----------------------------------+
                                                   |
                                                   v
                                 +-----------------------------------+
                                 |    backend_ops.hpp (Backend Hub)  |
                                 +-----------------------------------+
                                      /                         \
                                     v                           v
                      +-------------------------+   +-------------------------+
                      | CPU Reference Kernels   |   | ARM CMSIS-NN SIMD/DSP   |
                      | (Standard C++ loops)    |   | (Cortex-M Hardware)     |
                      +-------------------------+   +-------------------------+
```

### 1. Memory Management & Multi-Precision Storage (`memory.hpp`)
- **`MemoryBuffer`**: Manages raw contiguous float arrays for activation buffers.
- **`Affine Uniform Quantization`**: Allows for efficient memory usage 

- **`WeightBuffer`**: Stores layer parameters in specified target data types (`INT8`, `INT16`, `INT32`, `FLOAT16`, `FLOAT32`).
  - **Quantization Formula**:
    $$q = \text{clamp}\left(\text{round}\left(\frac{w}{\text{scale}}\right) + \text{zero\_point}\right)$$
  - **Dequantization Formula**:
    $$w = (q - \text{zero\_point}) \times \text{scale}$$

### 2. Primary Array Container (`array_api.hpp`)
- **`Matrix` / `Tensor`**: Lightweight 2D container storing dimensions (`rows`, `cols`), memory ownership (`MemoryBuffer`), and raw pointer access (`data`).

### 3. Backend Operations Hub (`backend_ops.hpp`)
- Centralized dispatch hub for all matrix operations (`matmul`, `matmul_AT_B`, `matmul_A_BT`, `add_bias`, `apply_leaky_relu`, `apply_dropout`).
- **Dynamic CMSIS-NN Toggle**: Runtime flag (`set_cmsis_nn_acceleration_enabled`) to switch between CPU reference implementations and hardware-accelerated CMSIS-NN SIMD routines.

### 4. Neural Network Layer Hierarchy (`layers.hpp`)
- Abstract base class `Layer` with `forward()`, `backward()`, and `update()` methods.
- **Concrete Layers**:
  - `LinearLayer`: Fully connected linear transformation ($Y = X \cdot W + b$). Supports setting weight data type per layer.
  - `BatchNorm1dLayer`: Batch normalization ($Y = \gamma \hat{X} + \beta$).
  - `LeakyReLULayer`: Leaky ReLU activation ($Y = \max(\alpha X, X)$).
  - `DropoutLayer`: Inverted dropout regularization.
  - `SequentialLayer`: Container managing sequential forward and reverse-order backward passes.

---

## 📁 Repository Structure

```
tensor_engine/
├── Core/
│   ├── Inc/
│   │   └── main.h
│   └── Src/
│       ├── main.c                          # Entry point & dual-pass benchmark suite
│       └── engine/
│           ├── memory/memory.hpp           # MemoryBuffer, WeightBuffer, Float16 & Quantization
│           ├── array_api/array_api.hpp     # Matrix / Tensor container definitions
│           ├── ops/
│           │   ├── ops.hpp                 # High-level matrix operation wrappers
│           │   └── backend_ops.hpp         # Hardware backend dispatch & CMSIS-NN routines
│           ├── autograd/autograd.hpp       # GradTensor & AutogradNode for operation-level graph
│           ├── layers/layers.hpp           # Linear, BatchNorm, LeakyReLU, Dropout, Sequential
│           └── bridge/
│               ├── c_bridge.h              # C-API header for STM32CubeIDE
│               └── c_bridge.cpp            # C-API wrapper implementation
├── Drivers/
│   └── CMSIS-NN/                          # ARM CMSIS-NN library source files & headers
└── README.md
```

---

## 💻 Usage Examples

### 1. Model Creation & Training via C-API (`main.c`)

```c
#include "engine/bridge/c_bridge.h"

// Define network with mixed weight precisions
PredictorHandle predictor = create_predictor();

// Layer 1: Linear 5 -> 32 with INT8 quantized weights (0 = INT8)
predictor_add_layer(predictor, create_linear_layer_with_dtype(5, 32, 0));
predictor_add_layer(predictor, create_batchnorm1d_layer(32, 1e-5f, 0.1f));
predictor_add_layer(predictor, create_leaky_relu_layer(0.1f));

// Layer 2: Linear 32 -> 16 with FLOAT16 weights (3 = FLOAT16)
predictor_add_layer(predictor, create_linear_layer_with_dtype(32, 16, 3));
predictor_add_layer(predictor, create_batchnorm1d_layer(16, 1e-5f, 0.1f));
predictor_add_layer(predictor, create_leaky_relu_layer(0.1f));

// Layer 3: Linear 16 -> 1 with FLOAT32 weights (4 = FLOAT32)
predictor_add_layer(predictor, create_linear_layer_with_dtype(16, 1, 4));

// Training step loop
for (int epoch = 1; epoch <= 100; epoch++) {
    float loss = train_step(predictor, inputs, targets, batch_size, learning_rate);
}

// Inference
float prediction = predict(predictor, test_sample);

// Cleanup
destroy_predictor(predictor);
```

### 2. Enabling CMSIS-NN Acceleration Benchmark

```c
// Run training pass with standard CPU fallback
set_cmsis_nn_acceleration_enabled(0);
run_training();

// Run training pass with ARM CMSIS-NN SIMD/DSP acceleration
set_cmsis_nn_acceleration_enabled(1);
run_training();
```

---

## Benchmark Output 

```text
+------------------------------------------------------------------+
|                STM32 EDGE TENSOR ENGINE BENCHMARK                |
|        Architecture: ARM Cortex-M4 @ 180MHz (STM32F429ZI)        |
|        Quantization: INT8  |  Half-FP: FLOAT16  |  FP32          |
+------------------------------------------------------------------+

+-- [PASS 1] Standard CPU Reference Backend (No Acceleration) -----+
|  Epoch [ 20/100]  ------>  MSE Loss: 0.3660                     |
|  Epoch [ 40/100]  ------>  MSE Loss: 0.1600                     |
|  Epoch [ 60/100]  ------>  MSE Loss: 0.1252                     |
|  Epoch [ 80/100]  ------>  MSE Loss: 0.0579                     |
|  Epoch [100/100]  ------>  MSE Loss: 0.0792                     |
|  Execution Time   : 12676 ms                                      |
|  Test Prediction  : 0.3898 (Ground Truth: 0.3708)              |
+------------------------------------------------------------------+

+-- [PASS 2] ARM CMSIS-NN Hardware Acceleration Backend (SIMD) ----+
|  Epoch [ 20/100]  ------>  MSE Loss: 0.3660                     |
|  Epoch [ 40/100]  ------>  MSE Loss: 0.1600                     |
|  Epoch [ 60/100]  ------>  MSE Loss: 0.1252                     |
|  Epoch [ 80/100]  ------>  MSE Loss: 0.0579                     |
|  Epoch [100/100]  ------>  MSE Loss: 0.0792                     |
|  Execution Time   : 7088 ms                                      |
|  Test Prediction  : 0.3898 (Ground Truth: 0.3708)              |
+------------------------------------------------------------------+

+------------------------------------------------------------------+
|                   BENCHMARK EXECUTION SUMMARY                    |
+----------------------+-+--------------+-+--------------+-+-------+
| Backend Mode         | | Exec Time    | | Final MSE    | | Output|
+----------------------+-+--------------+-+--------------+-+-------+
| CPU Reference        | | 12676 ms       | | 0.0792       | | 0.3898|
| CMSIS-NN Accelerated | | 7088 ms       | | 0.0792       | | 0.3898|
+----------------------+-+--------------+-+--------------+-+-------+
```

---
