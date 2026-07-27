#ifndef ENGINE_OPS_HPP_
#define ENGINE_OPS_HPP_

#include "backend_ops.hpp"

namespace engine {

// Re-export backend kernel functions into engine namespace
using backend::matmul;
using backend::matmul_AT_B;
using backend::matmul_A_BT;
using backend::add_bias;
using backend::apply_leaky_relu;
using backend::leaky_relu_backward;
using backend::apply_dropout;
using backend::dropout_backward;

} // namespace engine

#endif // ENGINE_OPS_HPP_
