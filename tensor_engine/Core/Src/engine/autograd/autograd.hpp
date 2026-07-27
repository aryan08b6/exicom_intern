#ifndef ENGINE_AUTOGRAD_HPP_
#define ENGINE_AUTOGRAD_HPP_

#include "../array_api/array_api.hpp"

namespace engine {

struct AutogradNode {
    virtual ~AutogradNode() {}
};

class GradTensor {
public:
    Matrix* data;
    Matrix* grad;
    AutogradNode* creator;

    GradTensor(Matrix* value) : data(value), grad(nullptr), creator(nullptr) {}
    ~GradTensor() {
        delete data;
        delete grad;
        delete creator;
    }
};

} // namespace engine

#endif // ENGINE_AUTOGRAD_HPP_
