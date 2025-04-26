#ifndef ACTIVATIONSVARIANT_H
#define ACTIVATIONSVARIANT_H

#include <variant>
#include "Activations.h"

using ActivationVariant = std::variant< activations::Sigmoid,
                                        activations::ReLU,
                                        activations::LeakyReLU,
                                        activations::tanh,
                                        activations::ELU,
                                        activations::Swish>;

#endif // ACTIVATIONSVARIANT_H