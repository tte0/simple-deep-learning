#ifndef OPTIMIZERSVARIANT_HPP
#define OPTIMIZERSVARIANT_HPP

#include <variant>
#include "Optimizers.h"

using OptimizersVariant = std::variant<optimizers::GradientDescent,
                                       optimizers::MomentumOptimizer,
                                       optimizers::AdamOptimizer>;

#endif // OPTIMIZERSVARIANT_HPP