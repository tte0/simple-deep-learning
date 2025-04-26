#ifndef INITIALIZERSVARIANT_H
#define INITIALIZERSVARIANT_H

#include <variant>
#include "Initializers.h"

using InitializerVariant = std::variant<initializers::RandomInitializer,
                                        initializers::HeInitializer>;

#endif // INITIALIZERSVARIANT_H