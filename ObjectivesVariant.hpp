#ifndef OBJECTIVESVARIANT_H
#define OBJECTIVESVARIANT_H

#include <variant>
#include "Objectives.h"

using ObjectiveVariant = std::variant<objectives::MeanSquaredError,
                                      objectives::CrossEntropyLoss>;

#endif // OBJECTIVESVARIANT_H