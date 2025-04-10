import numpy as np
import skfuzzy as fuzzy
from skfuzzy import control

dishes_amount = control.Antecedent(np.arange(0, 100, 1), 'dishes amount')
dirtiness = control.Antecedent(np.arange(0, 100, 1), 'dirtiness')

wash_time = control.Antecedent(np.arange(0, 100, 1), 'wash time')

dishes_amount["few"] = fuzzy.trimf(dishes_amount.universe, [0, 0, 30])
dishes_amount["medium"] = fuzzy.trimf(dishes_amount.universe, [10, 30, 60])
dishes_amount["many"] = fuzzy.trimf(dishes_amount.universe, [50, 60, 100])

