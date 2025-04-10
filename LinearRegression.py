import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import floor

correlation = 0.70710679
np.random.seed(253)
x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.rand(floor(1 / correlation**2), 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=20, random_state=50)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")

plt.scatter(x_test, y_test, color="green", label="Gerçek Değerler")
plt.plot(x_test, y_pred, color="blue", label="Model Tahmini")
plt.xlabel("Bağımsız Değişken (X)")
plt.ylabel("Bağımlı Değişken (y)")
plt.legend()
plt.title("Lineer Regresyon Sonucu")
plt.show()

# 7. Model Coefficients
print(f"slope (m): {model.coef_[0][0]:.2f}")
print(f"y intercept (b): {model.intercept_[0]:.2f}")