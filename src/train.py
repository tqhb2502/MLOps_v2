import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import os

print(os.path.join('..', 'data', 'Salary_Data.csv'))
dataset = pd.read_csv(os.path.join('data', 'Salary_Data.csv'))
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Đường dẫn tệp
file_path = os.path.join("saved_model", "model.pkl")

# Kiểm tra tệp tồn tại
if os.path.exists(file_path):
    with open(file_path, "rb") as f:
        regressor = pickle.load(f)
    print("Model loaded successfully!")
else:
    regressor = LinearRegression()
    print("Create new model!")

regressor.fit(X_train, y_train)

# Tạo thư mục nếu chưa tồn tại
os.makedirs("saved_model", exist_ok=True)

# Save model
with open(file_path, "wb") as f:
    pickle.dump(regressor, f, protocol=5)

# Test & Ghi log
y_pred = regressor.predict(X_test)

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
