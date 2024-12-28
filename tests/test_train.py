import pytest
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Đường dẫn đến file dữ liệu và mô hình đã lưu
data_path = r'C:\Users\ASUS\Desktop\ML_OP\MLOps_v2\data\Salary_Data.csv'
model_path = r'C:\Users\ASUS\Desktop\ML_OP\MLOps_v2\saved_model\model.pkl'

@pytest.fixture
def load_data():
    """Fixture để tải dữ liệu huấn luyện"""
    dataset = pd.read_csv(data_path)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    return X, y

@pytest.fixture
def train_model(load_data):
    """Fixture để huấn luyện mô hình và trả về mô hình đã huấn luyện"""
    X, y = load_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

    # Huấn luyện mô hình Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Lưu mô hình đã huấn luyện
    joblib.dump(model, model_path)
    return model

def test_train_model(train_model):
    """Kiểm tra quá trình huấn luyện mô hình"""
    # Kiểm tra xem mô hình có thể huấn luyện và lưu thành công không
    assert os.path.exists(model_path), f"Mô hình không được lưu tại {model_path}"

    # Kiểm tra mô hình có thể dự đoán với một số dữ liệu
    model = train_model
    test_input = np.array([[5]])  # 5 năm kinh nghiệm
    predicted_salary = model.predict(test_input)

    # Kiểm tra giá trị dự đoán có hợp lệ
    assert predicted_salary[0] > 0, f"Dự đoán lương không hợp lệ: {predicted_salary[0]}"

def test_model_saved_correctly():
    """Kiểm tra mô hình đã được lưu và có thể tải lại"""
    # Đảm bảo mô hình đã được lưu đúng cách
    model = joblib.load(model_path)
    assert model is not None, "Mô hình không thể tải lại"

def test_train_data(load_data):
    """Kiểm tra dữ liệu huấn luyện"""
    X, y = load_data
    assert X.shape[0] > 0, "Dữ liệu huấn luyện không có mẫu"
    assert X.shape[1] > 0, "Dữ liệu huấn luyện không có tính năng"
    assert len(y) > 0, "Dữ liệu nhãn không hợp lệ"
