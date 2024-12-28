import pytest
import joblib
import numpy as np

# Tải mô hình đã lưu
model_path = '../saved_models/model.pkl'  # Đường dẫn đúng

@pytest.fixture
def model():
    """Fixture để tải mô hình"""
    return joblib.load(model_path)

def test_model_prediction(model):
    """Kiểm tra dự đoán của mô hình"""
    # Ví dụ: Kiểm tra với 3 năm kinh nghiệm
    years_of_experience = np.array([[3]])  # 3 năm kinh nghiệm, format cho mô hình dự đoán
    predicted_salary = model.predict(years_of_experience)

    # Kiểm tra kết quả dự đoán (Giả sử mô hình có dự đoán một mức lương cụ thể cho 3 năm)
    assert predicted_salary[0] > 0, f"Dự đoán lương không hợp lệ: {predicted_salary[0]}"

def test_model_with_multiple_inputs(model):
    """Kiểm tra với nhiều giá trị đầu vào"""
    years_of_experience = np.array([[1], [5], [10]])  # 3 giá trị đầu vào
    predicted_salaries = model.predict(years_of_experience)
    
    assert len(predicted_salaries) == 3, f"Số lượng dự đoán không đúng: {len(predicted_salaries)}"
    assert all(salary > 0 for salary in predicted_salaries), f"Dự đoán lương không hợp lệ: {predicted_salaries}"
