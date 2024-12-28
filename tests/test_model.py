import pytest
import joblib
from huggingface_hub import hf_hub_download


@pytest.fixture
def model():
    REPO_ID = "h9art/MLOps_toy_model"  # Địa chỉ repo trên Hugging Face
    FILENAME = "model.pkl"  # Tên file trong repo (sửa lại theo tên đúng file mô hình của bạn)
    
    # Tải mô hình từ Hugging Face
    model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    # Tải mô hình bằng joblib
    model = joblib.load(model_path)
    return model


def test_model_prediction(model):
    """Kiểm tra dự đoán của mô hình"""
    # Ví dụ: Kiểm tra với 3 năm kinh nghiệm
    years_of_experience = [[3]]  # Dữ liệu đầu vào dưới dạng danh sách lồng nhau
    predicted_salary = model.predict(years_of_experience)

    # Kiểm tra kết quả dự đoán (Giả sử mô hình có dự đoán một mức lương cụ thể cho 3 năm)
    assert predicted_salary[0] > 0, f"Dự đoán lương không hợp lệ: {predicted_salary[0]}"


def test_model_with_multiple_inputs(model):
    """Kiểm tra với nhiều giá trị đầu vào"""
    years_of_experience = [[1], [5], [10]]  # Dữ liệu đầu vào dưới dạng danh sách lồng nhau
    predicted_salaries = model.predict(years_of_experience)
    
    assert len(predicted_salaries) == 3, f"Số lượng dự đoán không đúng: {len(predicted_salaries)}"
    assert all(salary > 0 for salary in predicted_salaries), f"Dự đoán lương không hợp lệ: {predicted_salaries}"
