from fastapi import FastAPI
from pydantic import BaseModel
import os
import csv
import pickle

app = FastAPI()

# Biến đếm số lần gọi API, dùng cho việc thay đổi API pipeline trong trường hợp cần
request_count = 0
test_realse = True

# Định nghĩa các model nhận request từ frontend
class SalaryPredictionRequest(BaseModel):
    experience_years: int  # Số năm kinh nghiệm

class ColectDataRequest(BaseModel):
    experience_years: int  # Số năm kinh nghiệm
    predicted_salary: str  # Lương dự đoán

# Hàm tải mô hình đã huấn luyện

# def load_model():
#     model_path = "D:/VsCode/projectMLOp/MLOps_v2/saved_model/model.pkl"  # Đường dẫn đến mô hình đã huấn luyện
#     with open(model_path, "rb") as model_file:
#         return pickle.load(model_file)

# # Tải mô hình khi ứng dụng khởi động
# model = load_model()

# Hàm dự đoán lương dựa trên mô hình
def predict_salary(experience_years: int) -> str:
    # Dự đoán sử dụng mô hình
    # predicted_salary = model.predict([[experience_years]])[0]  # Giả sử mô hình yêu cầu dữ liệu dưới dạng mảng 2D
    # return f"{predicted_salary} VND"
    return 0

# Hàm thu thập dữ liệu và lưu vào CSV
def save_to_csv_and_dvc(experience_years: int, predicted_salary: str):
    csv_file = "D:/VsCode/MLOpsFinal/data/datacollect.csv"  # Đường dẫn CSV lưu trữ
    
    file_exists = os.path.isfile(csv_file)
    
    with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow(['experience_years', 'predicted_salary'])  # Tiêu đề cột
        
        writer.writerow([experience_years, predicted_salary])  # Ghi dữ liệu vào file
    
    # Thực hiện thêm, commit và đẩy dữ liệu lên DVC (nếu sử dụng DVC cho version control)
    os.system(f"git add {csv_file}")
    os.system(f"git commit -m 'Add new collected data'")
    os.system(f"dvc push")

# API để dự đoán lương
@app.post("/predict_salary/")
async def predict_salary_endpoint(request: SalaryPredictionRequest):
    predicted_salary = predict_salary(request.experience_years)
    return {"experience_years": request.experience_years, "predicted_salary": predicted_salary}

# API để thu thập dữ liệu và lưu vào CSV
@app.post("/colectdata/")
async def colectdata(request: ColectDataRequest):
    print(f"Saving data: {request.experience_years} years => {request.predicted_salary} VND")
    save_to_csv_and_dvc(request.experience_years, request.predicted_salary)
    return {"status": "saved successfully"}
