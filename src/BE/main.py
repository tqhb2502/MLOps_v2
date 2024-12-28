from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import csv
import pickle
from huggingface_hub import Repository
import os
import shutil
from huggingface_hub import hf_hub_download
import joblib


app = FastAPI()
MODEL_DIR = "../../saved_model"
MODEL_NAME = "model.pkl"
MODEL_OLD_NAME = "model_old.pkl"
SAVE_PROTOCOL = 4  # Protocol pickle (thường dùng 4 cho tương thích tốt)
# Biến đếm số lần gọi API, dùng cho việc thay đổi API pipeline trong trường hợp cần
request_count = 0
test_realse = True

# Định nghĩa các model nhận request từ frontend
class SalaryPredictionRequest(BaseModel):
    experience_years: int  # Số năm kinh nghiệm

class ColectDataRequest(BaseModel):
    experience_years: int  # Số năm kinh nghiệm
    predicted_salary: str  # Lương dự đoán

csv_file = "D:/VsCode/testFolder/MLOps_v2/data/collectdata/datacollect.csv"  # Đường dẫn CSV lưu trữ

# Hàm tải mô hình đã huấn luyện
def load_model(name:str):
    model_path = "D:/VsCode/testFolder/MLOps_v2/saved_model/"+name  # Đường dẫn đến mô hình đã huấn luyện
    with open(model_path, "rb") as model_file:
        return pickle.load(model_file)

if not os.path.exists("../../saved_model"):
    print(f"Thư mục ../../saved_model chưa tồn tại. Đang tiến hành clone...")
    repo = Repository(
        local_dir="../../saved_model",
        clone_from="h9art/MLOps_toy_model",
        use_auth_token="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"
    )
    print("Clone thành công!")
else:
    print(f"Thư mục ../../saved_model đã tồn tại. Không cần clone lại.")
# # Tải mô hình khi ứng dụng khởi động
model = load_model("model_old.pkl")
model_test = load_model("model.pkl")

test_realse = False
count = 0
# Hàm dự đoán lương dựa trên mô hình
def predict_salary(experience_years: int) -> str:
    global count  # Sử dụng biến `count` toàn cục để theo dõi số lần gọi hàm

    if test_realse:
        # Tăng biến đếm
        count += 1
        
        # Chọn mô hình dựa trên giá trị của count
        if count % 6 == 0:  # Mỗi 6 lần, sử dụng `model_test`
            chosen_model = model_test
        else:  # Các trường hợp còn lại sử dụng `model`
            chosen_model = model
    else:
        # Nếu không bật test_realse, luôn sử dụng `model`
        chosen_model = model

    # Dự đoán sử dụng mô hình đã chọn
    predicted_salary = chosen_model.predict([[experience_years]])[0]  # Giả sử mô hình yêu cầu dữ liệu dưới dạng mảng 2D
    return f"{predicted_salary} VND"

# Hàm thu thập dữ liệu và lưu vào CSV
def save_to_csv_and_dvc(experience_years: int, predicted_salary: str):
    # Loại bỏ "VND" và chuyển sang dạng float với 1 số sau dấu phẩy
    try:
        predicted_salary_float = float(predicted_salary.replace("VND", "").strip())
        predicted_salary_formatted = round(predicted_salary_float, 1)
    except ValueError:
        print(f"Lỗi: Không thể chuyển đổi '{predicted_salary}' thành dạng số.")
        return

    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(['experience_years', 'predicted_salary'])  # Tiêu đề cột

        writer.writerow([experience_years, predicted_salary_formatted])  # Ghi dữ liệu vào file

    with open(csv_file, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)

        # Kiểm tra xem có đủ 5 mẫu không
        if len(rows) >= 6:  # Bao gồm cả tiêu đề, nên cần >= 6
            print("Có đủ 5 mẫu, tiến hành huấn luyện lại mô hình.")
            # Gọi lệnh huấn luyện lại mô hình và lưu kết quả
            train_and_save_model()
        else:
            print("Chưa đủ 5 mẫu, chờ thêm dữ liệu.")

    # Thực hiện thêm, commit và đẩy dữ liệu lên DVC (nếu sử dụng DVC cho version control)
    os.system(f"git add {csv_file}")
    os.system(f"git commit -m 'Add new collected data'")
    os.system(f"dvc push")

def train_and_save_model():
    global model, model_test

    try:
        # Đọc dữ liệu từ CSV
        dataset = pd.read_csv(csv_file)
        X = dataset.iloc[:, :-1].values  # Dữ liệu đầu vào (số năm kinh nghiệm)
        y = dataset.iloc[:, -1].values  # Dữ liệu đầu ra (lương dự đoán)

        # Chia dữ liệu thành tập huấn luyện và kiểm tra
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2,  # Tỷ lệ chia dữ liệu (80% huấn luyện, 20% kiểm tra)
            random_state=42
        )

        # Đường dẫn tệp mô hình
        file_path = os.path.join(MODEL_DIR, MODEL_NAME)

        newmodel = model

        # Huấn luyện mô hình
        newmodel.fit(X_train, y_train)

       # Kiểm tra xem có mô hình cũ không và xóa nếu có
        if os.path.exists(os.path.join(MODEL_DIR, MODEL_OLD_NAME)):
            os.remove(os.path.join(MODEL_DIR, MODEL_OLD_NAME))
            print(f"Đã xóa {MODEL_OLD_NAME}.")

         # Kiểm tra xem có mô hình hiện tại không, nếu có thì đổi tên thành model_old.pkl
        if os.path.exists(os.path.join(MODEL_DIR, MODEL_NAME)):
            os.rename(os.path.join(MODEL_DIR, MODEL_NAME), os.path.join(MODEL_DIR, MODEL_OLD_NAME))
            print(f"Đã đổi tên {MODEL_NAME} thành {MODEL_OLD_NAME}.")

         # Tạo thư mục nếu chưa tồn tại
        os.makedirs(MODEL_DIR, exist_ok=True)

        # Lưu mô hình mới vào file model.pkl
        with open(os.path.join(MODEL_DIR, MODEL_NAME), "wb") as f:
            pickle.dump(newmodel, f, protocol=SAVE_PROTOCOL)

        print(f"Mô hình mới đã được lưu thành công tại {MODEL_NAME}.")   

        push_model_to_huggingface("model_old.pkl")
        push_model_to_huggingface("model.pkl")

        model = load_model("model_old.pkl")
        model_test = load_model("model.pkl")
    except Exception as e:
        print(f"Đã có lỗi xảy ra khi huấn luyện mô hình: {e}")

def push_model_to_huggingface(namemodel:str):
    # Tạo thư mục tạm nếu chưa có
    repo_local_dir = "./h9art/MLOps_toy_model"
    
    # Khởi tạo repo hoặc clone repo nếu đã tồn tại
    repo = Repository(
        local_dir=repo_local_dir, 
        clone_from="h9art/MLOps_toy_model", 
        use_auth_token="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"
    )
    
    # Xóa model cũ trong repo (nếu có)
    model_file_path = os.path.join(repo_local_dir, namemodel)
    if os.path.exists(model_file_path):
        os.remove(model_file_path)
    
    # Sao chép mô hình vào thư mục repo
    new_model_path = os.path.join("D:/VsCode/testFolder/MLOps_v2/saved_model", namemodel)
    shutil.copy(new_model_path, model_file_path)
    
    # Push lên Hugging Face
    repo.push_to_hub(commit_message=namemodel)
    print("Model pushed to https://huggingface.co/"+ namemodel)
    
# API để dự đoán lương
@app.post("/predict_salary/")
async def predict_salary_endpoint(request: SalaryPredictionRequest):
    predicted_salary = predict_salary(request.experience_years)
    return {"experience_years": request.experience_years, "predicted_salary": predicted_salary}

    # return {"experience_years": request.experience_years, "predicted_salary": predicted_salary}
# API để thu thập dữ liệu và lưu vào CSV
@app.post("/colectdata/")
async def colectdata(request: ColectDataRequest):
    print(f"Saving data: {request.experience_years} years => {request.predicted_salary} VND")
    save_to_csv_and_dvc(request.experience_years, request.predicted_salary)
    return {"status": "saved successfully"}
